# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unified molecular-dynamics entry point.

The force field is selected by the ``potential`` config (LJ / LJ plus Coulomb /
tojax-MLFF / torch-MLFF). The potential is built directly with its parameters, so
a single force-field-agnostic ``MdState`` and driver serve every backend.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # for torch backends

import jax
import rich
import rich.logging
from nanoargs import NanoArgs
from pydantic import BaseModel

from kups.application.md.analysis import analyze_md_file
from kups.application.md.data import (
    MdParameters,
    MDParticles,
    MdRunConfig,
    MdState,
    MDSystems,
    md_state_from_ase,
)
from kups.application.md.simulation import make_md_propagator, run_md
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.simulations.potentials import (
    MaceConfig,
    PotentialConfig,
    TojaxPotentialConfig,
    UmaConfig,
)
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import key_chain

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich.logging.RichHandler()],
)


# optimal block averaging needs at least 4 blocks of 2 rows
_ANALYSIS_ROWS = 8


class Config(BaseModel):
    """Top-level configuration for a molecular-dynamics run."""

    run: MdRunConfig
    md: MdParameters
    potential: PotentialConfig
    inp_files: tuple[str | Path, ...]


def run(config: Config) -> MdState:
    """Run a molecular-dynamics simulation for the configured force field.

    Args:
        config: Run, MD, potential and input configuration.

    Returns:
        The final state.
    """
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))
    state_lens = identity_lens(MdState)

    all_particles: list[Table[ParticleId, MDParticles]] = []
    all_systems: list[Table[SystemId, MDSystems]] = []
    for inp_file in config.inp_files:
        logging.info(f"Loading structure from {inp_file}")
        key = next(chain) if config.md.initialize_momenta else None
        particles_i, systems_i = md_state_from_ase(
            inp_file,
            config.md,
            key=key,
            index_dtype=config.run.index_dtype,
            float_dtype=config.run.float_dtype,
        )
        all_particles.append(particles_i)
        all_systems.append(systems_i)
    particles, systems = Table.union(all_particles, all_systems)
    potential, cutoff = config.potential.build(
        state_lens, POSITIONS_AND_CELL, particles, systems
    )
    propagator = make_md_propagator(state_lens, config.md.integrator, potential)

    base = 1 if isinstance(config.potential, TojaxPotentialConfig) else 2
    multiplier = 2.0 if isinstance(config.potential, MaceConfig | UmaConfig) else 1.0
    state = MdState.create(
        particles, systems, cutoff, config.md.skin, base=base, multiplier=multiplier
    )
    return run_md(next(chain), propagator, state, config.run)


def main() -> None:
    """CLI entry point for molecular-dynamics simulations."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rows = config.run.num_steps // config.run.thermo_every
    if rows < _ANALYSIS_ROWS:
        logging.warning(
            f"Not analyzed: {rows} rows, the analysis needs {_ANALYSIS_ROWS}."
        )
        return
    rich.print(analyze_md_file(config.run.out_file))


if __name__ == "__main__":
    main()
