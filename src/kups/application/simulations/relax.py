# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unified structure-relaxation entry point.

The force field is selected by the ``potential`` config (LJ / LJ plus Coulomb /
tojax-MLFF / torch-MLFF). The potential is built directly with its parameters, so
a single force-field-agnostic ``RelaxState`` and driver serve every backend.
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

from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_ONLY
from kups.application.relaxation.analysis import analyze_relax_file
from kups.application.relaxation.data import (
    RelaxParticles,
    RelaxRunConfig,
    RelaxState,
    RelaxSystems,
    relax_state_from_ase,
)
from kups.application.relaxation.simulation import make_relax_propagator, run_relax
from kups.application.simulations.potentials import (
    MaceConfig,
    PotentialConfig,
    UmaConfig,
)
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.typing import ParticleId, SystemId
from kups.relaxation.config import make_optimizer

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich.logging.RichHandler()],
)


class Config(BaseModel):
    """Top-level configuration for a structure relaxation run."""

    run: RelaxRunConfig
    potential: PotentialConfig
    inp_files: tuple[str | Path, ...]


def run(config: Config) -> RelaxState:
    """Run a structure relaxation for the configured force field.

    Args:
        config: Run, potential and input configuration.

    Returns:
        The final state.
    """
    seed = time.time_ns() if config.run.seed is None else config.run.seed
    key = jax.random.key(seed)
    state_lens = identity_lens(RelaxState)
    optimizer = make_optimizer(config.run.optimizer)
    gradient = FRECHET_FILTER if config.run.optimize_cell else POSITIONS_ONLY

    all_particles: list[Table[ParticleId, RelaxParticles]] = []
    all_systems: list[Table[SystemId, RelaxSystems]] = []
    for inp_file in config.inp_files:
        logging.info(f"Loading structure from {inp_file}")
        particles_i, systems_i = relax_state_from_ase(inp_file)
        all_particles.append(particles_i)
        all_systems.append(systems_i)
    particles, systems = Table.union(all_particles, all_systems)
    potential, cutoff = config.potential.build(state_lens, gradient, particles, systems)
    propagator, opt_init = make_relax_propagator(
        state_lens, potential, optimizer, gradient
    )

    # Torch MLFF models need extra neighbor-list capacity headroom.
    multiplier = 2.0 if isinstance(config.potential, MaceConfig | UmaConfig) else 1.0
    opt_state = opt_init(particles, systems)
    state = RelaxState.create(
        particles,
        systems,
        cutoff,
        config.run.skin,
        opt_state,
        multiplier=multiplier,
    )
    logging.info("Starting relaxation")
    return run_relax(key, propagator, state, config.run)


def main() -> None:
    """CLI entry point for structure relaxation."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_relax_file(config.run.out_file))


if __name__ == "__main__":
    main()
