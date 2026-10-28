# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unified molecular-dynamics entry point.

The force field is selected by the ``potential`` config (LJ / tojax-MLFF /
torch-MLFF). The potential is built directly with its parameters, so a single
force-field-agnostic ``MdState`` and driver serve every backend.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # for torch backends

import jax
import jax.numpy as jnp
import rich
import rich.logging
from jax import Array
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
from kups.application.utils.particles import fixed_atoms_mask
from kups.core.data import Table
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
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


class Config(BaseModel):
    """Top-level configuration for a molecular-dynamics run."""

    run: MdRunConfig
    md: MdParameters
    potential: PotentialConfig
    inp_files: tuple[str | Path, ...]
    fixed_atoms: tuple[tuple[int, ...], ...] | None = None
    """Atoms held fixed (ASE ``FixAtoms``): one index tuple per entry of
    ``inp_files``, indexing that file's atoms. ``None`` leaves every atom free."""


def run(config: Config) -> None:
    """Run a molecular-dynamics simulation for the configured force field."""
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))
    state_lens = identity_lens(MdState)
    potential, cutoff = config.potential.build(state_lens, POSITIONS_AND_CELL)

    fixed_atoms = config.fixed_atoms or ((),) * len(config.inp_files)
    if len(fixed_atoms) != len(config.inp_files):
        raise ValueError(
            f"fixed_atoms has {len(fixed_atoms)} entries for "
            f"{len(config.inp_files)} input files"
        )
    mb_key = next(chain) if config.md.initialize_momenta else None
    all_particles: list[Table[ParticleId, MDParticles]] = []
    all_systems: list[Table[SystemId, MDSystems]] = []
    masks: list[Array] = []
    for inp_file, indices in zip(config.inp_files, fixed_atoms, strict=True):
        logging.info(f"Loading structure from {inp_file}")
        particles_i, systems_i = md_state_from_ase(inp_file, config.md, key=mb_key)
        all_particles.append(particles_i)
        all_systems.append(systems_i)
        masks.append(fixed_atoms_mask(particles_i.data.positions.shape[0], indices))
    particles, systems = Table.union(all_particles, all_systems)

    fixed = jnp.concatenate(masks)
    any_fixed = bool(fixed.any())
    if any_fixed:
        particles = bind(particles, lambda p: p.data.momenta).set(
            jnp.where(fixed[:, None], 0.0, particles.data.momenta)
        )
    propagator = make_md_propagator(
        state_lens, config.md.integrator, potential, fixed if any_fixed else None
    )

    base = 1 if isinstance(config.potential, TojaxPotentialConfig) else 2
    multiplier = 2.0 if isinstance(config.potential, MaceConfig | UmaConfig) else 1.0
    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, cutoff, base=base, multiplier=multiplier
    )
    state = MdState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        step=jnp.array([0]),
    )
    run_md(next(chain), propagator, state, config.run)


def main() -> None:
    """CLI entry point for molecular-dynamics simulations."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_md_file(config.run.out_file))


if __name__ == "__main__":
    main()
