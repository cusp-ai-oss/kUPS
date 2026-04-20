# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MLFF molecular dynamics simulation entry point."""

from __future__ import annotations

import logging
import time
from pathlib import Path

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
    MDSystems,
    md_state_from_ase,
)
from kups.application.md.simulation import make_md_propagator, run_md
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    NearestNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, key_chain
from kups.potential.mliap.tojax import TojaxedMliap, make_tojaxed_from_state

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
    """Top-level configuration for MLFF MD simulations."""

    run: MdRunConfig
    md: MdParameters
    inp_files: tuple[str | Path, ...]
    model_path: str | Path


@dataclass
class MlffMdState:
    """Simulation state for MLFF MD."""

    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters
    step: Array
    jaxified_model: TojaxedMliap

    @property
    def neighborlist(self) -> NearestNeighborList:
        return DenseNearestNeighborList.from_state(self)


def init_state(key: Array, config: Config) -> MlffMdState:
    """Initialise an MLFF MD state from configuration.

    Args:
        key: JAX PRNG key for momenta initialisation.
        config: Simulation configuration.

    Returns:
        Fully constructed MLFF MD state.
    """
    jaxified_model = TojaxedMliap.from_zip_file(config.model_path)
    mb_key = key if config.md.initialize_momenta else None
    all_particles, all_systems = [], []
    for inp_file in config.inp_files:
        particles_i, systems_i = md_state_from_ase(inp_file, config.md, key=mb_key)
        all_particles.append(particles_i)
        all_systems.append(systems_i)
    particles, systems = Table.union(all_particles, all_systems)
    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, jaxified_model.cutoff, base=1
    )
    return MlffMdState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        jaxified_model=jaxified_model,
        step=jnp.array([0]),
    )


def run(config: Config) -> None:
    """Run an MLFF MD simulation from the given configuration.

    Args:
        config: Simulation configuration.
    """
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))
    state = init_state(next(chain), config)
    state_lens = identity_lens(MlffMdState)
    potential = make_tojaxed_from_state(
        state_lens, compute_position_and_unitcell_gradients=True
    )
    propagator = make_md_propagator(state_lens, config.md.integrator, potential)
    state = run_md(next(chain), propagator, state, config.run)


def main() -> None:
    """CLI entry point for MLFF MD simulations."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_md_file(config.run.out_file))


if __name__ == "__main__":
    main()
