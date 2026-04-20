# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MLFF structure relaxation entry point."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import rich
import rich.logging
from jax import Array
from nanoargs import NanoArgs
from pydantic import BaseModel

from kups.application.relaxation.analysis import analyze_relax_file
from kups.application.relaxation.data import (
    RelaxParameters,
    RelaxParticles,
    RelaxRunConfig,
    RelaxSystems,
    relax_state_from_ase,
)
from kups.application.relaxation.simulation import (
    OptInit,
    make_relax_propagator,
    run_relax,
)
from kups.application.utils.path import get_model_path
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    NearestNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.mliap.tojax import TojaxedMliap, make_tojaxed_from_state
from kups.relaxation.optax import make_optimizer

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
    run: RelaxRunConfig
    relax: RelaxParameters
    inp_files: tuple[str | Path, ...]
    model_path: str | Path


@dataclass
class RelaxMlffState:
    particles: Table[ParticleId, RelaxParticles]
    systems: Table[SystemId, RelaxSystems]
    neighborlist_params: UniversalNeighborlistParameters
    opt_state: optax.OptState
    step: Array
    jaxified_model: TojaxedMliap

    @property
    def neighborlist(self) -> NearestNeighborList:
        return DenseNearestNeighborList.from_state(self)


def init_state(config: Config, opt_init: OptInit) -> RelaxMlffState:
    """Initialize relaxation state from config.

    Args:
        config: Run configuration.
        opt_init: Optimizer initializer.

    Returns:
        Initial relaxation state.
    """
    model_path = get_model_path(config.model_path)
    jaxified_model = TojaxedMliap.from_zip_file(model_path)
    all_particles, all_systems = [], []
    for inp_file in config.inp_files:
        logging.info(f"Loading structure from {inp_file}")
        particles_i, systems_i = relax_state_from_ase(inp_file)
        all_particles.append(particles_i)
        all_systems.append(systems_i)
    particles, systems = Table.union(all_particles, all_systems)
    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, jaxified_model.cutoff
    )
    opt_state = opt_init(
        (particles.data.positions, systems.data.unitcell.lattice_vectors)
    )
    return RelaxMlffState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        opt_state=opt_state,
        step=jnp.array([0]),
        jaxified_model=jaxified_model,
    )


def run(config: Config) -> None:
    """Run structure relaxation.

    Args:
        config: Full run configuration.
    """
    key = jax.random.key(config.run.seed or time.time_ns())
    state_lens = identity_lens(RelaxMlffState)
    optimizer = make_optimizer(config.relax.optimizer)
    potential = make_tojaxed_from_state(
        state_lens, compute_position_and_unitcell_gradients=True
    )
    propagator, opt_init = make_relax_propagator(
        state_lens, potential, optimizer, config.relax.optimize_cell
    )
    state = init_state(config, opt_init)
    logging.info("Starting relaxation")
    run_relax(key, propagator, state, config.run)


def main() -> None:
    """CLI entry point for MLFF relaxation."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_relax_file(config.run.out_file))


if __name__ == "__main__":
    main()
