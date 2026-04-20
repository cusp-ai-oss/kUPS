# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Lennard-Jones structure relaxation entry point."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import rich
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
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    NearestNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    MixingRule,
    make_lennard_jones_from_state,
)
from kups.relaxation.optax import make_optimizer

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class LjConfig(BaseModel):
    """Lennard-Jones potential parameters."""

    tail_correction: bool
    cutoff: float
    parameters: dict[str, tuple[float | None, float | None]]
    mixing_rule: MixingRule


class Config(BaseModel):
    """Top-level configuration for LJ relaxation."""

    run: RelaxRunConfig
    relax: RelaxParameters
    lj: LjConfig
    inp_file: str | Path


@dataclass
class RelaxLjState:
    """Simulation state for Lennard-Jones relaxation."""

    particles: Table[ParticleId, RelaxParticles]
    systems: Table[SystemId, RelaxSystems]
    neighborlist_params: UniversalNeighborlistParameters
    opt_state: optax.OptState
    step: Array
    lj_parameters: LennardJonesParameters

    @property
    def neighborlist(self) -> NearestNeighborList:
        return DenseNearestNeighborList.from_state(self)


def init_state(config: Config, opt_init: OptInit) -> RelaxLjState:
    """Initialise an LJ relaxation state from configuration.

    Args:
        config: Simulation configuration.
        opt_init: Optimizer state initialiser from the relaxation propagator.

    Returns:
        Fully constructed LJ relaxation state.
    """
    lj_params = LennardJonesParameters.from_dict(
        cutoff=config.lj.cutoff,
        parameters=config.lj.parameters,
        mixing_rule=config.lj.mixing_rule,
    )
    particles, systems = relax_state_from_ase(config.inp_file)

    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, lj_params.cutoff
    )
    opt_state = opt_init(
        (particles.data.positions, systems.data.unitcell.lattice_vectors)
    )
    return RelaxLjState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        opt_state=opt_state,
        step=jnp.array([0]),
        lj_parameters=lj_params,
    )


def run(config: Config) -> None:
    """Run an LJ structure relaxation from the given configuration.

    Args:
        config: Simulation configuration.
    """
    key = jax.random.key(config.run.seed or time.time_ns())
    state_lens = identity_lens(RelaxLjState)
    optimizer = make_optimizer(config.relax.optimizer)
    potential = make_lennard_jones_from_state(
        state_lens, compute_position_and_unitcell_gradients=True
    )
    propagator, opt_init = make_relax_propagator(
        state_lens, potential, optimizer, config.relax.optimize_cell
    )
    state = init_state(config, opt_init)
    logging.info("Starting relaxation")
    run_relax(key, propagator, state, config.run)


def main() -> None:
    """CLI entry point for LJ relaxation."""
    cli = NanoArgs(Config)
    config = cli.parse()
    run(config)
    rich.print(analyze_relax_file(config.run.out_file))


if __name__ == "__main__":
    main()
