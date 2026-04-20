# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import rich
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
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    MixingRule,
    make_lennard_jones_from_state,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class LjConfig(BaseModel):
    tail_correction: bool
    cutoff: float
    parameters: dict[str, tuple[float | None, float | None]]
    mixing_rule: MixingRule


class Config(BaseModel):
    run: MdRunConfig
    md: MdParameters
    lj: LjConfig
    inp_file: str | Path


@dataclass
class LjMdState:
    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters
    step: Array
    lj_parameters: LennardJonesParameters

    @property
    def neighborlist(self) -> NearestNeighborList:
        return DenseNearestNeighborList.from_state(self)


def init_state(key: Array, config: Config) -> LjMdState:
    lj_params = LennardJonesParameters.from_dict(
        cutoff=config.lj.cutoff,
        parameters=config.lj.parameters,
        mixing_rule=config.lj.mixing_rule,
    )
    mb_key = key if config.md.initialize_momenta else None
    particles, systems = md_state_from_ase(config.inp_file, config.md, key=mb_key)
    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, lj_params.cutoff
    )
    return LjMdState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        step=jnp.array([0]),
        lj_parameters=lj_params,
    )


def run(config: Config) -> None:
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))
    state = init_state(next(chain), config)
    state_lens = identity_lens(LjMdState)
    potential = make_lennard_jones_from_state(
        state_lens, compute_position_and_unitcell_gradients=True
    )
    propagator = make_md_propagator(state_lens, config.md.integrator, potential)
    state = run_md(next(chain), propagator, state, config.run)


def main() -> None:
    cli = NanoArgs(Config)
    config = cli.parse()
    run(config)
    rich.print(analyze_md_file(config.run.out_file))


if __name__ == "__main__":
    main()
