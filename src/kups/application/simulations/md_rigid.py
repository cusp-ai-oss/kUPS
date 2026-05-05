# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body molecular dynamics CLI entry point.

Drives NVE / NVT (BAOAB Langevin or CSVR) / NPT MD for systems of rigid
molecules (TIP4P/2005 water, CO₂, …). The atom-level potential is composed
of Lennard-Jones + Ewald summation in atomic units; per-step force/torque
aggregation onto group COMs is handled by the rigid integrator factories.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import rich
from jax import Array
from nanoargs import NanoArgs
from pydantic import BaseModel

from kups.application.mcmc.data import AdsorbateConfig, MotifParticles
from kups.application.md.rigid_analysis import analyze_rigid_md_file
from kups.application.md.data import (
    MDRigidGroup,
    MDRigidParticles,
    MDSystems,
    MdRunConfig,
    RigidMdParameters,
    build_rigid_state_from_grid,
)
from kups.application.md.simulation import make_rigid_md_propagator, run_rigid_md
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    NearestNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.potential import sum_potentials
from kups.core.typing import GroupId, MotifParticleId, ParticleId, SystemId
from kups.core.utils.jax import dataclass, key_chain
from kups.potential.classical.ewald import EwaldParameters, make_ewald_from_state
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
    MixingRule,
    make_lennard_jones_from_state,
    make_lennard_jones_tail_correction_from_state,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class LjConfig(BaseModel):
    """Lennard-Jones potential configuration."""

    cutoff: float
    parameters: dict[str, tuple[float | None, float | None]]
    mixing_rule: MixingRule
    tail_correction: bool = False
    """If True, add the analytical long-range LJ tail correction. Required for
    quantitative density at typical (8-10 Å) cutoffs against canonical TIP4P/2005."""


class EwaldConfig(BaseModel):
    """Ewald summation configuration."""

    real_cutoff: float
    precision: float


class BoxConfig(BaseModel):
    """Cubic box specification."""

    edge_length: float
    """Edge of a cubic box in Å. Use the same value for all three axes."""


class Config(BaseModel):
    """Top-level configuration for rigid-body MD simulations."""

    adsorbates: tuple[AdsorbateConfig, ...]
    n_molecules: tuple[int, ...]
    box: BoxConfig
    run: MdRunConfig
    md: RigidMdParameters
    lj: LjConfig
    ewald: EwaldConfig


@dataclass
class RigidMdState:
    """Full state for a rigid-body MD simulation."""

    particles: Table[ParticleId, MDRigidParticles]
    groups: Table[GroupId, MDRigidGroup]
    motifs: Table[MotifParticleId, MotifParticles]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters
    step: Array
    lj_parameters: GlobalTailCorrectedLennardJonesParameters
    ewald_parameters: EwaldParameters

    @property
    def neighborlist(self) -> NearestNeighborList:
        return DenseNearestNeighborList.from_state(self)

    @property
    def max_cutoff(self) -> Table[SystemId, Array]:
        """Per-system maximum cutoff across LJ and Ewald."""
        return Table(
            self.systems.keys,
            jnp.maximum(self.lj_parameters.cutoff.data, self.ewald_parameters.cutoff.data),
        )


def init_state(key: Array, config: Config) -> RigidMdState:
    chain = key_chain(key)
    edge = config.box.edge_length
    box_size: tuple[float, float, float] = (edge, edge, edge)
    particles, groups, motifs, systems = build_rigid_state_from_grid(
        next(chain),
        config.adsorbates,
        config.n_molecules,
        box_size,
        config.md,
    )
    lj_params = GlobalTailCorrectedLennardJonesParameters.from_dict(
        cutoff=config.lj.cutoff,
        parameters=config.lj.parameters,
        mixing_rule=config.lj.mixing_rule,
        tail_correction=config.lj.tail_correction,
    )
    ewald_params = EwaldParameters.make(
        particles,
        systems,
        epsilon_total=config.ewald.precision,
        real_cutoff=config.ewald.real_cutoff,
    )
    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts,
        systems,
        Table(
            systems.keys,
            jnp.maximum(lj_params.cutoff.data, ewald_params.cutoff.data),
        ),
    )
    return RigidMdState(
        particles=particles,
        groups=groups,
        motifs=motifs,
        systems=systems,
        neighborlist_params=neighborlist_params,
        step=jnp.array([0]),
        lj_parameters=lj_params,
        ewald_parameters=ewald_params,
    )


def run(config: Config) -> RigidMdState:
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))
    state = init_state(next(chain), config)
    state_lens = identity_lens(RigidMdState)
    potential = sum_potentials(
        make_ewald_from_state(
            state_lens,
            compute_position_and_unitcell_gradients=True,
            include_exclusion_mask=True,
        ),
        make_lennard_jones_from_state(
            state_lens, compute_position_and_unitcell_gradients=True
        ),
        make_lennard_jones_tail_correction_from_state(
            state_lens, compute_position_and_unitcell_gradients=True
        ),
    )
    propagator = make_rigid_md_propagator(state_lens, config.md.integrator, potential)
    state = run_rigid_md(next(chain), propagator, state, config.run)
    return state


def main() -> None:
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_rigid_md_file(config.run.out_file))


if __name__ == "__main__":
    main()
