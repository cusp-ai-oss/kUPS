# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Domain-decomposed Lennard-Jones structure relaxation entry point.

Parallel to :mod:`kups.application.simulations.relax` (LJ backend), but every
optimizer step runs under a ``shard_map`` over the ``OriginDeviceId`` mesh:
each device holds all atoms and builds only its owned-incident edge shard (see
:mod:`kups.core.domain`), while the unchanged relaxation propagator, run loop,
convergence check, HDF5 logging, and capacity-resize machinery operate on the
fully replicated state. Forces come out full and replicated, so the trajectory
matches a single-device run to tight tolerance (identical up to floating-point
summation order). The only differences from
:mod:`kups.application.simulations.relax` are the origin-bearing particles/state,
the sharded LJ potential, and the shard-mapped propagator wrapper.
"""

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

from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_ONLY
from kups.application.relaxation.analysis import analyze_relax_file
from kups.application.relaxation.data import (
    RelaxParticles,
    RelaxRunConfig,
    RelaxSystems,
    relax_state_from_ase,
)
from kups.application.relaxation.simulation import make_relax_propagator, run_relax
from kups.application.simulations._domain_decomposition import (
    ShardMappedPropagator,
    load_and_partition,
    make_sharded_lj_potential,
    mesh_max_cell_list_view,
    origin_mesh,
)
from kups.application.simulations.potentials import LjPotentialConfig
from kups.core.capacity import FixedCapacity
from kups.core.data import Index, Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.sharding import device_put_replicated
from kups.core.typing import OriginDeviceId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.lennard_jones import LennardJonesParameters
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
    """Top-level configuration for a domain-decomposed LJ relaxation run.

    Mirrors :class:`kups.application.simulations.relax.Config`, restricted to
    the LJ backend (the DD potential is hand-wired around the sharded graph
    constructor, so the discriminated ``potential`` union does not apply).
    """

    run: RelaxRunConfig
    potential: LjPotentialConfig
    inp_files: tuple[str | Path, ...]


@dataclass
class RelaxParticlesDD(RelaxParticles):
    """Relaxation particles tagged with their owner device for domain decomposition.

    ``origin`` is a real field (legal as a required field:
    ``RelaxParticles.exclusion`` is keyword-only); ``inclusion`` stays the
    inherited derived property.
    """

    origin: Index[OriginDeviceId]


@dataclass
class RelaxLjDDState:
    """Mirrors :class:`kups.application.relaxation.data.RelaxState`; particles carry ``origin``."""

    particles: Table[ParticleId, RelaxParticlesDD]
    systems: Table[SystemId, RelaxSystems]
    neighborlist_params: UniversalNeighborlistParameters
    opt_state: optax.OptState
    step: Array


def init_state(
    config: Config, n_devices: int
) -> tuple[
    Table[ParticleId, RelaxParticlesDD],
    Table[SystemId, RelaxSystems],
    UniversalNeighborlistParameters,
    FixedCapacity[int],
    LennardJonesParameters,
]:
    """Load and partition the DD relaxation inputs.

    Returns the state's pieces rather than the state itself: ``opt_state`` needs
    the optimizer ``run`` builds, so ``run`` assembles the state once.

    Returns:
        Tuple of the origin-tagged particles, the systems, the neighbor-list
        parameters, the owned capacity, and the LJ parameters.
    """
    lj = LennardJonesParameters.from_dict(
        cutoff=config.potential.cutoff,
        parameters=config.potential.parameters,
        mixing_rule=config.potential.mixing_rule,
    )
    particles, systems, neighborlist_params, cap_owned = load_and_partition(
        config.inp_files,
        relax_state_from_ase,
        RelaxParticlesDD,
        lj.cutoff,
        n_devices,
    )
    return particles, systems, neighborlist_params, cap_owned, lj


def run(config: Config, mesh: jax.sharding.Mesh | None = None) -> RelaxLjDDState:
    """Run a domain-decomposed LJ structure relaxation.

    Args:
        config: Run configuration.
        mesh: Device mesh to decompose over; all local devices by default.
    """
    key = jax.random.key(config.run.seed or time.time_ns())
    state_lens = identity_lens(RelaxLjDDState)
    mesh = mesh if mesh is not None else origin_mesh()
    optimizer = make_optimizer(config.run.optimizer)
    gradient = FRECHET_FILTER if config.run.optimize_cell else POSITIONS_ONLY

    particles, systems, neighborlist_params, cap_owned, lj = init_state(
        config, mesh.size
    )

    neighborlist = mesh_max_cell_list_view(
        state_lens.focus(lambda s: s.neighborlist_params), lj.cutoff
    )
    potential = make_sharded_lj_potential(
        state_lens, lj, neighborlist, cap_owned, gradient
    )
    propagator, opt_init = make_relax_propagator(
        state_lens, potential, optimizer, gradient
    )
    state = RelaxLjDDState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        opt_state=opt_init(particles, systems),
        step=jnp.array([0]),
    )
    state = device_put_replicated(state, mesh)
    return run_relax(key, ShardMappedPropagator(propagator, mesh), state, config.run)


def main() -> None:
    """CLI entry point for domain-decomposed LJ structure relaxation."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_relax_file(config.run.out_file))


if __name__ == "__main__":
    main()
