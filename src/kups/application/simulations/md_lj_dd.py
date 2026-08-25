# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Domain-decomposed Lennard-Jones molecular dynamics entry point.

Parallel to :mod:`kups.application.simulations.md` (LJ backend), but every
integrator step runs under a ``shard_map`` over the ``OriginDeviceId`` mesh:
each device holds all atoms and builds only its owned-incident edge shard (see
:mod:`kups.core.domain`), while the unchanged MD propagator, run loop, HDF5
logging, and capacity-resize machinery operate on the fully replicated state.
Stochastic thermostats draw from the replicated PRNG key, so every device
samples identical noise and the trajectory matches a single-device run to tight
tolerance (identical up to floating-point summation order). The only differences
from :mod:`kups.application.simulations.md` are the origin-bearing
particles/state, the sharded LJ potential, and the shard-mapped propagator
wrapper.
"""

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
from kups.application.potential.filter import POSITIONS_AND_CELL
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
from kups.core.utils.jax import dataclass, key_chain
from kups.potential.classical.lennard_jones import LennardJonesParameters

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
    """Top-level configuration for a domain-decomposed LJ MD run.

    Mirrors :class:`kups.application.simulations.md.Config`, restricted to the
    LJ backend (the DD potential is hand-wired around the sharded graph
    constructor, so the discriminated ``potential`` union does not apply).
    """

    run: MdRunConfig
    md: MdParameters
    potential: LjPotentialConfig
    inp_files: tuple[str | Path, ...]


@dataclass
class MDParticlesDD(MDParticles):
    """MD particles tagged with their owner device for domain decomposition.

    ``origin`` is a real field (legal as a required field:
    ``MDParticles.exclusion`` is keyword-only); ``inclusion`` stays the
    inherited derived property.
    """

    origin: Index[OriginDeviceId]


@dataclass
class LjMdStateDD:
    """Mirrors :class:`kups.application.md.data.MdState`; particles carry ``origin``."""

    particles: Table[ParticleId, MDParticlesDD]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters
    step: Array


def init_state(
    key: Array | None, config: Config, n_devices: int
) -> tuple[LjMdStateDD, FixedCapacity[int], LennardJonesParameters]:
    """Build the DD MD state; also return the owned capacity and LJ parameters."""
    lj = LennardJonesParameters.from_dict(
        cutoff=config.potential.cutoff,
        parameters=config.potential.parameters,
        mixing_rule=config.potential.mixing_rule,
    )
    particles, systems, neighborlist_params, cap_owned = load_and_partition(
        config.inp_files,
        lambda inp_file: md_state_from_ase(inp_file, config.md, key=key),
        MDParticlesDD,
        lj.cutoff,
        n_devices,
    )
    state = LjMdStateDD(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        step=jnp.array([0]),
    )
    return state, cap_owned, lj


def run(config: Config, mesh: jax.sharding.Mesh | None = None) -> LjMdStateDD:
    """Run a domain-decomposed LJ molecular-dynamics simulation.

    Args:
        config: Run configuration.
        mesh: Device mesh to decompose over; all local devices by default.
    """
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))
    mesh = mesh if mesh is not None else origin_mesh()

    mb_key = next(chain) if config.md.initialize_momenta else None
    state, cap_owned, lj = init_state(mb_key, config, mesh.size)
    return run_from_state(next(chain), state, cap_owned, lj, config, mesh)


def run_from_state(
    key: Array,
    state: LjMdStateDD,
    cap_owned: FixedCapacity[int],
    lj: LennardJonesParameters,
    config: Config,
    mesh: jax.sharding.Mesh,
) -> LjMdStateDD:
    """Wire the sharded potential and shard-mapped step around ``state``, then run.

    The seam ``run`` uses after ``init_state``, so a caller that has to adjust
    the freshly built state (its neighbor-list estimate, say) can do so first.

    Args:
        key: PRNG key for the run loop.
        state: Initial replicated-on-host DD state.
        cap_owned: Owned-buffer capacity from ``init_state``.
        lj: Lennard-Jones parameters from ``init_state``.
        config: Run configuration.
        mesh: Device mesh to decompose over.
    """
    state_lens = identity_lens(LjMdStateDD)
    neighborlist = mesh_max_cell_list_view(
        state_lens.focus(lambda s: s.neighborlist_params), lj.cutoff
    )
    potential = make_sharded_lj_potential(
        state_lens, lj, neighborlist, cap_owned, POSITIONS_AND_CELL
    )
    propagator = ShardMappedPropagator(
        make_md_propagator(state_lens, config.md.integrator, potential), mesh
    )
    return run_md(key, propagator, device_put_replicated(state, mesh), config.run)


def main() -> None:
    """CLI entry point for domain-decomposed LJ molecular dynamics."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_md_file(config.run.out_file))


if __name__ == "__main__":
    main()
