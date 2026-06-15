# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Protocol

from jax import Array

from kups.application.md.data import (
    MDParticles,
    MdRunConfig,
    MDSystems,
)
from kups.application.md.logging import MDLoggedData
from kups.application.utils.propagate import (
    run_simulation_cycles,
    run_verlet_cycles,
    run_warmup_cycles,
)
from kups.core.cell import Cell
from kups.core.data import Table
from kups.core.lens import Lens, lens
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.neighborlist import NearestNeighborList
from kups.core.neighborlist.verlet import RebuildSkinStep, TriggerStep
from kups.core.potential import (
    EMPTY,
    CachedPotential,
    EmptyType,
    MappedPotential,
    Potential,
    PotentialAsPropagator,
    PotentialOut,
)
from kups.core.propagator import (
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    step_counter_propagator,
)
from kups.core.storage import HDF5StorageWriter
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.functools import identity
from kups.core.utils.jax import key_chain
from kups.md.integrators import Integrator, make_md_step_from_state


class IsMdGradients(Protocol):
    """Protocol for MD gradient outputs.

    Attributes:
        positions: Position gradients as Table[ParticleId, Array].
        cell: Cell gradients as Table[SystemId, Cell].
    """

    @property
    def positions(self) -> Table[ParticleId, Array]: ...
    @property
    def cell(self) -> Table[SystemId, Cell]: ...


class IsMdState(Protocol):
    """Protocol for the full MD simulation state.

    Attributes:
        particles: Per-particle data (positions, momenta, forces, etc.).
        systems: Per-system data (cell, thermostat parameters, etc.).
        step: Current simulation step counter.
    """

    @property
    def particles(self) -> Table[ParticleId, MDParticles]: ...
    @property
    def systems(self) -> Table[SystemId, MDSystems]: ...
    @property
    def step(self) -> Array: ...


def make_md_propagator[State: IsMdState, Grad: IsMdGradients](
    state_lens: Lens[State, State],
    integrator: Integrator,
    potential: Potential[State, Grad, EmptyType, Any],
    forces_only: bool = False,
) -> Propagator[State]:
    """Build a single MD propagator step with error recovery and step counting.

    Args:
        state_lens: Lens focusing on the MD sub-state within the full state.
        integrator: Integration algorithm for equations of motion.
        potential: Potential energy function providing forces and gradients.
        forces_only: NVE/NVT optimization. When True, the step neither maps nor caches the
            cell-virial (``dE/dcell``); pair with a ``forces_only=True`` potential to skip
            computing it. Invalid for the NPT integrators (the barostat needs stress).
            ``cell_gradients`` is then not refreshed, so logged stress/pressure is
            unavailable — keep the default when you need the stress observable.
    """
    if forces_only and integrator in ("csvr_npt", "csvr_npt_1eval"):
        raise ValueError(
            f"forces_only=True is invalid for NPT integrator {integrator!r}: "
            "the barostat needs the cell-virial (stress)."
        )
    if forces_only:
        # Map positions from either a forces_only potential (gradient is a positions Table,
        # has .data) or a both-gradients potential (PositionAndCell, has .positions); cache
        # only position_gradients (cell_gradients is left untouched).
        derivative_computation = PotentialAsPropagator(
            CachedPotential(
                MappedPotential(
                    potential,
                    lambda x: x.positions.data if hasattr(x, "positions") else x.data,  # type: ignore
                    identity,
                ),
                lens(
                    lambda x: PotentialOut(
                        x.systems.map_data(lambda x: x.potential_energy),
                        x.particles.data.position_gradients,
                        EMPTY,
                    )
                ),
                lambda x: PotentialOut(
                    x.systems.index,  # type: ignore
                    x.particles.data.system,
                    EMPTY,
                ),  # type: ignore
            )
        )
    else:
        derivative_computation = PotentialAsPropagator(
            CachedPotential(
                MappedPotential(
                    potential, lambda x: (x.positions.data, x.cell.data), identity
                ),
                lens(
                    lambda x: PotentialOut(
                        x.systems.map_data(lambda x: x.potential_energy),
                        (
                            x.particles.data.position_gradients,
                            x.systems.data.cell_gradients,
                        ),
                        EMPTY,
                    )
                ),
                lambda x: PotentialOut(
                    x.systems.index,  # type: ignore
                    (x.particles.data.system, x.systems.index),
                    EMPTY,
                ),  # type: ignore
            )
        )
    md_propagator = make_md_step_from_state(
        state_lens, derivative_computation, integrator
    )
    step_count_propagator = step_counter_propagator(state_lens.focus(lambda x: x.step))
    propagator = ResetOnErrorPropagator(
        SequentialPropagator((md_propagator, step_count_propagator))
    )
    return propagator


def run_md[State: IsMdState](
    key: Array, propagator: Propagator[State], state: State, config: MdRunConfig
) -> State:
    """Run a full MD simulation with warmup and production phases.

    Args:
        key: JAX PRNG key.
        propagator: MD propagator produced by `make_md_propagator`.
        state: Initial simulation state.
        config: Run configuration (steps, output file, seed).

    Returns:
        Final simulation state after production run.
    """
    chain = key_chain(key)
    logging.info("Warmup")
    state = run_warmup_cycles(next(chain), propagator, state, config.num_warmup_steps)

    logging.info("Starting MD simulation")
    logger = CompositeLogger(
        TqdmLogger(config.num_steps),
        HDF5StorageWriter(config.out_file, MDLoggedData(), state, config.num_steps),
    )
    state = run_simulation_cycles(
        next(chain), propagator, state, config.num_steps, logger
    )
    return state


def make_verlet_md_propagators[State: IsMdState, Grad: IsMdGradients](
    state_lens: Lens[State, State],
    integrator: Integrator,
    potential: Potential[State, Grad, EmptyType, Any],
    cutoff: float,
    skin: float,
    skin_nl: NearestNeighborList,
    forces_only: bool = False,
) -> tuple[Propagator[State], Propagator[State]]:
    """Build the (reuse, rebuild) propagator pair for a Verlet-skin MD run.

    ``forces_only`` (NVE/NVT only) skips the cell-virial; see :func:`make_md_propagator`.

    Both share the same MD step (which reads ``state.neighborlist`` — a
    ``RefineCutoffNeighborList`` over the stored skin list); ``rebuild`` first refreshes
    the skin list + references via the injected ``skin_nl`` builder (dense for box ~ cutoff,
    cell-list for large boxes — see ``neighborlist.verlet.dense_skin_nl`` / ``cell_skin_nl``).
    Each appends a ``TriggerStep`` that sets ``state.should_rebuild`` for the next dispatch.
    """
    md_propagator = make_md_propagator(state_lens, integrator, potential, forces_only)
    trigger = TriggerStep(cutoff, skin)
    rebuild = RebuildSkinStep(cutoff, skin, skin_nl)

    # Closures (not a SequentialPropagator) so the MD step receives `key` directly in both
    # variants: the rebuild/trigger steps don't consume the PRNG chain, so the dynamics are
    # independent of the rebuild schedule (bit-identical to the every-step path).
    def reuse_prop(key: Array, state: State) -> State:
        return trigger(key, md_propagator(key, state))

    def rebuild_prop(key: Array, state: State) -> State:
        return trigger(key, md_propagator(key, rebuild(key, state)))

    return reuse_prop, rebuild_prop


def run_md_verlet[State: IsMdState](
    key: Array,
    reuse_propagator: Propagator[State],
    rebuild_propagator: Propagator[State],
    state: State,
    config: MdRunConfig,
) -> State:
    """Run a full MD simulation (warmup + production) with a Verlet-skin neighbor list."""
    chain = key_chain(key)
    logging.info("Warmup (Verlet skin)")
    state = run_verlet_cycles(
        next(chain),
        reuse_propagator,
        rebuild_propagator,
        state,
        config.num_warmup_steps,
    )
    logging.info("Starting MD simulation (Verlet skin)")
    logger = CompositeLogger(
        TqdmLogger(config.num_steps),
        HDF5StorageWriter(config.out_file, MDLoggedData(), state, config.num_steps),
    )
    state = run_verlet_cycles(
        next(chain),
        reuse_propagator,
        rebuild_propagator,
        state,
        config.num_steps,
        logger,
    )
    return state
