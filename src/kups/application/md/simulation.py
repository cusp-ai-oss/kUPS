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
from kups.application.md.integrators import make_md_step_from_state
from kups.application.md.logging import MDLoggedData
from kups.application.utils.propagate import (
    make_cycle_function,
    run_simulation_cycles,
    run_warmup_cycles,
)
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Table
from kups.core.lens import Lens, lens
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.neighborlist import IsSkinListState
from kups.core.potential import (
    EMPTY,
    CachedPotential,
    EmptyType,
    LinearMappedPotential,
    LinearMappedPotentialInput,
    Potential,
    PotentialAsPropagator,
    PotentialOut,
)
from kups.core.propagator import (
    LoopPropagator,
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    step_counter_propagator,
)
from kups.core.storage import HDF5StorageWriter
from kups.core.typing import IsState, ParticleId, SystemId
from kups.core.utils.jax import key_chain
from kups.md.integrators import Integrator


class IsMdGradients(Protocol):
    """Protocol for MD gradient outputs.

    Attributes:
        positions: Position gradients as Table[ParticleId, Array].
        cell: Cell gradients as Table[SystemId, Cell].
    """

    @property
    def positions(self) -> Table[ParticleId, Array]: ...
    @property
    def cell(self) -> Table[SystemId, Cell[AnyPeriodicity]]: ...


class IsMdState(IsState[MDParticles, MDSystems], IsSkinListState, Protocol):
    """Protocol for the full MD simulation state.

    Attributes:
        particles: Per-particle data (positions, momenta, forces, etc.).
        systems: Per-system data (cell, thermostat parameters, etc.).
        neighborlist_params: Capacities of the edges read from the skin list.
        skin_list: Neighbor pairs kept across steps.
        step: Current simulation step counter.
    """

    @property
    def step(self) -> Array: ...


def potential_map(
    input: LinearMappedPotentialInput[IsMdState, IsMdGradients, Any],
) -> PotentialOut[tuple[Array, Cell[AnyPeriodicity]], EmptyType]:
    """Map the full potential output to the specific gradients needed for MD."""
    return PotentialOut(
        input.potential_out.total_energies,
        (
            input.potential_out.gradients.positions.data,
            input.potential_out.gradients.cell.data,
        ),
        EMPTY,
    )


def make_md_propagator[State: IsMdState, Grad: IsMdGradients](
    state_lens: Lens[State, State],
    integrator: Integrator,
    potential: Potential[State, Grad, EmptyType, Any],
) -> tuple[Propagator[State], Propagator[State]]:
    """Build the MD force evaluation and one MD step with step counting.

    Every force evaluation applies the potential's patch, which stores the skin list
    the potential read. A failed evaluation or step keeps the skin list, since every
    read checks it again. States start without forces, so evaluate them once before
    the first step.

    Args:
        state_lens: Lens focusing on the MD sub-state within the full state.
        integrator: Integration algorithm for equations of motion.
        potential: Potential reading the skin list, e.g. through
            ``SkinNeighborList.from_state``.

    Returns:
        The force evaluation, and the propagator that advances the state by one
        MD step.
    """
    mapped_potential = LinearMappedPotential(potential, potential_map)
    derivative_computation = PotentialAsPropagator(
        CachedPotential(
            mapped_potential,
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
            # pyrefly: ignore [bad-argument-type]
            lambda x: PotentialOut(
                x.systems.index,  # type: ignore
                (x.particles.data.system, x.systems.index),
                EMPTY,
            ),
        )
    )
    md_propagator = make_md_step_from_state(
        state_lens, derivative_computation, integrator
    )
    step_count_propagator = step_counter_propagator(state_lens.focus(lambda x: x.step))
    skin_list = state_lens.focus(lambda x: x.skin_list)
    return (
        ResetOnErrorPropagator(derivative_computation, keep=skin_list),
        ResetOnErrorPropagator(
            SequentialPropagator((md_propagator, step_count_propagator)),
            keep=skin_list,
        ),
    )


def run_md[State: IsMdState](
    key: Array, propagator: Propagator[State], state: State, config: MdRunConfig
) -> State:
    """Run a full MD simulation with warmup and production phases.

    Args:
        key: JAX PRNG key.
        propagator: MD step produced by `make_md_propagator`.
        state: Initial simulation state, with its forces evaluated.
        config: Run configuration (steps, output file and intervals, seed).

    Returns:
        Final simulation state after production run.
    """
    chain = key_chain(key)
    # Check assertions after each step during warmup cycles
    cycle_fn = make_cycle_function(propagator)
    logging.info("Warmup")
    state = run_warmup_cycles(next(chain), cycle_fn, state, config.num_warmup_steps)

    logging.info("Starting MD simulation")
    # each cycle fuses block_size steps into one dispatch; output is logged per cycle
    cycle_fn = make_cycle_function(LoopPropagator(propagator, config.block_size))
    num_cycles = config.num_steps // config.block_size
    logger = CompositeLogger(
        TqdmLogger(num_cycles),
        HDF5StorageWriter(
            config.out_file, MDLoggedData.create(config), state, num_cycles
        ),
    )
    state = run_simulation_cycles(next(chain), cycle_fn, state, num_cycles, logger)
    return state
