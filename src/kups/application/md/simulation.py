# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Protocol

import jax.numpy as jnp
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
from kups.core.lens import LambdaLens, Lens, View, bind, lens
from kups.core.logging import CompositeLogger, TqdmLogger
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


class IsMdState(IsState[MDParticles, MDSystems], Protocol):
    """Protocol for the full MD simulation state.

    Attributes:
        particles: Per-particle data (positions, momenta, forces, etc.).
        systems: Per-system data (cell, thermostat parameters, etc.).
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


def fix_atoms[State: IsMdState](
    state_lens: Lens[State, State], fixed: Array
) -> Lens[State, State]:
    """Hold the masked atoms fixed (ASE ``FixAtoms``) for every write through the lens.

    ``set`` keeps the current positions and momenta of the masked atoms, so no
    integrator or thermostat step moves them or gives them momentum. Positions
    are pinned in Cartesian coordinates, so under a barostat fixed atoms do not
    follow the cell (as with ASE's ``FixAtoms``).

    Args:
        state_lens: Lens the integrator steps write through.
        fixed: Boolean mask over the particles, ``True`` to hold fixed.
    """

    def _set(state: State, /, value: State) -> State:
        old = state_lens.get(state).particles.data
        new = value.particles.data
        value = bind(value, lambda s: s.particles.data.positions).set(
            jnp.where(fixed[:, None], old.positions, new.positions)
        )
        value = bind(value, lambda s: s.particles.data.momenta).set(
            jnp.where(fixed[:, None], old.momenta, new.momenta)
        )
        return state_lens.set(state, value)

    return LambdaLens(state_lens.get, _set)


def make_md_propagator[State: IsMdState, Grad: IsMdGradients](
    state_lens: Lens[State, State],
    integrator: Integrator,
    potential: Potential[State, Grad, EmptyType, Any],
    fixed: Array | None = None,
) -> Propagator[State]:
    """Build a single MD propagator step with error recovery and step counting.

    Args:
        state_lens: Lens focusing on the MD sub-state within the full state.
        integrator: Integration algorithm for equations of motion.
        potential: Potential energy function providing forces and gradients.
        fixed: Optional boolean mask over the particles; masked atoms keep their
            positions and momenta, and the CSVR thermostats count only the free
            atoms' degrees of freedom (``3N_free``, momentum is not conserved).

    Returns:
        Propagator that advances the state by one MD step.
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
    step_lens = state_lens
    degrees_of_freedom: View[State, Array] | None = None
    if fixed is not None:
        step_lens = fix_atoms(state_lens, fixed)

        def free_degrees_of_freedom(s: State) -> Array:
            system = state_lens.get(s).particles.data.system
            n_fixed = system.sum_over(fixed.astype(int)).data
            return 3 * (system.counts.data - n_fixed) - 3 * (n_fixed == 0)

        degrees_of_freedom = free_degrees_of_freedom

    md_propagator = make_md_step_from_state(
        step_lens,
        derivative_computation,
        integrator,
        degrees_of_freedom=degrees_of_freedom,
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
    # Check assertions after each step during warmup cycles
    cycle_fn = make_cycle_function(propagator)
    logging.info("Warmup")
    state = run_warmup_cycles(next(chain), cycle_fn, state, config.num_warmup_steps)

    logging.info("Starting MD simulation")
    # Each cycle fuses block_size steps into one dispatch and saves its final frame, so the
    # trajectory keeps num_steps // block_size frames (block_size=1 = one step per cycle).
    cycle_fn = make_cycle_function(LoopPropagator(propagator, config.block_size))
    num_cycles = config.num_steps // config.block_size
    logger = CompositeLogger(
        TqdmLogger(num_cycles),
        HDF5StorageWriter(config.out_file, MDLoggedData(), state, num_cycles),
    )
    state = run_simulation_cycles(next(chain), cycle_fn, state, num_cycles, logger)
    return state
