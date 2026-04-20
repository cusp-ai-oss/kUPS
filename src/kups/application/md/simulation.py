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
from kups.application.utils.propagate import run_simulation_cycles, run_warmup_cycles
from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.lens import Lens, lens
from kups.core.logging import CompositeLogger, TqdmLogger
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
from kups.core.unitcell import UnitCell
from kups.core.utils.functools import identity
from kups.core.utils.jax import key_chain
from kups.md.integrators import Integrator, make_md_step_from_state


class IsMdGradients(Protocol):
    """Protocol for MD gradient outputs.

    Attributes:
        positions: Position gradients as Table[ParticleId, Array].
        unitcell: Unit cell gradients as Table[SystemId, UnitCell].
    """

    @property
    def positions(self) -> Table[ParticleId, Array]: ...
    @property
    def unitcell(self) -> Table[SystemId, UnitCell]: ...


class IsMdState(Protocol):
    """Protocol for the full MD simulation state.

    Attributes:
        particles: Per-particle data (positions, momenta, forces, etc.).
        systems: Per-system data (unit cell, thermostat parameters, etc.).
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
) -> Propagator[State]:
    """Build a single MD propagator step with error recovery and step counting.

    Args:
        state_lens: Lens focusing on the MD sub-state within the full state.
        integrator: Integration algorithm for equations of motion.
        potential: Potential energy function providing forces and gradients.

    Returns:
        Propagator that advances the state by one MD step.
    """
    mapped_potential = MappedPotential(
        potential, lambda x: (x.positions.data, x.unitcell.data), identity
    )
    derivative_computation = PotentialAsPropagator(
        CachedPotential(
            mapped_potential,
            lens(
                lambda x: PotentialOut(
                    x.systems.map_data(lambda x: x.potential_energy),
                    (
                        x.particles.data.position_gradients,
                        x.systems.data.unitcell_gradients,
                    ),
                    EMPTY,
                )
            ),
            lambda x: PotentialOut(
                Index.new(x.systems.keys),  # type: ignore
                (x.particles.data.system, Index.new(x.systems.keys)),
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
