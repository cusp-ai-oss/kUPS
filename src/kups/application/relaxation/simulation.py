# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Relaxation propagator construction and simulation runner."""

from typing import Any, Callable, Protocol, no_type_check

import jax
import jax.numpy as jnp
import optax
from jax import Array

from kups.application.relaxation.data import (
    RelaxParticles,
    RelaxRunConfig,
    RelaxSystems,
    relax_gradients,
    relax_index_prefix,
    relax_parameters,
)
from kups.application.relaxation.logging import RelaxLoggedData
from kups.application.utils.propagate import make_cycle_function, run_simulation_cycles
from kups.core.data import Table
from kups.core.lens import Lens, View, lens
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.potential import (
    EMPTY,
    CachedPotential,
    EmptyType,
    Potential,
    PotentialOut,
)
from kups.core.propagator import (
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    step_counter_propagator,
)
from kups.core.storage import HDF5StorageWriter
from kups.core.typing import IsState, ParticleId, SystemId
from kups.core.utils.jax import jit
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndCellIndex,
    PositionsAndSystemIndex,
)
from kups.relaxation.convergence import converged_per_system
from kups.relaxation.optimizer import Optimizer, SupportsReset
from kups.relaxation.propagator import RelaxationPropagator


class IsRelaxState(IsState[RelaxParticles, RelaxSystems], Protocol):
    @property
    def opt_state(self) -> optax.OptState: ...
    @property
    def step(self) -> Array: ...


type OptInit = Callable[
    [Table[ParticleId, RelaxParticles], Table[SystemId, RelaxSystems]], optax.OptState
]
type OptReset = Callable[
    [
        Table[ParticleId, RelaxParticles],
        Table[SystemId, RelaxSystems],
        optax.OptState,
        Table[SystemId, Array],
    ],
    optax.OptState,
]
type IndexPrefix = Callable[
    [Table[ParticleId, RelaxParticles], Table[SystemId, RelaxSystems]],
    PositionsAndCellIndex,
]


def make_relax_step[State, Substate: IsRelaxState](
    state_lens: Lens[State, Substate],
    potential: Potential[State, PositionsAndCell, EmptyType, Any],
    optimizer: Optimizer[PositionsAndCell, Any],
    gradient: Lens[Geometry, PositionsAndCell],
    *,
    accept: View[State, Table[SystemId, Array]] | None = None,
    index_prefix: IndexPrefix = relax_index_prefix,
) -> tuple[Propagator[State], OptInit, OptReset]:
    """Wire an optimizer step, initialization, and explicit per-system reset.

    The acceptance view reads the freshly evaluated gradients. Index mappings
    are supplied explicitly; streaming may use a fixed row-to-slot mapping.
    """

    def to_geometry(s: State) -> Geometry:
        sub = state_lens.get(s)
        return Geometry(
            sub.particles.map_data(
                lambda p: PositionsAndSystemIndex(p.positions, p.system)
            ),
            sub.systems.map_data(lambda x: x.cell),
        )

    def cached_out(s: State) -> PotentialOut[PositionsAndCell, EmptyType]:
        sub = state_lens.get(s)
        return PotentialOut(
            sub.systems.map_data(lambda x: x.potential_energy),
            relax_gradients(sub),
            EMPTY,
        )

    @no_type_check  # CachedPotential uses a heterogeneous pytree prefix at this boundary.
    def cached_index(s: State) -> PotentialOut[PositionsAndCell, EmptyType]:
        sub = state_lens.get(s)
        return PotentialOut(
            sub.systems.index, relax_index_prefix(sub.particles, sub.systems), EMPTY
        )

    def opt_init(
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
    ) -> optax.OptState:
        return optimizer.init(
            relax_parameters(particles, systems), index_prefix(particles, systems)
        )

    def opt_reset(
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
        opt_state: optax.OptState,
        mask: Table[SystemId, Array],
    ) -> optax.OptState:
        if not isinstance(optimizer, SupportsReset):
            raise ValueError("Streaming requires an optimizer with per-system reset.")
        return optimizer.reset(
            opt_state,
            relax_parameters(particles, systems),
            index_prefix(particles, systems),
            mask,
        )

    step = RelaxationPropagator(
        potential=CachedPotential(potential, lens(cached_out), cached_index),
        property=lens(to_geometry).nest(gradient),
        opt_state=state_lens.focus(lambda x: x.opt_state),
        optimizer=optimizer,
        accept=accept,
        mask_idx=lambda s: index_prefix(
            state_lens.get(s).particles, state_lens.get(s).systems
        ),
    )
    return step, opt_init, opt_reset


def make_relax_propagator[State, Substate: IsRelaxState](
    state_lens: Lens[State, Substate],
    potential: Potential[State, PositionsAndCell, EmptyType, Any],
    optimizer: Optimizer[PositionsAndCell, Any],
    gradient: Lens[Geometry, PositionsAndCell],
    *,
    force_tolerance: float | None = None,
    include_cell: bool = True,
) -> tuple[Propagator[State], OptInit]:
    """Build a relaxation step with counting and capacity-error recovery."""

    def accept(s: State) -> Table[SystemId, Array]:
        sub = state_lens.get(s)
        assert force_tolerance is not None
        return converged_per_system(
            relax_gradients(sub),
            relax_index_prefix(sub.particles, sub.systems),
            force_tolerance,
            include_cell=include_cell,
        ).map_data(jnp.logical_not)

    step, init, _ = make_relax_step(
        state_lens,
        potential,
        optimizer,
        gradient,
        accept=accept if force_tolerance is not None else None,
    )
    return ResetOnErrorPropagator(
        SequentialPropagator(
            (
                step,
                step_counter_propagator(state_lens.focus(lambda s: s.step)),
            )
        )
    ), init


def run_relax[State: IsRelaxState](
    key: Array, propagator: Propagator[State], state: State, config: RelaxRunConfig
) -> State:
    """Run structure relaxation with early stopping on convergence.

    Args:
        key: JAX PRNG key.
        propagator: Relaxation propagator from ``make_relax_propagator``.
        state: Initial simulation state.
        config: Run configuration (max_steps, force_tolerance, out_file).

    Returns:
        Final relaxation state after convergence or ``max_steps``.
    """

    @jit
    def converged_value(s: State) -> Array:
        converged = converged_per_system(
            relax_gradients(s),
            relax_index_prefix(s.particles, s.systems),
            config.force_tolerance,
            include_cell=config.optimize_cell,
        )
        return jnp.all(converged.data)

    def converged(s: State) -> bool:
        return bool(converged_value(s))

    @jit
    def _postfix_jit(s: State) -> dict[str, Array]:
        e = jnp.asarray(s.systems.data.potential_energy).sum()
        fmax = jnp.max(jnp.linalg.norm(s.particles.data.forces, axis=-1))
        return {"E[eV]": e, "fmax[eV/Å]": fmax}

    def _postfix(s: State) -> dict[str, Any]:
        data = _postfix_jit(s)
        return jax.tree.map(lambda x: f"{float(x):.4e}", data)

    logger = CompositeLogger(
        TqdmLogger(config.max_steps, postfix=_postfix),
        HDF5StorageWriter(config.out_file, RelaxLoggedData(), state, config.max_steps),
    )
    state = run_simulation_cycles(
        key,
        make_cycle_function(propagator),
        state,
        config.max_steps,
        logger,
        convergence_fn=converged,
    )
    return state
