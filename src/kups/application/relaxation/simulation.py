# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Relaxation propagator construction and simulation runner."""

from typing import Any, Protocol

import jax.numpy as jnp
import optax
from jax import Array

from kups.application.relaxation.data import (
    RelaxParticles,
    RelaxRunConfig,
    RelaxSystems,
)
from kups.application.relaxation.logging import RelaxLoggedData
from kups.application.utils.propagate import run_simulation_cycles
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Table
from kups.core.lens import Lens, View, lens
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.potential import (
    EMPTY,
    CachedPotential,
    EmptyType,
    MappedPotential,
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
from kups.core.utils.functools import identity
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.propagator import RelaxationPropagator


class IsRelaxState(IsState[RelaxParticles, RelaxSystems], Protocol):
    """Protocol for relaxation simulation states."""

    @property
    def opt_state(self) -> optax.OptState: ...
    @property
    def step(self) -> Array: ...


class IsRelaxGradients(Protocol):
    """Protocol for gradient containers returned by relaxation potentials."""

    @property
    def positions(self) -> Table[ParticleId, Array]: ...
    @property
    def cell(self) -> Table[SystemId, Cell[AnyPeriodicity]]: ...


class OptInit(Protocol):
    """Protocol for initialising an Optax optimizer state from gradients."""

    def __call__(
        self,
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
    ) -> optax.OptState: ...


def make_relax_propagator[State: IsRelaxState, Gradients: IsRelaxGradients](
    state_lens: Lens[State, State],
    potential: Potential[State, Gradients, EmptyType, Any],
    optimizer: Optimizer[Any, Any],
    optimize_cell: bool = False,
) -> tuple[Propagator[State], OptInit]:
    """Build a relaxation propagator with step counting and error recovery.

    Args:
        state_lens: Lens focusing on the relaxation sub-state.
        potential: Potential whose gradients drive the optimisation.
        optimizer: Optimizer (e.g. FIRE, Adam, L-BFGS).
        optimize_cell: If True, optimise both positions and lattice vectors;
            otherwise optimise positions only.

    Returns:
        Tuple of ``(propagator, opt_init)`` where *propagator* performs one
        optimisation step and *opt_init* initialises the optimizer state.
    """
    # Cache the gradient and forces within the state
    pot = MappedPotential(
        potential, lambda x: (x.positions.data, x.cell.data), identity
    )
    pot = CachedPotential(
        pot,
        lens(
            lambda x: PotentialOut(
                x.systems.map_data(lambda x: x.potential_energy),
                (x.particles.data.position_gradients, x.systems.data.cell_gradients),
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

    def relax_prop_and_opt_init[T](
        prop_view: View[tuple[Array, Cell[AnyPeriodicity]], T],
    ):
        prop_lens = state_lens.focus(
            lambda x: prop_view((x.particles.data.positions, x.systems.data.cell))
        )

        def opt_init(
            particles: Table[ParticleId, RelaxParticles],
            systems: Table[SystemId, RelaxSystems],
        ) -> optax.OptState:
            params = (particles.data.positions, systems.data.cell)
            indices = (particles.data.system, systems.index)
            return optimizer.init(prop_view(params), prop_view(indices))  # type: ignore

        return RelaxationPropagator(
            potential=MappedPotential(pot, prop_view, identity),
            property=prop_lens,
            opt_state=state_lens.focus(lambda x: x.opt_state),
            optimizer=optimizer,
        ), opt_init

    relax_prop, opt_init = (
        relax_prop_and_opt_init(lens(identity))
        if optimize_cell
        else relax_prop_and_opt_init(lens(lambda x: x[0]))
    )
    step_prop = step_counter_propagator(state_lens.focus(lambda x: x.step))
    return ResetOnErrorPropagator(
        SequentialPropagator((relax_prop, step_prop))
    ), opt_init


def run_relax[State: IsRelaxState](
    key: Array, propagator: Propagator[State], state: State, config: RelaxRunConfig
) -> State:
    """Run structure relaxation with early stopping on force convergence.

    Args:
        key: JAX PRNG key.
        propagator: Relaxation propagator from ``make_relax_propagator``.
        state: Initial simulation state.
        config: Run configuration (max_steps, force_tolerance, out_file).

    Returns:
        Final relaxation state after convergence or ``max_steps``.
    """

    def converged(s: State) -> bool:
        forces = s.particles.data.forces
        max_force = jnp.max(jnp.linalg.norm(forces, axis=-1))
        return bool(max_force < config.force_tolerance)

    def _postfix(s: State) -> dict[str, Any]:
        e = jnp.asarray(s.systems.data.potential_energy).sum()
        fmax = jnp.max(jnp.linalg.norm(s.particles.data.forces, axis=-1))
        return {"E[eV]": f"{float(e): .6f}", "fmax[eV/Å]": f"{float(fmax): .4e}"}

    logger = CompositeLogger(
        TqdmLogger(config.max_steps, postfix=_postfix),
        HDF5StorageWriter(config.out_file, RelaxLoggedData(), state, config.max_steps),
    )
    state = run_simulation_cycles(
        key, propagator, state, config.max_steps, logger, convergence_fn=converged
    )
    return state
