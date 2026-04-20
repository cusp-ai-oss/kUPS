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
from kups.core.data import Table
from kups.core.data.index import Index
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
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import UnitCell
from kups.core.utils.functools import identity
from kups.relaxation.propagator import RelaxationPropagator


class IsRelaxState(Protocol):
    """Protocol for relaxation simulation states."""

    @property
    def particles(self) -> Table[ParticleId, RelaxParticles]: ...
    @property
    def systems(self) -> Table[SystemId, RelaxSystems]: ...
    @property
    def opt_state(self) -> optax.OptState: ...
    @property
    def step(self) -> Array: ...


class IsRelaxGradients(Protocol):
    """Protocol for gradient containers returned by relaxation potentials."""

    @property
    def positions(self) -> Table[ParticleId, Array]: ...
    @property
    def unitcell(self) -> Table[SystemId, UnitCell]: ...


class OptInit(Protocol):
    """Protocol for initialising an Optax optimizer state from gradients."""

    def __call__(self, grads: tuple[Array, Array]) -> optax.OptState: ...


def make_relax_propagator[State: IsRelaxState, Gradients: IsRelaxGradients](
    state_lens: Lens[State, State],
    potential: Potential[State, Gradients, EmptyType, Any],
    optimizer: optax.GradientTransformationExtraArgs,
    optimize_unitcell: bool = False,
) -> tuple[Propagator[State], OptInit]:
    """Build a relaxation propagator with step counting and error recovery.

    Args:
        state_lens: Lens focusing on the relaxation sub-state.
        potential: Potential whose gradients drive the optimisation.
        optimizer: Optax gradient transformation (e.g. FIRE, Adam, L-BFGS).
        optimize_unitcell: If True, optimise both positions and lattice vectors;
            otherwise optimise positions only.

    Returns:
        Tuple of ``(propagator, opt_init)`` where *propagator* performs one
        optimisation step and *opt_init* initialises the Optax optimizer state.
    """
    # Cache the gradient and forces within the state
    pot = CachedPotential(
        MappedPotential(
            potential, lambda x: (x.positions.data, x.unitcell.data), identity
        ),
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

    def relax_prop_and_opt_init[T](prop_view: View[tuple[Array, UnitCell], T]):
        prop_lens = state_lens.focus(
            lambda x: prop_view((x.particles.data.positions, x.systems.data.unitcell))
        )
        return RelaxationPropagator(
            potential=MappedPotential(pot, prop_view, identity),
            property=prop_lens,
            opt_state=state_lens.focus(lambda x: x.opt_state),
            optimizer=optimizer,
        ), lambda grads: optimizer.init(prop_view(grads))  # type: ignore

    relax_prop, opt_init = (
        relax_prop_and_opt_init(lens(identity))
        if optimize_unitcell
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

    logger = CompositeLogger(
        TqdmLogger(config.max_steps),
        HDF5StorageWriter(config.out_file, RelaxLoggedData(), state, config.max_steps),
    )
    state = run_simulation_cycles(
        key, propagator, state, config.max_steps, logger, convergence_fn=converged
    )
    return state
