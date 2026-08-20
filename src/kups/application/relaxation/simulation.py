# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Relaxation propagator construction and simulation runner."""

from typing import Any, Protocol, no_type_check

import jax
import jax.numpy as jnp
import optax
from jax import Array

from kups.application.relaxation.data import (
    RelaxParticles,
    RelaxRunConfig,
    RelaxSystems,
)
from kups.application.relaxation.logging import RelaxLoggedData
from kups.application.utils.propagate import make_cycle_function, run_simulation_cycles
from kups.core.data import Table
from kups.core.lens import Lens, lens
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.neighborlist import IsVerletState, VerletSkinPropagator
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
    PositionsAndSystemIndex,
)
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.propagator import RelaxationPropagator


class IsRelaxState(IsState[RelaxParticles, RelaxSystems], Protocol):
    """Protocol for relaxation simulation states."""

    @property
    def opt_state(self) -> optax.OptState: ...
    @property
    def step(self) -> Array: ...


class IsVerletRelaxState(IsRelaxState, IsVerletState, Protocol):
    """Relaxation state that can also carry the Verlet-skin group."""


class OptInit(Protocol):
    """Protocol for initialising an Optax optimizer state from gradients."""

    def __call__(
        self,
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
    ) -> optax.OptState: ...


def make_relax_propagator[State: IsVerletRelaxState](
    state_lens: Lens[State, State],
    potential: Potential[State, Any, EmptyType, Any],
    optimizer: Optimizer[PositionsAndCell, Any],
    gradient: Lens[Geometry, PositionsAndCell],
    *,
    verlet_skin: float = 0.0,
    cutoffs: Table[SystemId, Array] | None = None,
) -> tuple[Propagator[State], OptInit]:
    """Build a relaxation propagator with step counting and error recovery.

    Args:
        state_lens: Lens focusing on the relaxation sub-state.
        potential: Potential reporting the DOF gradient ``∂E/∂u`` (built with the
            same ``gradient`` filter).
        optimizer: Optimizer (e.g. FIRE, Adam, L-BFGS).
        gradient: Relaxation filter selecting the optimizer DOFs ``u`` — must be
            the one ``potential`` was built with. The propagator optimises *these*
            DOFs (not raw ``(positions, cell)``) so the filter's atoms-ride-the-cell
            coupling is applied on every ``set``; using the raw property would drop
            that coupling and diverge from ASE's cell filters.
        verlet_skin: Neighbor-list skin width (Å). ``0`` (default) leaves the
            potential's own neighbor-list construction untouched. ``> 0`` wraps
            the optimisation step in a
            [`VerletSkinPropagator`][kups.core.neighborlist.verlet.VerletSkinPropagator],
            whose docstring states the contract on the potential, the state and
            the error recovery.
        cutoffs: True per-system cutoffs; required when ``verlet_skin > 0``.

    Returns:
        Tuple of ``(propagator, opt_init)`` where *propagator* performs one
        optimisation step and *opt_init* initialises the optimizer state.
    """

    def to_geometry(x: State) -> Geometry:
        return Geometry(
            x.particles.map_data(
                lambda p: PositionsAndSystemIndex(p.positions, p.system)
            ),
            x.systems.map_data(lambda s: s.cell),
        )

    def cached_out(x: State) -> PotentialOut[PositionsAndCell, EmptyType]:
        return PotentialOut(
            x.systems.map_data(lambda s: s.potential_energy),
            PositionsAndCell(
                x.particles.map_data(lambda x: x.position_gradients),
                x.systems.map_data(lambda x: x.cell_gradients),
            ),
            EMPTY,
        )

    @no_type_check
    def cached_index(x: State) -> PotentialOut[PositionsAndCell, EmptyType]:
        return PotentialOut(
            x.systems.index,
            PositionsAndCell(x.particles.data.system, x.systems.index),
            EMPTY,
        )

    def opt_init(
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
    ) -> optax.OptState:
        params = PositionsAndCell(
            particles.map_data(lambda p: p.positions),
            systems.map_data(lambda s: s.cell),
        )
        # pyrefly: ignore [bad-argument-type]
        prefix = PositionsAndCell(particles.data.system, systems.index)
        return optimizer.init(params, prefix)

    pot = CachedPotential(potential, lens(cached_out), cached_index)
    relax_prop: Propagator[State] = RelaxationPropagator(
        potential=pot,
        property=lens(to_geometry).nest(gradient),
        opt_state=state_lens.focus(lambda x: x.opt_state),
        optimizer=optimizer,
    )
    if verlet_skin > 0:
        assert cutoffs is not None, "verlet_skin > 0 requires cutoffs."
        relax_prop = VerletSkinPropagator(relax_prop, cutoffs, verlet_skin)
    step_prop = step_counter_propagator(state_lens.focus(lambda x: x.step))
    prop = ResetOnErrorPropagator(SequentialPropagator((relax_prop, step_prop)))
    return prop, opt_init


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
        max_dof = jnp.max(jnp.linalg.norm(s.particles.data.position_gradients, axis=-1))
        if config.optimize_cell:
            leaves = jax.tree.leaves(s.systems.data.cell_gradients)
            cell_dof = jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in leaves]))
            max_dof = jnp.maximum(max_dof, cell_dof)
        return max_dof < config.force_tolerance

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
