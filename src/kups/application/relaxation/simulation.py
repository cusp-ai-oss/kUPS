# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Relaxation propagator construction and simulation runner."""

from typing import Any, Protocol

import jax
import jax.numpy as jnp
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
from kups.core import units
from kups.core.data import Table
from kups.core.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndCellIndex,
    PositionsAndSystemIndex,
    position_and_cell_idx_view,
)
from kups.core.lens import LambdaLens, Lens, View, lens
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.potential import (
    EMPTY,
    CachedPotential,
    EmptyType,
    LinearMappedPotential,
    LinearMappedPotentialInput,
    Potential,
    PotentialOut,
)
from kups.core.propagator import (
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    step_counter_propagator,
)
from kups.core.result import as_result_function
from kups.core.storage import HDF5StorageWriter
from kups.core.typing import IsState, ParticleId, SystemId
from kups.core.units import catalog as u
from kups.core.units import magnitudes_of
from kups.core.utils.jax import jit
from kups.relaxation.convergence import converged_per_system
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.propagator import RelaxationPropagator, UpdateMask


class IsRelaxState[OptState](IsState[RelaxParticles, RelaxSystems], Protocol):
    """Relaxation state: particle and system tables, optimizer state and step count."""

    @property
    def opt_state(self) -> OptState: ...
    @property
    def step(self) -> Array: ...


class OptInit[OptState](Protocol):
    """Initialize optimizer state from relaxation particle and system tables."""

    def __call__(
        self,
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
    ) -> OptState: ...


class IndexPrefixFn(Protocol):
    """Map the optimizer's particle and cell parameters to their owning systems."""

    def __call__(
        self,
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
    ) -> PositionsAndCellIndex: ...


def make_relax_step[State, OptState](
    state_lens: Lens[State, IsRelaxState[OptState]],
    potential: Potential[State, PositionsAndCell[units.Force], EmptyType, Any],
    optimizer: Optimizer[PositionsAndCell[units.Length], OptState],
    gradient: Lens[Geometry, PositionsAndCell[units.Length]],
    *,
    accept: View[State, Table[SystemId, Array]] | None = None,
    index_prefix: IndexPrefixFn = relax_index_prefix,
) -> tuple[Propagator[State], OptInit[OptState]]:
    """Wire one optimizer step and its optimizer-state initializer.

    Args:
        state_lens: Lens focusing on the relaxation sub-state.
        potential: Potential reporting the DOF gradient ``∂E/∂u`` (built with the
            same ``gradient`` filter).
        optimizer: Optimizer (e.g. FIRE, Adam, L-BFGS).
        gradient: Relaxation filter selecting the optimizer DOFs ``u`` — must be
            the one ``potential`` was built with. The step optimises *these* DOFs
            (not raw ``(positions, cell)``) so the filter's atoms-ride-the-cell
            coupling is applied on every ``set``; using the raw property would
            drop that coupling and diverge from ASE's cell filters.
        accept: Optional per-system update mask, read after the freshly
            evaluated energies and gradients are cached. Rejected systems keep
            their parameters (up to the filter's set/get round-off); their
            optimizer state still advances.
        index_prefix: Maps every optimizer DOF to its owning system, for
            ``optimizer.init`` and the update mask. Streaming may use a fixed
            row-to-slot mapping.

    Fixed particles (``RelaxParticles.fixed``) get a zero position gradient,
    which the cache, convergence and line searches all see, and keep their
    position DOF on every set, so ``get(set(s, u)) != u`` on their rows.

    Returns:
        Tuple of ``(step, opt_init)``. ``step`` does not count steps or roll back
        on failed runtime assertions; :func:`make_relax_propagator` adds both.
    """

    def to_geometry(s: State) -> Geometry:
        sub = state_lens.get(s)
        return Geometry(
            sub.particles.map_data(
                lambda p: PositionsAndSystemIndex(p.positions, p.system)
            ),
            sub.systems.map_data(lambda x: x.cell),
        )

    def cached_out(s: State) -> PotentialOut[PositionsAndCell[units.Force], EmptyType]:
        sub = state_lens.get(s)
        return PotentialOut(
            sub.systems.map_data(lambda x: x.potential_energy),
            relax_gradients(sub),
            EMPTY,
        )

    def opt_init(
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
    ) -> OptState:
        # The optimizer's memory holds magnitudes, as its update does: a
        # transform mixes a parameter with its gradient, so dimensions live at
        # the propagator boundary rather than inside the chain.
        return optimizer.init(
            magnitudes_of(relax_parameters(particles, systems)),
            index_prefix(particles, systems),
        )

    def system_index(s: State) -> PositionsAndCellIndex:
        sub = state_lens.get(s)
        return index_prefix(sub.particles, sub.systems)

    def pin_fixed_gradients(
        inp: LinearMappedPotentialInput[
            State, PositionsAndCell[units.Force], EmptyType
        ],
    ) -> PotentialOut[PositionsAndCell[units.Force], EmptyType]:
        # Linear, so it also holds for the Kahan compensation.
        particles = state_lens.get(inp.state).particles.data
        out = inp.potential_out
        positions = out.gradients.positions.map_data(
            lambda g: particles.pin_fixed(g, units.zeros_like(g))
        )
        return PotentialOut(
            out.total_energies,
            out.gradients._replace(positions=positions),
            out.hessians,
        )

    dofs = lens(to_geometry).nest(gradient)

    def set_dofs(s: State, value: PositionsAndCell[units.Length]) -> State:
        particles = state_lens.get(s).particles.data
        current = dofs.get(s).positions.data
        positions = value.positions.map_data(lambda q: particles.pin_fixed(q, current))
        return dofs.set(s, value._replace(positions=positions))

    step = RelaxationPropagator(
        potential=CachedPotential(
            LinearMappedPotential(potential, pin_fixed_gradients),
            lens(cached_out),
            lambda s: position_and_cell_idx_view(to_geometry(s)),
        ),
        property=LambdaLens(dofs.get, set_dofs),
        opt_state=state_lens.focus(lambda x: x.opt_state),
        optimizer=optimizer,
        mask=None
        if accept is None
        else UpdateMask(accept=accept, system_index=system_index),
    )
    return step, opt_init


def make_relax_propagator[State, OptState](
    state_lens: Lens[State, IsRelaxState[OptState]],
    potential: Potential[State, PositionsAndCell[units.Force], EmptyType, Any],
    optimizer: Optimizer[PositionsAndCell[units.Length], OptState],
    gradient: Lens[Geometry, PositionsAndCell[units.Length]],
) -> tuple[Propagator[State], OptInit[OptState]]:
    """Build a relaxation propagator with step counting and error recovery.

    Wraps :func:`make_relax_step` (see it for the arguments, notably the
    ``gradient`` filter contract) in a step counter and a
    ``ResetOnErrorPropagator``.

    Returns:
        Tuple of ``(propagator, opt_init)`` where *propagator* performs one
        optimisation step and *opt_init* initialises the optimizer state.
    """

    step, init = make_relax_step(state_lens, potential, optimizer, gradient)
    count = step_counter_propagator(state_lens.focus(lambda s: s.step))
    return ResetOnErrorPropagator(SequentialPropagator((step, count))), init


def run_relax[State: IsRelaxState[object]](
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

    Raises:
        AssertionError: If a step leaves a non-finite position gradient, or a
            non-finite cell gradient when ``config.optimize_cell`` is set.
    """

    @jit
    @as_result_function
    def converged_value(s: State) -> Array:
        converged = converged_per_system(
            relax_gradients(s),
            relax_index_prefix(s.particles, s.systems),
            config.force_tolerance,
            include_cell=config.optimize_cell,
        )
        return jnp.all(converged.data)

    def converged(s: State) -> bool:
        result = converged_value(s)
        result.raise_assertion()
        return bool(result.value)

    @jit
    def _postfix_jit(s: State) -> dict[str, Array]:
        e = s.systems.data.potential_energy.to(u.eV).sum()
        fmax = jnp.max(
            units.norm(s.particles.data.forces, axis=-1).to(u.eV / u.angstrom)
        )
        return {"E[eV]": e, "fmax[eV/Å]": fmax}

    def _postfix(s: State) -> dict[str, Any]:
        data = _postfix_jit(s)
        return jax.tree.map(lambda x: f"{float(x):.4e}", data)

    logger = CompositeLogger(
        TqdmLogger(config.max_steps, postfix=_postfix),
        HDF5StorageWriter(config.out_file, RelaxLoggedData(), state, config.max_steps),
    )
    return run_simulation_cycles(
        key,
        make_cycle_function(propagator),
        state,
        config.max_steps,
        logger,
        convergence_fn=converged,
    )
