# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Gradient-based relaxation using Optax optimizers.

This module provides a [Propagator][kups.core.propagator.Propagator] implementation
for gradient-based optimization using [Optax](https://optax.readthedocs.io/).

The [RelaxationPropagator][kups.relaxation.propagator.RelaxationPropagator] supports
both standard optimizers (Adam, SGD) and line-search optimizers (L-BFGS, backtracking).
"""

from typing import Any

import jax.numpy as jnp
from jax import Array

from kups.core import units
from kups.core.data.table import Table
from kups.core.lens import Lens, View
from kups.core.patch import IndexLensPatch
from kups.core.potential import Potential
from kups.core.propagator import Propagator
from kups.core.typing import SystemId
from kups.core.units import magnitudes_of
from kups.core.utils.jax import dataclass, field
from kups.relaxation.optimizer import (
    Optimizer,
    apply_updates,
    relabel_like,
)


@dataclass(kw_only=True)
class UpdateMask[State, Indices]:
    """Per-system acceptance and the matching parameter-index view.

    Both views read the state after the potential's patch has been applied.

    Attributes:
        accept: Per-system mask; only ``True`` systems receive the update.
        system_index: Index prefix of the optimized parameters mapping each
            parameter row to its system.
    """

    accept: View[State, Table[SystemId, Array]] = field(static=True)
    system_index: View[State, Indices] = field(static=True)


@dataclass
class RelaxationPropagator[State, Params, OptState, Indices](Propagator[State]):
    """Propagator for one step of gradient-based optimization.

    Works with optax optimizers and with the per-system line searches in
    :mod:`kups.relaxation.transforms`. Each step passes the optimizer the
    gradient, the per-system energies and a ``value_and_grad_fn`` for trial
    points, which line searches use and other transforms ignore. Applying the
    potential's patch each step lets it update state such as neighbor lists.

    The optimizer runs on magnitudes: parameters, gradients and energies enter
    it in the parameters' base (that of the first quantity leaf, ATOMISTIC by
    default: Å, eV, amu), and the step it returns is relabelled with each
    parameter's dimension before it is applied. Numeric hyperparameters of the
    transforms are read in the same base.

    Type Parameters:
        State: Simulation state.
        Params: Optimized parameters.
        OptState: State type of ``optimizer``.
        Indices: Index prefix returned by the optional update mask.

    Attributes:
        potential: Potential computing ``∂E/∂`` of the property, structured like
            it with each leaf in energy over that leaf's dimension (e.g.
            ``PositionsAndCell[units.Force]`` for ``PositionsAndCell[units.Length]``).
        property: Lens to get/set the property being optimized
        opt_state: Lens to get/set the optimizer state
        optimizer: Gradient transformation
        mask: Optional acceptance and parameter-index views, read after evaluating
            the potential and applying its patch. Rejected rows are written back
            through ``property``, so they stay bit-identical only when
            ``property.set(s, property.get(s))`` round-trips exactly (under
            ``FRECHET_FILTER`` positions move by round-off). The optimizer state
            of rejected systems still advances; reset it before they update again.
    """

    potential: Potential[State, Any, Any, Any] = field(static=True)
    property: Lens[State, Params] = field(static=True)
    opt_state: Lens[State, OptState] = field(static=True)
    optimizer: Optimizer[Params, OptState] = field(static=True)
    mask: UpdateMask[State, Indices] | None = field(static=True, default=None)

    def __call__(self, key: Array, state: State) -> State:
        del key
        params = self.property.get(state)
        bs = next(
            (
                units.base_of(x)
                for x in units.tree_leaves(params)
                if units.is_quantity(x)
            ),
            units.active_base(),
        )
        params_in_bs = _rebased(params, bs)

        def value_and_grad_fn(p: Params) -> tuple[Any, Params]:
            trial = relabel_like(params_in_bs, p)
            out = self.potential(self.property.set(state, trial)).data
            return (
                magnitudes_of(_rebased(out.total_energies, bs)),
                magnitudes_of(_rebased(out.gradients, bs)),
            )

        potential_out = self.potential(state)
        grad = potential_out.data.gradients
        # Apply the patch
        energies = potential_out.data.total_energies
        state = potential_out.patch(
            state, energies.set_data(jnp.ones(len(energies), dtype=bool))
        )

        opt_state_current = self.opt_state.get(state)

        # grad, energies (the current per-system energies) and value_and_grad_fn
        # are the per-system objective the line-search transforms read; standard
        # transforms ignore them.
        mag_grad = magnitudes_of(_rebased(grad, bs))
        updates, new_opt_state = self.optimizer.update(
            mag_grad,
            opt_state_current,
            magnitudes_of(params_in_bs),
            grad=mag_grad,
            energies=energies.map_data(lambda e: units.magnitudes(e.rebase(bs))),
            value_and_grad_fn=value_and_grad_fn,
        )

        new_params = apply_updates(params, relabel_like(params_in_bs, updates))
        if self.mask is None:
            state = self.property.set(state, new_params)
        else:
            state = IndexLensPatch(
                new_params, self.mask.system_index(state), self.property
            )(state, self.mask.accept(state))
        state = self.opt_state.set(state, new_opt_state)
        return state


def _rebased[T](tree: T, bs: units.BaseSystem) -> T:
    """Express every quantity leaf of ``tree`` in base ``bs``."""
    return units.tree_map(lambda x: x.rebase(bs) if units.is_quantity(x) else x, tree)
