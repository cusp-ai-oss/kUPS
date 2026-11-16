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

from kups.core.data.table import Table
from kups.core.lens import Lens, View
from kups.core.patch import IndexLensPatch
from kups.core.potential import Potential
from kups.core.propagator import Propagator
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field
from kups.relaxation.optimizer import Optimizer, apply_updates


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

    Type Parameters:
        State: Simulation state.
        Params: Optimized parameters (must match the potential's gradient type).
        OptState: State type of ``optimizer``.
        Indices: Index prefix returned by the optional update mask.

    Attributes:
        potential: Potential that computes energy and gradients of type Params
        property: Lens to get/set the property being optimized
        opt_state: Lens to get/set the optimizer state
        optimizer: Gradient transformation
        mask: Optional acceptance and parameter-index views, read after evaluating
            the potential and applying its patch. Rejected rows are written back
            through ``property``, so they stay bit-identical only when
            ``property.set(s, property.get(s))`` round-trips exactly (under
            ``FRECHET_FILTER`` positions move by round-off). The optimizer state
            of rejected systems still advances; reset it before they update again.

    Example:
        ```python
        propagator = RelaxationPropagator(
            potential=my_potential,
            property=positions_lens,
            opt_state=lens(lambda s: s.opt_state),
            optimizer=chain(
                ScaleByAseLbfgs(memory_size=10),
                optax.scale(-1.0),
                ScaleByMoreThuenteLinesearch(),
            ),
        )
        state = propagator(key, state)
        ```
    """

    potential: Potential[State, Params, Any, Any] = field(static=True)
    property: Lens[State, Params] = field(static=True)
    opt_state: Lens[State, OptState] = field(static=True)
    optimizer: Optimizer[Params, OptState] = field(static=True)
    mask: UpdateMask[State, Indices] | None = field(static=True, default=None)

    def __call__(self, key: Array, state: State) -> State:
        del key
        params = self.property.get(state)

        def value_and_grad_fn(p: Params) -> tuple[Table[SystemId, Array], Params]:
            out = self.potential(self.property.set(state, p)).data
            return out.total_energies, out.gradients

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
        updates, new_opt_state = self.optimizer.update(
            grad,
            opt_state_current,
            params,
            grad=grad,
            energies=energies,
            value_and_grad_fn=value_and_grad_fn,
        )

        new_params = apply_updates(params, updates)
        if self.mask is None:
            state = self.property.set(state, new_params)
        else:
            state = IndexLensPatch(
                new_params, self.mask.system_index(state), self.property
            )(state, self.mask.accept(state))
        state = self.opt_state.set(state, new_opt_state)
        return state
