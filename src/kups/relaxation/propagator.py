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
    ``system_index`` returns an index prefix of the optimized parameters.
    """

    accept: View[State, Table[SystemId, Array]] = field(static=True)
    system_index: View[State, Indices] = field(static=True)


@dataclass
class RelaxationPropagator[State, Params, OptState, Indices](Propagator[State]):
    """Unified propagator for gradient-based optimization.

    Uses a Potential to compute energy and gradients. Supports standard optax
    optimizers (Adam, SGD) and the per-system line searches in
    :mod:`kups.relaxation.transforms` (backtracking, More-Thuente).

    Each step it passes the optimizer the gradient, the current per-system
    energies, and a ``value_and_grad_fn`` for evaluating trial points; the line
    searches use these to evaluate the objective along the search direction,
    standard transforms ignore them.

    After computing energy and gradients, the potential's patch is applied to the
    state. This allows potentials to update internal state (e.g., neighbor lists)
    at each relaxation step.

    Type Parameters:
        State: The simulation state type
        Params: Optimized parameters (must match the potential's gradient type).
        OptState: Optimizer state preserved through initialization and updates.
        Indices: Index prefix returned by the optional update mask.

    Attributes:
        potential: Potential that computes energy and gradients of type Params
        property: Lens to get/set the property being optimized
        opt_state: Lens to get/set the optimizer state
        optimizer: Gradient transformation
        mask: Optional acceptance and parameter-index views, read after evaluating
            the potential and applying its patch. Optimizer state still advances;
            a reused slot must be reset before its next update.

    Example:
        ```python
        import optax
        from kups.relaxation.optimizer import chain
        from kups.relaxation.propagator import RelaxationPropagator
        from kups.relaxation.transforms import ScaleByAseLbfgs, ScaleByMoreThuenteLinesearch

        # Standard optimizer (Adam)
        propagator = RelaxationPropagator(
            potential=my_potential,
            property=positions_lens,
            opt_state=lens(lambda s: s.opt_state),
            optimizer=optax.adam(0.01),
        )

        # L-BFGS with a per-system strong-Wolfe line search
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

        state = propagator(key, state)  # One optimization step
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
