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

from kups.core.lens import Lens
from kups.core.potential import Potential
from kups.core.propagator import Propagator
from kups.core.utils.jax import dataclass, field
from kups.relaxation.optimizer import Optimizer, apply_updates


@dataclass
class RelaxationPropagator[State, PyTree, OptState](Propagator[State]):
    """Propagator for one step of gradient-based optimization.

    Works with optax optimizers and with the per-system line searches in
    :mod:`kups.relaxation.transforms`. Each step passes the optimizer the
    gradient, the per-system energies and a ``value_and_grad_fn`` for trial
    points, which line searches use and other transforms ignore. The
    potential's patch is applied each step so it can update state such as
    neighbor lists.

    Attributes:
        potential: Potential that computes energy and gradients of type PyTree
        property: Lens to get/set the property being optimized
        opt_state: Lens to get/set the optimizer state
        optimizer: Gradient transformation

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

    potential: Potential[State, PyTree, Any, Any] = field(static=True)
    property: Lens[State, PyTree] = field(static=True)
    opt_state: Lens[State, OptState] = field(static=True)
    optimizer: Optimizer[PyTree, OptState] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        params = self.property.get(state)

        def value_and_grad_fn(p: PyTree) -> tuple[Any, PyTree]:
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
        state = self.property.set(state, new_params)
        state = self.opt_state.set(state, new_opt_state)
        return state
