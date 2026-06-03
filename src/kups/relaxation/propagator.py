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
        PyTree: The type of the property being optimized (must match Potential's gradient type)

    Attributes:
        potential: Potential that computes energy and gradients of type PyTree
        property: Lens to get/set the property being optimized
        opt_state: Lens to get/set the optimizer state
        optimizer: Gradient transformation

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
