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
import optax
from jax import Array

from kups.core.lens import Lens
from kups.core.potential import Potential
from kups.core.propagator import Propagator
from kups.core.utils.jax import dataclass, field


@dataclass
class RelaxationPropagator[State, PyTree](Propagator[State]):
    """Unified propagator for gradient-based optimization using Optax.

    Uses a Potential to compute energy and gradients. Supports both standard
    optimizers (Adam, SGD) and line-search optimizers (L-BFGS, backtracking).

    For line-search optimizers, the potential is evaluated at trial points during
    the line search. For standard optimizers, it's evaluated once per step.

    After computing energy and gradients, the potential's patch is applied to the
    state. This allows potentials to update internal state (e.g., neighbor lists)
    at each relaxation step.

    Type Parameters:
        State: The simulation state type
        PyTree: The type of the property being optimized (must match Potential's gradient type)

    Attributes:
        potential: Potential that computes energy and gradients of type PyTree
        property: Lens to get/set the property being optimized
        opt_state: Lens to get/set the Optax optimizer state
        optimizer: Optax gradient transformation

    Example:
        ```python
        import optax
        from kups.relaxation.propagator import RelaxationPropagator
        from kups.core.potential import MappedPotential

        # Standard optimizer (Adam)
        propagator = RelaxationPropagator(
            potential=my_potential,
            property=positions_lens,
            opt_state=lens(lambda s: s.opt_state),
            optimizer=optax.adam(0.01),
        )

        # Line-search optimizer (L-BFGS)
        propagator = RelaxationPropagator(
            potential=my_potential,
            property=positions_lens,
            opt_state=lens(lambda s: s.opt_state),
            optimizer=optax.lbfgs(),
        )

        # With gradient projection
        mapped_potential = MappedPotential(
            full_potential,
            gradient_map=lambda g: g.positions,
            hessian_map=lambda h: h,
        )
        propagator = RelaxationPropagator(
            potential=mapped_potential,
            property=positions_lens,
            opt_state=lens(lambda s: s.opt_state),
            optimizer=optax.lbfgs(),
        )

        state = propagator(key, state)  # One optimization step
        ```
    """

    potential: Potential[State, PyTree, Any, Any] = field(static=True)
    property: Lens[State, PyTree] = field(static=True)
    opt_state: Lens[State, optax.OptState] = field(static=True)
    optimizer: optax.GradientTransformationExtraArgs = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        params = self.property.get(state)

        def value_fn(p: PyTree) -> Array:
            updated_state = self.property.set(state, p)
            result = self.potential(updated_state)
            return result.data.total_energies.data.sum()

        potential_out = self.potential(state)
        value = potential_out.data.total_energies.data.sum()
        grad = potential_out.data.gradients
        # Apply the patch
        energies = potential_out.data.total_energies
        state = potential_out.patch(
            state, energies.set_data(jnp.ones(len(energies), dtype=bool))
        )

        opt_state_current = self.opt_state.get(state)

        updates, new_opt_state = self.optimizer.update(
            grad,  # type: ignore - optax typing
            opt_state_current,
            params,  # type: ignore - optax typing
            value=value,
            grad=grad,
            value_fn=value_fn,  # necessary for line-search optimizers
        )

        new_params: PyTree = optax.apply_updates(params, updates)  # type: ignore
        state = self.property.set(state, new_params)
        state = self.opt_state.set(state, new_opt_state)
        return state
