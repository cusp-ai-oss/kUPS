# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Optimizer protocol and chain combinator for relaxation transforms.

The config-spec layer (``Transform``, ``make_optimizer``, …) lives in
:mod:`kups.relaxation.config`, which depends on this module — keeping the
factory out of here avoids a circular import with
:mod:`kups.relaxation.transforms`.
"""

from typing import Any, Protocol, no_type_check, override

import optax

from kups.core import units
from kups.core.typing import PyTree
from kups.core.utils.jax import dataclass


class Optimizer[Params, OptState](Protocol):
    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> OptState: ...
    def update(
        self,
        updates: Params,
        state: OptState,
        params: Params | None = None,
        **kwargs: Any,
    ) -> tuple[Params, OptState]: ...


def relabel_like[T](reference: T, tree: T) -> T:
    """Give ``tree``'s leaves the dimensions of ``reference``'s.

    Args:
        reference: Pytree whose leaves carry the intended dimensions.
        tree: Pytree of magnitudes with ``reference``'s structure.

    Returns:
        ``tree`` with each leaf labelled like its counterpart: a quantity's
        dimension in its own base, and a plain leaf as it stands.
    """

    def like(ref: Any, x: Any) -> Any:
        """One leaf of magnitudes, labelled like its counterpart."""
        if units.is_quantity(ref):
            return units.from_dim(x, units.dim_of(ref), units.base_of(ref))
        return x

    return units.tree_map(like, reference, tree)


def apply_updates[Params](parameters: Params, updates: Params) -> Params:
    """Add each update to its parameter, keeping the parameter's dtype.

    Args:
        parameters: The parameter pytree, whose leaves may carry dimensions.
        updates: A pytree of the same structure, including each quantity's
            dimension and base.

    Returns:
        The updated parameters, with ``None`` leaves left in place.
    """
    return optax.apply_updates(parameters, updates)  # type: ignore


type ChainOptState = tuple[PyTree, ...]


@dataclass
class ChainOptimizer[Params](Optimizer[Params, ChainOptState]):
    optimizers: tuple[
        Optimizer[Params, PyTree] | optax.GradientTransformationExtraArgs, ...
    ]

    @override
    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> ChainOptState:
        states: list[PyTree] = []
        for optimizer in self.optimizers:
            if isinstance(optimizer, optax.GradientTransformation):
                state = optimizer.init(parameters)  # type: ignore
            else:
                state = optimizer.init(parameters, index_prefix)
            states.append(state)
        return tuple(states)

    @override
    @no_type_check  # optax is not well typed
    def update(
        self,
        updates: Params,
        state: ChainOptState,
        params: Params | None = None,
        **extra_args: Any,
    ) -> tuple[Params, ChainOptState]:
        new_states: list[PyTree] = []
        for optimizer, opt_state in zip(self.optimizers, state, strict=True):
            updates, new_opt_state = optimizer.update(
                updates, opt_state, params=params, **extra_args
            )
            new_states.append(new_opt_state)
        return updates, tuple(new_states)


def chain[Params](
    *optimizers: Optimizer[Params, PyTree] | optax.GradientTransformation,
) -> ChainOptimizer[Params]:
    return ChainOptimizer(
        tuple(
            (
                optax.with_extra_args_support(opt)
                if isinstance(opt, optax.GradientTransformation)
                else opt
            )
            for opt in optimizers
        )
    )
