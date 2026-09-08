# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Optimizer protocol and chain combinator for relaxation transforms.

The config-spec layer (``Transform``, ``make_optimizer``, …) lives in
:mod:`kups.relaxation.config`, which depends on this module — keeping the
factory out of here avoids a circular import with
:mod:`kups.relaxation.transforms`.
"""

from typing import Any, Callable, Protocol, no_type_check, override, runtime_checkable

import jax
import optax
from jax import Array

from kups.core.data.index import SupportsSorting
from kups.core.data.table import Table
from kups.core.typing import PyTree
from kups.core.utils.jax import dataclass

type SystemMask = Table[SupportsSorting, Array]
"""Per-system boolean mask keyed like the ``index_prefix`` leaves."""


class Optimizer[Params, OptState](Protocol):
    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> OptState: ...
    def update(
        self,
        updates: Params,
        state: OptState,
        params: Params | None = None,
        *,
        grad: Params | None = None,
        energies: Table[SupportsSorting, Array] | None = None,
        value_and_grad_fn: Callable[
            [Params], tuple[Table[SupportsSorting, Array], Params]
        ]
        | None = None,
        **kwargs: Any,
    ) -> tuple[Params, OptState]:
        """One optimisation step.

        Besides the optax arguments, :class:`kups.relaxation.propagator.RelaxationPropagator`
        passes the raw gradient ``grad``, the per-system ``energies`` and a
        ``value_and_grad_fn`` evaluating trial points; line searches consume
        them, plain transforms ignore them.
        """
        ...


@runtime_checkable
class SupportsReset[Params, OptState](Protocol):
    """Optional optimizer capability required when recycling individual systems."""

    def reset(
        self,
        state: OptState,
        parameters: Params,
        index_prefix: PyTree,
        mask: SystemMask,
    ) -> OptState:
        """Reset selected systems using the transform's explicit state layout."""
        ...


def apply_updates[Params](parameters: Params, updates: Params) -> Params:
    return optax.apply_updates(parameters, updates)  # type: ignore


type ChainOptState = tuple[PyTree, ...]


@dataclass
class ChainOptimizer[Params](
    Optimizer[Params, ChainOptState], SupportsReset[Params, ChainOptState]
):
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

    @override
    def reset(
        self,
        state: ChainOptState,
        parameters: Params,
        index_prefix: PyTree,
        mask: SystemMask,
    ) -> ChainOptState:
        new_states: list[PyTree] = []
        for optimizer, opt_state in zip(self.optimizers, state, strict=True):
            if isinstance(optimizer, optax.GradientTransformation):
                if jax.tree.leaves(opt_state):
                    raise ValueError(
                        "Streaming requires per-system optimizer state; "
                        "stateful Optax transforms need an explicit reset adapter."
                    )
                new_state = opt_state
            elif isinstance(optimizer, SupportsReset):
                new_state = optimizer.reset(opt_state, parameters, index_prefix, mask)
            else:
                raise ValueError(
                    f"{type(optimizer).__name__} does not support per-system reset."
                )
            new_states.append(new_state)
        return tuple(new_states)


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
