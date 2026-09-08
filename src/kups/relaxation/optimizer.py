# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Optimizer protocol and chain combinator for relaxation transforms.

The config-spec layer (``Transform``, ``make_optimizer``, …) lives in
:mod:`kups.relaxation.config`, which depends on this module — keeping the
factory out of here avoids a circular import with
:mod:`kups.relaxation.transforms`.
"""

from typing import Any, Callable, Protocol, no_type_check, override

import optax
from jax import Array

from kups.core.data.index import SupportsSorting
from kups.core.data.table import Table
from kups.core.lens import Lens, View
from kups.core.typing import PyTree
from kups.core.utils.jax import dataclass, field


@dataclass(kw_only=True)
class ResetLayout[OptState, Data, Indices]:
    """Describe which optimizer fields to reset and which systems own their rows.

    Fresh values come from ``Optimizer.init``; this layout only describes how to
    select and mask them. Fields outside the lens, such as a shared history
    cursor, survive replacement.

    ``Data`` and ``Indices`` retain the concrete value and index-prefix types.
    Their pytree alignment is checked by ``IndexLensPatch`` when applied.

    Attributes:
        fields: Lens reading and writing the resettable part of optimizer state.
        system_index: View returning a matching pytree prefix of ``Index`` objects.
            Each index maps selected data rows to systems so ``IndexLensPatch``
            can apply the refill mask. Include reserved particle rows even when
            unoccupied. For FIRE, velocity uses particle-to-system indices;
            dt, alpha and counters use system indices.
    """

    fields: Lens[OptState, Data] = field(static=True)
    system_index: View[OptState, Indices] = field(static=True)


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


def apply_updates[Params](parameters: Params, updates: Params) -> Params:
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
