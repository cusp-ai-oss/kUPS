# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Optimizer protocol and chain combinator for relaxation transforms.

The config-spec layer (``Transform``, ``make_optimizer``, …) lives in
:mod:`kups.relaxation.config`, which depends on this module — keeping the
factory out of here avoids a circular import with
:mod:`kups.relaxation.transforms`.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, no_type_check, override

import optax
from jax import Array

from kups.core.data.index import SupportsSorting
from kups.core.data.table import Table
from kups.core.lens import Lens, View
from kups.core.patch import IndexLensPatch
from kups.core.typing import PyTree, SystemId
from kups.core.utils.jax import dataclass, field


@dataclass(kw_only=True)
class ResetLayout[OptState, Data, Indices]:
    """Optimizer state to reset per system, and the system that owns each row.

    ``reset`` copies ``fields`` from a fresh ``init`` state into the systems
    selected by ``mask``. State outside ``fields``, such as the L-BFGS ring
    cursor ``count``, keeps its value.

    Attributes:
        fields: Lens onto the resettable state. Its focus paths must not overlap.
        system_index: View returning a pytree prefix of the ``fields`` value
            whose ``Index`` leaves map each row to its system. For FIRE,
            velocity rows are particles and ``dt`` rows are systems. Map padding
            rows to their system too. ``reset`` raises if the two pytrees do not
            align.
    """

    fields: Lens[OptState, Data] = field(static=True)
    system_index: View[OptState, Indices] = field(static=True)

    def within[Outer](
        self, outer: Lens[Outer, OptState]
    ) -> ResetLayout[Outer, Data, Indices]:
        """Lift this layout into a larger state holding ``OptState`` at ``outer``."""
        return ResetLayout(
            fields=outer.nest(self.fields),
            system_index=lambda s: self.system_index(outer.get(s)),
        )

    def merge[Data2, Indices2](
        self, other: ResetLayout[OptState, Data2, Indices2]
    ) -> ResetLayout[OptState, tuple[Data, Data2], tuple[Indices, Indices2]]:
        """Reset the fields of both layouts."""
        return ResetLayout(
            fields=self.fields.merge(other.fields),
            system_index=lambda s: (self.system_index(s), other.system_index(s)),
        )

    def reset(
        self, state: OptState, fresh: OptState, mask: Table[SystemId, Array]
    ) -> OptState:
        """Copy ``fresh`` field values into the systems selected by ``mask``."""
        return IndexLensPatch(
            self.fields.get(fresh), self.system_index(state), self.fields
        )(state, mask)


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
        """One optimization step.

        [RelaxationPropagator][kups.relaxation.propagator.RelaxationPropagator]
        also passes the raw gradient ``grad``, the per-system ``energies`` and
        ``value_and_grad_fn``, which evaluates trial points. Line searches use
        them; other transforms ignore them.
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
