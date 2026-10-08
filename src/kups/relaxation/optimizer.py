# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Optimizer protocol, chain combinators and per-system reset layouts.

[ResetLayout][kups.relaxation.optimizer.ResetLayout] describes which optimizer
state to reset for selected systems. A
[Resettable][kups.relaxation.optimizer.Resettable] optimizer carries its own
layout, and [resettable_chain][kups.relaxation.optimizer.resettable_chain]
composes its members' layouts by position, so callers never pair layouts with
chain members by hand.

The config-spec layer (``Transform``, ``make_optimizer``, …) lives in
:mod:`kups.relaxation.config`, which depends on this module — keeping the
factory out of here avoids a circular import with
:mod:`kups.relaxation.transforms`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Protocol, no_type_check, override

import jax
import optax
from jax import Array

from kups.core.data.index import SupportsSorting
from kups.core.data.table import Table
from kups.core.lens import LambdaLens, Lens, View, const_lens
from kups.core.patch import IndexLensPatch
from kups.core.typing import PyTree, SystemId
from kups.core.utils.jax import dataclass, field


@dataclass(kw_only=True)
class ResetLayout[OptState, Data, Indices]:
    """Describe which optimizer fields to reset and which systems own their rows.

    Fresh values come from ``Optimizer.init``; this layout only describes how to
    select and mask them. Fields outside the lens, such as a shared history
    cursor, survive replacement.

    ``Data`` and ``Indices`` retain the concrete value and index-prefix types.
    Their pytree alignment is checked by ``IndexLensPatch`` when applied. The
    lens must select disjoint parts of the state, as for any inferred lens.

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

    @staticmethod
    def empty[S]() -> ResetLayout[S, tuple[()], tuple[()]]:
        """Layout of a state that holds nothing per system, so a reset keeps it."""
        return ResetLayout(fields=const_lens(()), system_index=lambda s: ())


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


class Resettable[Params, OptState](Optimizer[Params, OptState]):
    """Optimizer that knows which part of its state to reset per system.

    Implementers subclass ``Resettable`` explicitly. Membership is nominal, so
    the type checker binds each ``reset_layout`` to the optimizer's own state,
    and a subclass that omits the property cannot be instantiated. Every native
    transform is resettable and a chain of them built with
    [resettable_chain][kups.relaxation.optimizer.resettable_chain] composes its
    members' layouts. Optax transforms are not, so code that resets per-system
    state can require a ``Resettable`` and reject them statically;
    [Stateless][kups.relaxation.optimizer.Stateless] adapts stateless ones and
    rejects stateful ones when ``init`` runs.
    """

    @property
    @abstractmethod
    def reset_layout(self) -> ResetLayout[OptState, object, object]:
        """Layout of the per-system state that ``init`` would produce afresh.

        Implementations narrow the data and index types to their own.
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


@dataclass
class Stateless(Resettable[Any, PyTree]):
    """A stateless Optax transform as a member of a resettable chain.

    The transform keeps nothing between steps, so there is nothing to reset.
    ``init`` raises ``ValueError`` for a transform whose state has any leaves,
    since that state would go stale across resets. A leafless state, empty or a
    composite of empty states such as ``optax.sgd``'s, is passed back to the
    transform unchanged, and extra arguments are forwarded as in
    [chain][kups.relaxation.optimizer.chain]. Optax transforms accept any
    parameter pytree, so the adapter takes no type argument:
    ``Stateless(optax.scale(-1.0))``.

    Attributes:
        transform: The wrapped Optax transform.
    """

    transform: optax.GradientTransformation = field(static=True)

    @override
    def init(self, parameters: Any, index_prefix: PyTree | None = None) -> PyTree:
        state = self.transform.init(parameters)
        if jax.tree.leaves(state):
            raise ValueError(
                "Stateless wraps only transforms without state; this one keeps "
                "per-step state that a reset would not restore."
            )
        return state

    @override
    @no_type_check  # optax is not well typed
    def update(
        self,
        updates: Any,
        state: PyTree,
        params: Any = None,
        **extra_args: Any,
    ) -> tuple[Any, PyTree]:
        transform = optax.with_extra_args_support(self.transform)
        return transform.update(updates, state, params, **extra_args)

    @property
    @override
    def reset_layout(self) -> ResetLayout[PyTree, tuple[()], tuple[()]]:
        """Nothing to reset: the wrapped transform keeps no state."""
        return ResetLayout.empty()


@dataclass
class ResettableChain[Params](
    ChainOptimizer[Params], Resettable[Params, ChainOptState]
):
    """Chain whose reset layout is composed from its members' layouts.

    Built by [resettable_chain][kups.relaxation.optimizer.resettable_chain].
    Member ``i``'s layout applies to entry ``i`` of the chain state, so the
    pairing holds by construction; the chain's data and index types are erased
    to tuples.
    """

    optimizers: tuple[Resettable[Params, Any], ...]

    @property
    @override
    def reset_layout(
        self,
    ) -> ResetLayout[ChainOptState, tuple[Any, ...], tuple[Any, ...]]:
        layouts = tuple(optimizer.reset_layout for optimizer in self.optimizers)

        def get(states: ChainOptState) -> tuple[Any, ...]:
            return tuple(
                layout.fields.get(state)
                for layout, state in zip(layouts, states, strict=True)
            )

        def put(states: ChainOptState, /, value: tuple[Any, ...]) -> ChainOptState:
            return tuple(
                layout.fields.set(state, data)
                for layout, state, data in zip(layouts, states, value, strict=True)
            )

        return ResetLayout(
            fields=LambdaLens(get, put),
            system_index=lambda states: tuple(
                layout.system_index(state)
                for layout, state in zip(layouts, states, strict=True)
            ),
        )


def resettable_chain[Params](
    *optimizers: Resettable[Params, Any],
) -> ResettableChain[Params]:
    """Chain optimizers that each declare their reset layout.

    Unlike [chain][kups.relaxation.optimizer.chain], every member must be
    [Resettable][kups.relaxation.optimizer.Resettable], which the type checker
    enforces: a bare Optax transform has no reset layout. Wrap a stateless one
    in [Stateless][kups.relaxation.optimizer.Stateless], whose ``init`` raises
    ``ValueError`` for a transform that keeps state.
    """
    return ResettableChain(optimizers)
