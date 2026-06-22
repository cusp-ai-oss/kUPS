# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for local MLIAP potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, neighbor list, and model into the
state-agnostic factory in [kups.potential.mliap.local][].

The model may live on the state (``state.local_mliap_model``) or be passed
directly via ``parameters=``; in the latter case it is bound with a constant
lens and the state need not carry a model field. For incremental (probe) updates
with a constant model, the cache is read from a conventional ``local_mliap_cache``
attribute.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

import jax.numpy as jnp

from kups.core.cell import AnyPeriodicity
from kups.core.lens import Lens, bind, const_lens
from kups.core.neighborlist import (
    IsAdaptiveCutoffNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
    adaptive_cutoff_neighborlist_from_state,
)
from kups.core.patch import Patch, Probe
from kups.core.potential import EMPTY_LENS, Potential, PotentialOut
from kups.core.typing import HasCache, HasCell, IsState, MaybeCached
from kups.potential.common.graph import IsGraphProbe
from kups.potential.mliap.local import (
    IsLocalMLIAPGraphParticles,
    LocalMLIAPData,
    LocalMLIAPInput,
    make_local_mliap_potential,
)


class IsLocalMLIAPGraphState(
    IsState[IsLocalMLIAPGraphParticles, HasCell[AnyPeriodicity]],
    IsAdaptiveCutoffNeighborListState[IsUniversalNeighborlistParams],
    Protocol,
):
    """Particles, systems, and neighbor list for a local MLIAP graph (no model)."""


class IsLocalMLIAPState[Model](IsLocalMLIAPGraphState, Protocol):
    """:class:`IsLocalMLIAPGraphState` that also carries the MLIAP model on the state."""

    @property
    def local_mliap_model(self) -> Model: ...


class IsCachedLocalMLIAPState[Cache](IsLocalMLIAPGraphState, Protocol):
    """:class:`IsLocalMLIAPGraphState` carrying an incremental-update cache (model passed in)."""

    @property
    def local_mliap_cache(self) -> Cache: ...


@overload
def make_local_mliap_from_state[State, Gradient, Hessian](
    state: Lens[State, IsLocalMLIAPState[MaybeCached[LocalMLIAPData, Any]]],
    probe: None = None,
    gradient_lens: Lens[
        LocalMLIAPInput[State, IsLocalMLIAPGraphParticles, HasCell[AnyPeriodicity]],
        Gradient,
    ] = ...,
    hessian_lens: Lens[Gradient, Hessian] = ...,
    hessian_idx_view: Lens[State, Hessian] = ...,
    out_idx_view: None = None,
    *,
    parameters: None = None,
    neighborlist_factory: NeighborListFactory[
        IsLocalMLIAPState[MaybeCached[LocalMLIAPData, Any]]
    ] = ...,
) -> Potential[State, Gradient, Hessian, Patch[Any]]: ...


@overload
def make_local_mliap_from_state[State, Ptch: Patch[Any], Gradient, Hessian](
    state: Lens[
        State,
        IsLocalMLIAPState[HasCache[LocalMLIAPData, PotentialOut[Gradient, Hessian]]],
    ],
    probe: Probe[State, Ptch, IsGraphProbe[IsLocalMLIAPGraphParticles, Literal[2]]],
    gradient_lens: Lens[
        LocalMLIAPInput[State, IsLocalMLIAPGraphParticles, HasCell[AnyPeriodicity]],
        Gradient,
    ] = ...,
    hessian_lens: Lens[Gradient, Hessian] = ...,
    hessian_idx_view: Lens[State, Hessian] = ...,
    out_idx_view: Lens[State, PotentialOut[Gradient, Hessian]] | None = None,
    *,
    parameters: None = None,
    neighborlist_factory: NeighborListFactory[
        IsLocalMLIAPState[HasCache[LocalMLIAPData, PotentialOut[Gradient, Hessian]]]
    ] = ...,
) -> Potential[State, Gradient, Hessian, Ptch]: ...


@overload
def make_local_mliap_from_state[State, Gradient, Hessian](
    state: Lens[State, IsLocalMLIAPGraphState],
    probe: None = None,
    gradient_lens: Lens[
        LocalMLIAPInput[State, IsLocalMLIAPGraphParticles, HasCell[AnyPeriodicity]],
        Gradient,
    ] = ...,
    hessian_lens: Lens[Gradient, Hessian] = ...,
    hessian_idx_view: Lens[State, Hessian] = ...,
    out_idx_view: None = None,
    *,
    parameters: LocalMLIAPData,
    neighborlist_factory: NeighborListFactory[IsLocalMLIAPGraphState] = ...,
) -> Potential[State, Gradient, Hessian, Patch[Any]]: ...


@overload
def make_local_mliap_from_state[State, Ptch: Patch[Any], Gradient, Hessian](
    state: Lens[State, IsCachedLocalMLIAPState[PotentialOut[Gradient, Hessian]]],
    probe: Probe[State, Ptch, IsGraphProbe[IsLocalMLIAPGraphParticles, Literal[2]]],
    gradient_lens: Lens[
        LocalMLIAPInput[State, IsLocalMLIAPGraphParticles, HasCell[AnyPeriodicity]],
        Gradient,
    ] = ...,
    hessian_lens: Lens[Gradient, Hessian] = ...,
    hessian_idx_view: Lens[State, Hessian] = ...,
    out_idx_view: Lens[State, PotentialOut[Gradient, Hessian]] | None = None,
    *,
    parameters: LocalMLIAPData,
    neighborlist_factory: NeighborListFactory[
        IsCachedLocalMLIAPState[PotentialOut[Gradient, Hessian]]
    ] = ...,
) -> Potential[State, Gradient, Hessian, Ptch]: ...


def make_local_mliap_from_state(
    state: Any,
    probe: Any = None,
    gradient_lens: Any = EMPTY_LENS,
    hessian_lens: Any = EMPTY_LENS,
    hessian_idx_view: Any = EMPTY_LENS,
    out_idx_view: Any = None,
    *,
    parameters: LocalMLIAPData | None = None,
    neighborlist_factory: NeighborListFactory[
        Any
    ] = adaptive_cutoff_neighborlist_from_state,
) -> Any:
    """Create a local MLIAP potential from a typed state, optionally with incremental updates.

    Convenience wrapper around
    [make_local_mliap_potential][kups.potential.mliap.local.make_local_mliap_potential].
    When ``probe`` is ``None``, extracts views from a state satisfying
    [IsLocalMLIAPState][kups.application.potential.mliap.local.IsLocalMLIAPState].
    When ``probe`` is provided, additionally wires the ``PotentialOut`` cache for
    efficient incremental caching across Monte Carlo steps.

    Args:
        state: Lens into the sub-state providing particles, cell, neighbor list,
            and local MLIAP model (when ``parameters`` is not given).
        probe: Detects which particles and neighbor-list edges changed since the last
            step. ``None`` for full-only computation.
        gradient_lens: Specifies which gradients to compute (e.g., forces).
        hessian_lens: Specifies which Hessians to compute.
        hessian_idx_view: Index structure for Hessian updates.
        out_idx_view: Index into the cached output for partial updates. Only used when
            ``probe`` is not ``None``. Defaults to full re-indexing of the cached
            ``total_energies``.
        parameters: Constant MLIAP model. When given it is bound with a constant
            lens and the state need not carry ``local_mliap_model``; with a
            ``probe``, the cache is read from ``state.local_mliap_cache``.

    Returns:
        Configured local MLIAP [Potential][kups.core.potential.Potential].
    """

    def _model(x: Any) -> LocalMLIAPData:
        m = x.local_mliap_model
        return m.data if isinstance(m, HasCache) else m

    if probe is None:
        if parameters is not None:
            model_lens = const_lens(parameters)
        else:
            model_lens = state.focus(_model)

        def neighborlist_view(s: Any) -> NeighborList[Literal[2]]:
            return neighborlist_factory(state(s), model_lens(s).cutoff)

        return make_local_mliap_potential(
            state.focus(lambda x: x.particles),
            state.focus(lambda x: x.systems),
            neighborlist_view,
            model_lens,
            None,
            gradient_lens,
            hessian_lens,
            hessian_idx_view,
            None,
            None,
        )

    if parameters is not None:
        model_lens = const_lens(parameters)
        cache_view = state.focus(lambda x: x.local_mliap_cache)
    else:
        model_lens = state.focus(lambda x: x.local_mliap_model.data)
        cache_view = state.focus(lambda x: x.local_mliap_model.cache)

    if out_idx_view is None:
        out_idx_view = cache_view.focus(
            lambda c: bind(c, lambda x: x.total_energies.data).apply(
                lambda x: jnp.arange(x.size, dtype=int)
            )
        )

    def neighborlist_view(s: Any) -> NeighborList[Literal[2]]:
        return neighborlist_factory(state(s), model_lens(s).cutoff)

    return make_local_mliap_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        neighborlist_view,
        model_lens,
        probe,
        gradient_lens,
        hessian_lens,
        hessian_idx_view,
        out_idx_view,
        cache_view,
    )
