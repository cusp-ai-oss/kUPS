# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unified interface for graph-based machine learning interatomic potentials (MLIAPs).

This module provides generic protocols and factory functions for integrating
graph-based ML potentials (MACE, NequIP, Allegro, etc.) into kUPS.

Models can return:
- Energy only (Gradients=EmptyType): autodiff computes forces and Hessians
- Energy + forces (Gradients=Array): model-provided forces, optional autodiff for Hessians
- Energy + forces + virials (Gradients=VirialTheoremGradients): full stress support

Example (autodiff forces):
    ```python
    from kups.potential.mliap.interface import make_mliap_potential

    def my_energy_fn(inp: MLIAPInput) -> WithPatch[PotentialOut[EmptyType, EmptyType], IdPatch]:
        energy = model(inp.graph)
        return WithPatch(PotentialOut(energy, EMPTY, EMPTY), IdPatch())

    potential = make_mliap_potential(my_energy_fn, gradient_lens=..., ...)
    ```

Example (precomputed forces):
    ```python
    def my_forces_fn(inp: MLIAPInput) -> WithPatch[PotentialOut[Array, EmptyType], IdPatch]:
        energy, forces = model(inp.graph)
        return WithPatch(PotentialOut(energy, -forces, EMPTY), IdPatch())

    potential = make_mliap_potential(my_forces_fn, ...)  # No gradient_lens needed
    ```
"""

from __future__ import annotations

import functools
from typing import Any, Literal, Protocol, overload

from jax import Array

from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import NearestNeighborList
from kups.core.patch import Patch, WithPatch
from kups.core.potential import EMPTY_LENS, EmptyType, Potential, PotentialOut
from kups.core.typing import (
    HasPositionsAndSystemIndex,
    HasUnitCell,
    ParticleId,
    SystemId,
)
from kups.potential.common.direct import DirectPotential
from kups.potential.common.energy import PotentialFromEnergy
from kups.potential.common.graph import (
    FullGraphSumComposer,
    GraphPotentialInput,
    IsRadiusGraphPoints,
    RadiusGraphConstructor,
)

type MLIAPInput[
    Model,
    P: HasPositionsAndSystemIndex,
    S: HasUnitCell,
] = GraphPotentialInput[Model, P, S, Literal[2]]


def _extract_energy(
    model_fn: ModelFunction, inp: GraphPotentialInput
) -> WithPatch[Any, Any]:
    """Extract energy from a model function's PotentialOut result."""
    result = model_fn(inp)
    return WithPatch(result.data.total_energies, result.patch)


class ModelFunction[
    Model,
    Gradients,
    Hessians,
    P: HasPositionsAndSystemIndex,
    S: HasUnitCell,
    Ptch: Patch,
](Protocol):
    """Protocol for MLIAP model functions.

    Type parameter semantics:
    - Gradients=EmptyType, Hessians=EmptyType: energy only (autodiff required for forces)
    - Gradients=Array, Hessians=EmptyType: energy + forces (precomputed)
    - Gradients=VirialTheoremGradients, Hessians=EmptyType: energy + forces + virials
    """

    def __call__(
        self, inp: MLIAPInput[Model, P, S]
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Ptch]: ...


@overload
def make_mliap_potential[
    Model,
    State,
    Gradients,
    Hessians,
    P: IsRadiusGraphPoints,
    S: HasUnitCell,
    Ptch: Patch,
](
    model_fn: ModelFunction[Model, EmptyType, EmptyType, P, S, Ptch],
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NearestNeighborList],
    model_view: View[State, Model],
    cutoffs_view: View[State, Table[SystemId, Array]],
    gradient_lens: Lens[MLIAPInput[Model, P, S], Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    *,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Patch[State]]: ...


@overload
def make_mliap_potential[
    Model,
    State,
    Gradients,
    P: IsRadiusGraphPoints,
    S: HasUnitCell,
    Ptch: Patch,
](
    model_fn: ModelFunction[Model, EmptyType, EmptyType, P, S, Ptch],
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NearestNeighborList],
    model_view: View[State, Model],
    cutoffs_view: View[State, Table[SystemId, Array]],
    gradient_lens: Lens[MLIAPInput[Model, P, S], Gradients],
    *,
    patch_idx_view: View[State, PotentialOut[Gradients, EmptyType]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, EmptyType]] | None = None,
) -> Potential[State, Gradients, EmptyType, Patch[State]]: ...


@overload
def make_mliap_potential[
    Model,
    State,
    Gradients,
    Hessians,
    P: IsRadiusGraphPoints,
    S: HasUnitCell,
    Ptch: Patch,
](
    model_fn: ModelFunction[Model, Gradients, Hessians, P, S, Ptch],
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NearestNeighborList],
    model_view: View[State, Model],
    cutoffs_view: View[State, Table[SystemId, Array]],
    *,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Patch[State]]: ...


def make_mliap_potential(
    model_fn: Any,
    particles_view: Any,
    systems_view: Any,
    neighborlist_view: Any,
    model_view: Any,
    cutoffs_view: Any = None,
    gradient_lens: Any = None,
    hessian_lens: Any = None,
    hessian_idx_view: Any = None,
    patch_idx_view: Any = None,
    out_cache_lens: Any = None,
) -> Any:
    """Create a graph-based MLIAP potential.

    Three modes based on arguments:
    - gradient_lens + hessian_lens + hessian_idx_view: autodiff with hessians
    - gradient_lens only: autodiff without hessians
    - neither: model provides gradients directly via DirectPotential

    Args:
        model_fn: Function returning PotentialOut with energy (and optionally
            gradients).
        particles_view: View to extract particles from state.
        systems_view: View to extract systems (unit cell) from state.
        neighborlist_view: View to extract neighbor list from state.
        model_view: View to extract model from state.
        cutoffs_view: View to extract cutoffs as ``Indexed[SystemId, Array]``.
        gradient_lens: Lens for gradient computation (None = use model's
            gradients directly via DirectPotential).
        hessian_lens: Lens selecting gradients for Hessian computation.
        hessian_idx_view: View for Hessian row/column indices from state.
        patch_idx_view: View for cached output indices (optional).
        out_cache_lens: Lens for output cache (optional).
    """
    radius_graph_fn = RadiusGraphConstructor(
        particles=particles_view,
        systems=systems_view,
        cutoffs=cutoffs_view,
        neighborlist=neighborlist_view,
        probe=None,
    )
    composer = FullGraphSumComposer(radius_graph_fn, model_view)

    if gradient_lens is not None:
        return PotentialFromEnergy(
            composer=composer,
            energy_fn=functools.partial(_extract_energy, model_fn),
            gradient_lens=gradient_lens,
            hessian_lens=hessian_lens if hessian_lens is not None else EMPTY_LENS,
            hessian_idx_view=hessian_idx_view
            if hessian_idx_view is not None
            else EMPTY_LENS,
            cache_lens=out_cache_lens,
            patch_idx_view=patch_idx_view,
        )
    else:
        return DirectPotential(
            direct_potential_fn=model_fn,
            composer=composer,
            cache_lens=out_cache_lens,
            patch_idx_view=patch_idx_view,
        )
