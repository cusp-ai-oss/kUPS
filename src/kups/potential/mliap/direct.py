# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Factory for graph-based MLIAPs whose models output gradients directly.

Bridges a torch- or JAX-side ``model_fn`` that returns a ``PotentialOut``
(energy + gradients + hessians) into a kUPS
[Potential][kups.core.potential.Potential] via
[DirectPotential][kups.potential.common.direct.DirectPotential].

This module covers the "direct" branch only: the model produces the gradients
(forces, virials, …) itself. For energy-only models that should be
differentiated via JAX autodiff, construct
[PotentialFromEnergy][kups.potential.common.energy.PotentialFromEnergy]
directly — see [tojax][kups.potential.mliap.tojax] for that pattern.

Example:
    ```python
    from kups.potential.mliap.direct import make_direct_mliap_potential

    def my_forces_fn(inp: DirectMliapInput) -> WithPatch[PotentialOut[Array, EmptyType], IdPatch]:
        energy, forces = model(inp.graph)
        return WithPatch(PotentialOut(energy, -forces, EMPTY), IdPatch())

    potential = make_direct_mliap_potential(my_forces_fn, ...)
    ```
"""

from __future__ import annotations

from typing import Literal, Protocol

from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import NeighborList
from kups.core.patch import Patch, WithPatch
from kups.core.potential import Potential, PotentialOut
from kups.core.typing import (
    HasCell,
    HasPositionsAndSystemIndex,
    ParticleId,
    SystemId,
)
from kups.potential.common.direct import DirectPotential
from kups.potential.common.graph import (
    FullGraphSumComposer,
    GraphPotentialInput,
    IsRadiusGraphPoints,
    RadiusGraphConstructor,
)

type DirectMliapInput[
    Model,
    P: HasPositionsAndSystemIndex,
    S: HasCell,
] = GraphPotentialInput[Model, P, S, Literal[2]]


class DirectMliapFn[
    Model,
    Gradients,
    Hessians,
    P: HasPositionsAndSystemIndex,
    S: HasCell,
    Ptch: Patch,
](Protocol):
    """Protocol for a direct MLIAP model function.

    Returns a ``PotentialOut`` that bundles energy, gradients and (optionally)
    hessians for one graph input. Conventional ``Gradients`` payloads:

    - ``Array``: position gradients only (``∂E/∂r``).
    - ``PositionAndCell``: position + cell gradients (forces + stress).
    - ``EmptyType``: no gradients — but in that case the autodiff path
      ([PotentialFromEnergy][kups.potential.common.energy.PotentialFromEnergy])
      is more natural; this module is for the gradient-producing case.
    """

    def __call__(
        self, inp: DirectMliapInput[Model, P, S]
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Ptch]: ...


def make_direct_mliap_potential[
    Model,
    State,
    Gradients,
    Hessians,
    P: IsRadiusGraphPoints,
    S: HasCell,
    Ptch: Patch,
](
    model_fn: DirectMliapFn[Model, Gradients, Hessians, P, S, Ptch],
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    model_view: View[State, Model],
    *,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Patch[State]]:
    """Wrap a direct-gradient ``model_fn`` into a kUPS ``Potential``.

    Args:
        model_fn: Direct MLIAP function — see
            [DirectMliapFn][kups.potential.mliap.direct.DirectMliapFn].
        particles_view: View to extract particles from state.
        systems_view: View to extract systems (cell) from state.
        neighborlist_view: View to extract a cutoff-bound neighbor list from state.
        model_view: View to extract model from state.
        patch_idx_view: View for cached output indices (optional).
        out_cache_lens: Lens for output cache (optional).

    Returns:
        Configured kUPS ``Potential`` backed by ``DirectPotential``.
    """
    composer = FullGraphSumComposer(
        RadiusGraphConstructor(
            particles=particles_view,
            systems=systems_view,
            neighborlist=neighborlist_view,
            probe=None,
        ),
        model_view,
    )
    return DirectPotential(
        direct_potential_fn=model_fn,
        composer=composer,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
