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

from typing import Any, Literal, Protocol, overload

from kups.core import units
from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import NeighborList, NeighborListPoints
from kups.core.patch import Patch, WithPatch
from kups.core.potential import Potential, PotentialOut
from kups.core.typing import (
    HasCell,
    HasPositionsAndSystemIndex,
    ParticleId,
    SystemId,
)
from kups.core.utils.kahan import KahanSummand
from kups.potential.common.direct import DirectPotential
from kups.potential.common.energy import FullSumComposer
from kups.potential.common.geometry import Geometry, PositionsAndCell
from kups.potential.common.graph import (
    GraphConstructor,
    GraphInputConstructor,
    GraphPotentialInput,
)


@overload
def filter_pullback(
    geometry: Geometry,
    cotangents: PositionsAndCell[units.Force],
    gradient: Lens[Geometry, PositionsAndCell[units.Length]],
) -> PositionsAndCell[units.Force]: ...
@overload
def filter_pullback(
    geometry: Geometry,
    cotangents: PositionsAndCell[units.Force],
    gradient: Lens[Geometry, Any],
) -> Any: ...
def filter_pullback(
    geometry: Geometry,
    cotangents: PositionsAndCell[units.Force],
    gradient: Lens[Geometry, Any],
) -> Any:
    """Map direct ``(∂E/∂r, ∂E/∂h|_r)`` outputs to the DOF gradient ``∂E/∂u``.

    Carrier- and codomain-agnostic: the ``gradient`` lens's ``set`` is the map
    ``u → pose``, and differentiating the energy ``⟨∂E/∂r, r(u)⟩ + ⟨∂E/∂h, h(u)⟩``
    it defines yields ``∂E/∂u`` for any codomain ``U``, with the cell-factor and
    ``expm`` chain rule (and the atoms-ride-the-cell coupling) falling out of the
    derivative.

    Args:
        geometry: The carrier's geometric view (e.g. ``GRAPH_GEOMETRY.get(inp)``).
        cotangents: Direct outputs — ``positions`` is the ``∂E/∂r`` block; ``cell``
            carries the partial ``∂E/∂h|_r`` in its frame parameters.
        gradient: Relaxation filter ``Lens[Geometry, U]`` selecting the DOFs ``u``.

    Returns:
        ``∂E/∂u`` in the gradient lens's DOF pytree, each leaf in energy over its
        DOF's dimension (``PositionsAndCell[units.Force]`` for the standard filters).
    """
    u0 = gradient.get(geometry)
    g_r = cotangents.positions.data
    # Recover the raw cartesian ∂E/∂h: map the partial gradient's frame
    # parameters back through the real cell's frame Jacobian.
    dE_dh = geometry.systems.data.frame.vectors_gradient(cotangents.cell.data.frame)

    def contracted(u: Any) -> units.Energy:
        """The energy this gradient pair assigns to the pose ``u`` names."""
        pose = gradient.set(geometry, u)
        return units.sum(pose.particles.data.positions * g_r) + units.sum(
            pose.systems.data.vectors * dE_dh
        )

    return units.grad(contracted)(u0)


type DirectMliapInput[
    Model,
    P: HasPositionsAndSystemIndex,
    S: HasCell[AnyPeriodicity],
] = GraphPotentialInput[Model, P, S, Literal[2]]


class DirectMliapFn[
    Model,
    Gradients,
    Hessians,
    P: HasPositionsAndSystemIndex,
    S: HasCell[AnyPeriodicity],
    Ptch: Patch[Any],
](Protocol):
    """Protocol for a direct MLIAP model function.

    Returns a ``PotentialOut`` that bundles energy, gradients and (optionally)
    hessians for one graph input. Conventional ``Gradients`` payloads:

    - ``Array``: position gradients only (``∂E/∂r``).
    - ``PositionsAndCell``: position + cell gradients (forces + stress).
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
    P: NeighborListPoints,
    S: HasCell[AnyPeriodicity],
    Ptch: Patch[Any],
](
    model_fn: DirectMliapFn[Model, Gradients, Hessians, P, S, Ptch],
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    model_view: View[State, Model],
    *,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
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
    composer = FullSumComposer(
        GraphInputConstructor(
            GraphConstructor(
                particles=particles_view,
                systems=systems_view,
                neighborlist=neighborlist_view,
                probe=None,
            ),
            model_view,
        )
    )
    return DirectPotential(
        direct_potential_fn=model_fn,
        composer=composer,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
