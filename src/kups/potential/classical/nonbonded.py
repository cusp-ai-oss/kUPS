# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Fused real-space nonbonded potential: Lennard-Jones + short-range Ewald.

Combines the two real-space potentials kups builds separately —
``make_lennard_jones_from_state`` and the short-range term of ``make_ewald_from_state`` —
into a single ``PotentialFromEnergy`` that builds one radius graph (one neighbor-list
query, one edge gather) and runs one ``jax.vjp`` over the summed energy. Forces (``dE/dr``)
and NPT stress (``dE/dcell``) come from autodiff through the shared ``PositionAndCell``
gradient lens.

Two energy bodies are provided:

  * ``nonbonded_energy`` calls the existing ``lennard_jones_energy`` and
    ``ewald_short_range_energy`` on the shared graph and tree-adds the per-system results.
    Numerics are identical to the two potentials summed; the win is the shared graph and
    the single backward pass.
  * ``fused_nonbonded_edge_energy`` does one edge gather, one ``r²``, and both LJ and
    screened-Coulomb in the same body, saving the second gather and distance inside the
    forward pass. It duplicates the per-edge formulas from the two source functions and
    must be kept in sync with them.

Full-recompute only (no Monte Carlo incremental/probe path).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp

from kups.core.data import Table
from kups.core.lens import Lens, SimpleLens, View
from kups.core.neighborlist import NearestNeighborList
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import EMPTY_LENS, Energy, Potential
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field
from kups.potential.classical.ewald import (
    TO_STANDARD_UNITS,
    EwaldParameters,
    IsEwaldPointData,
    ewald_short_range_energy,
)
from kups.potential.classical.lennard_jones import (
    IsLJGraphParticles,
    LennardJonesParameters,
    lennard_jones_energy,
)
from kups.potential.common.energy import (
    PositionAndCell,
    PotentialFromEnergy,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import (
    GraphPotentialInput,
    LocalGraphSumComposer,
    RadiusGraphConstructor,
)


@runtime_checkable
class IsNonbondedPoints(IsLJGraphParticles, IsEwaldPointData, Protocol):
    """Particle data carrying everything both terms need: positions, system index,
    inclusion/exclusion (the radius graph), species labels (LJ), and charges (Ewald)."""


@dataclass
class NonbondedParameters:
    """Bundle of the two existing parameter objects.

    Attributes:
        lj: Lennard-Jones parameters (species sigma/epsilon matrices, cutoff).
        ewald: Ewald parameters (alpha, real-space cutoff, k-vectors). Only ``alpha`` and
            ``cutoff`` are used by the real-space term; ``reciprocal_lattice_shifts`` is
            carried so the same object also feeds the long-range/self potentials.
    """

    lj: LennardJonesParameters
    ewald: EwaldParameters


type NonbondedInput = GraphPotentialInput[
    NonbondedParameters, IsNonbondedPoints, Any, Literal[2]
]


def nonbonded_energy(
    inp: NonbondedInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    """LJ + real-space Ewald on one shared graph, summed per system.

    Reuses the existing energy functions verbatim by re-bundling the shared graph with each
    parameter set, so numerics match ``lennard_jones_energy + ewald_short_range_energy``.
    The win is structural: a single ``PotentialFromEnergy`` over one graph build and one
    ``jax.vjp``.
    """
    graph = inp.graph
    lj_e = lennard_jones_energy(
        cast(Any, GraphPotentialInput(inp.parameters.lj, graph))
    )
    sr_e = ewald_short_range_energy(
        cast(Any, GraphPotentialInput(inp.parameters.ewald, graph))
    )
    total = jax.tree.map(jnp.add, lj_e.data, sr_e.data)
    return WithPatch(total, IdPatch())


def fused_nonbonded_edge_energy(
    inp: NonbondedInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    """Single-pass LJ + screened-Coulomb: one edge gather, one ``r²``, both terms.

    Duplicates the per-edge formulas from ``lennard_jones_edge_energy`` and
    ``ewald_short_range_energy`` (keep in sync); equivalent numerics. Saves the second edge
    gather and distance computation inside the forward pass.
    """
    graph = inp.graph
    lj, ew = inp.parameters.lj, inp.parameters.ewald

    edg = graph.particles[graph.edges.indices]
    r2 = jnp.sum(graph.edge_shifts[:, 0] ** 2, axis=-1)
    r = jnp.sqrt(r2)
    batch = graph.edge_batch_mask

    species = edg.labels.indices_in(lj.labels)
    eps = lj.epsilon[species[:, 0], species[:, 1]]
    sig = lj.sigma[species[:, 0], species[:, 1]]
    c6 = (sig**2 / r2) ** 3
    e_lj = 4 * eps * (c6**2 - c6)
    e_lj *= r2 < jnp.pow(lj.cutoff.data, 2)[batch.indices]

    qij = edg.charges[:, 0] * edg.charges[:, 1]
    erfc = jax.scipy.special.erfc(ew.alpha[batch] * r)
    e_c = qij * erfc / r * TO_STANDARD_UNITS
    e_c *= r < ew.cutoff[batch]

    total = batch.sum_over(e_lj + e_c) / 2
    return WithPatch(total, IdPatch())


class IsNonbondedState(Protocol):
    """State providing both potentials' inputs (intersection of IsLJState and IsEwaldState)."""

    @property
    def particles(self) -> Table[ParticleId, IsNonbondedPoints]: ...
    @property
    def systems(self) -> Table[SystemId, Any]: ...
    @property
    def neighborlist(self) -> NearestNeighborList: ...
    @property
    def lj_parameters(self) -> LennardJonesParameters: ...
    @property
    def ewald_parameters(self) -> EwaldParameters: ...


@dataclass
class _NonbondedParamView:
    """View bundling the two parameter views into one ``NonbondedParameters``.

    A dataclass (not a bare lambda) so it is a hashable/comparable static pytree field,
    matching how the composers hold their views.
    """

    lj: View[Any, LennardJonesParameters] = field(static=True)
    ewald: View[Any, EwaldParameters] = field(static=True)

    def __call__(self, state: Any) -> NonbondedParameters:
        return NonbondedParameters(lj=self.lj(state), ewald=self.ewald(state))


@dataclass
class _MaxCutoffView:
    """Per-system ``max(lj.cutoff, ewald.cutoff)`` so the one shared graph covers both terms.

    Uses the Ewald cutoff's per-system keys (the LJ cutoff table is length-1 and
    broadcasts). Each term still masks edges by its own cutoff inside the energy fn, so an
    over-wide graph only costs a few extra masked candidate edges.
    """

    lj: View[Any, LennardJonesParameters] = field(static=True)
    ewald: View[Any, EwaldParameters] = field(static=True)

    def __call__(self, state: Any) -> Table[SystemId, Any]:
        a = self.lj(state).cutoff
        b = self.ewald(state).cutoff
        return Table(
            b.keys, jnp.broadcast_to(jnp.maximum(a.data, b.data), b.data.shape)
        )


def make_nonbonded_from_state(
    state: Lens[Any, IsNonbondedState],
    *,
    compute_position_and_cell_gradients: bool = False,
    fused_edge: bool = False,
    forces_only: bool = False,
) -> Potential[Any, PositionAndCell, Any, Patch]:
    """Fused LJ + real-space-Ewald potential from a typed state (full-recompute only).

    Drop-in for ``sum_potentials(make_lennard_jones_from_state(...),
    make_ewald_from_state(...).short_range)``: build this for the real-space pair, then sum
    with the Ewald long-range / self (/ exclusion) potentials.

    Args:
        state: Lens to the sub-state with particles, systems, neighborlist,
            ``lj_parameters`` and ``ewald_parameters``.
        compute_position_and_cell_gradients: forces (dE/dr) + NPT stress (dE/dcell) via autodiff.
        fused_edge: use the single-pass edge body; else the maximal-reuse body.
        forces_only: differentiate w.r.t. positions only (forces, no cell-virial), for
            NVE/NVT where stress is unused. The gradient is a ``Table[ParticleId, Array]``
            (not ``PositionAndCell``); sum only with other forces-only potentials.
    """
    lj_view: Any = state.focus(lambda s: s.lj_parameters)
    ew_view: Any = state.focus(lambda s: s.ewald_parameters)

    graph_fn = RadiusGraphConstructor(
        particles=state.focus(lambda s: s.particles),
        systems=state.focus(lambda s: s.systems),
        cutoffs=cast(Any, _MaxCutoffView(lj_view, ew_view)),
        neighborlist=state.focus(lambda s: s.neighborlist),
        probe=None,
    )
    composer = LocalGraphSumComposer(
        graph_constructor=graph_fn,
        parameter_view=cast(Any, _NonbondedParamView(lj_view, ew_view)),
    )

    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if forces_only:
        gradient_lens = SimpleLens[GraphPotentialInput, Any](
            lambda x: x.graph.particles.map_data(lambda p: p.positions)
        )
    elif compute_position_and_cell_gradients:
        gradient_lens = SimpleLens[GraphPotentialInput, PositionAndCell](
            lambda x: PositionAndCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.cell),
            )
        )
        patch_idx_view = position_and_cell_idx_view

    return cast(
        Any,
        PotentialFromEnergy(
            energy_fn=fused_nonbonded_edge_energy if fused_edge else nonbonded_energy,
            composer=composer,
            gradient_lens=gradient_lens,
            hessian_lens=EMPTY_LENS,
            hessian_idx_view=EMPTY_LENS,
            cache_lens=None,
            patch_idx_view=patch_idx_view,
        ),
    )
