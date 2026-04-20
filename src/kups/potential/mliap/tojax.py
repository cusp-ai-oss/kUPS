# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Jaxified machine learning interatomic potential interface.

This module provides integration with generic JAX-exported MLFF models
via the ``AtomGraphInput`` / ``EnergyFn`` protocol.  Supports periodic
systems with graph-based atomic representations.
"""

import json
import zipfile
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, overload

import jax
import jax.numpy as jnp
from jax import Array, export

from kups.core.data import Table
from kups.core.lens import Lens, SimpleLens, View
from kups.core.neighborlist import NearestNeighborList
from kups.core.patch import IdPatch, WithPatch
from kups.core.potential import EMPTY_LENS, EmptyType, Energy, Potential, PotentialOut
from kups.core.typing import HasAtomicNumbers, HasUnitCell, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, sequential_vmap_with_vjp
from kups.core.utils.msgpack import deserialize as msgpack_deserialize
from kups.potential.common.energy import PositionAndUnitCell, PotentialFromEnergy
from kups.potential.common.graph import (
    FullGraphSumComposer,
    GraphPotentialInput,
    IsRadiusGraphPoints,
    RadiusGraphConstructor,
)


class AtomGraphInput(TypedDict):
    """Typed dictionary for jaxified model graph input."""

    pos: Array  # (N, 3)
    atomic_numbers: Array  # (N,)
    cell: Array  # (B, 3, 3)
    pbc: Array  # (B, 3)
    edge_index: Array  # (2, E)
    cell_offsets: Array  # (E, 3)
    batch: Array  # (N,)
    charge: Array  # (B,)
    spin: Array  # (B,)


class EnergyFn(Protocol):
    """Protocol for a jaxified energy function."""

    def call(self, params: list[Array], data: AtomGraphInput) -> Array: ...


class IsTojaxedParticles(IsRadiusGraphPoints, HasAtomicNumbers, Protocol): ...


@dataclass
class TojaxedMliap:
    """Jaxified model container.

    Attributes:
        cutoff: Model cutoff radius [Angstrom].
        params: Model parameters as a list of arrays.
        model: Exported JAX model.
    """

    cutoff: Table[SystemId, Array]
    params: list[Array]
    model: export.Exported = field(static=True)

    @staticmethod
    def from_zip_file(zip_file: str | Path) -> "TojaxedMliap":
        """Load a jaxified model from a zip archive.

        Expects the archive to contain ``model.jax``, ``metadata.json``
        (with a ``cutoff`` key), and ``params.msgpack``.

        Args:
            zip_file: Path to the ``.zip`` archive.

        Returns:
            Loaded jaxified model.
        """
        with zipfile.ZipFile(zip_file, "r") as zf:
            with zf.open("model.jax") as f:
                model = export.deserialize(f.read())  # type: ignore
            with zf.open("metadata.json") as f:
                cutoff = json.loads(f.read().decode())["cutoff"]
            with zf.open("params.msgpack") as f:
                params = list(msgpack_deserialize(f.read()))
        return TojaxedMliap(
            cutoff=Table((SystemId(0),), jnp.array([cutoff], float)),
            params=params,
            model=model,
        )

    def call(self, input: AtomGraphInput) -> Array:
        """Call the jaxified model on the given input."""
        args = (self.params, input)
        kwargs = {}
        leafes = self.model.in_tree.flatten_up_to((args, kwargs))
        leafes = jax.tree.map(
            jax.lax.convert_element_type,
            leafes,
            jax.tree.map(lambda x: x.dtype, list(self.model.in_avals)),
        )
        args, kwargs = self.model.in_tree.unflatten(leafes)
        return self.model.call(*args, **kwargs)


type JaxifiedInput = GraphPotentialInput[
    TojaxedMliap, IsTojaxedParticles, HasUnitCell, Literal[2]
]


def tojaxed_energy(
    inp: JaxifiedInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    """Compute energy using a jaxified model.

    Prepares graph data and calls the exported model.

    Args:
        inp: Graph potential input containing the jaxified model and graph data.

    Returns:
        Per-system energies.
    """
    graph = inp.graph.sorted_by_system(sort_edges=True)

    n_sys = graph.systems.data.unitcell.lattice_vectors.shape[0] + 1

    positions = jnp.pad(
        graph.particles.data.positions,
        ((0, 1), (0, 0)),
        constant_values=0,
    )
    atomic_numbers = jnp.pad(
        graph.particles.data.atomic_numbers,
        (0, 1),
        constant_values=0,
    )
    batch = jnp.pad(
        graph.particles.data.system.indices,
        (0, 1),
        constant_values=graph.particles.data.system.num_labels,
    )
    cell = graph.systems.data.unitcell.lattice_vectors
    cell = jnp.concatenate([cell, jnp.zeros((1, 3, 3))], axis=0)

    edge_indices = graph.edges.indices.indices_in(graph.particles.keys)

    input_dict = AtomGraphInput(
        pos=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=jnp.ones((n_sys, 3), dtype=bool),
        edge_index=edge_indices.T,
        cell_offsets=graph.edges.shifts.squeeze(1),
        batch=batch,
        charge=jnp.zeros(n_sys),
        spin=jnp.zeros(n_sys),
    )
    energy = sequential_vmap_with_vjp(inp.parameters.call)(input_dict)
    return WithPatch(graph.systems.set_data(energy[:-1]), IdPatch())  # Remove padding


def make_tojaxed_potential[State, Gradients, Hessians](
    particles_view: View[State, Table[ParticleId, IsTojaxedParticles]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    neighborlist_view: View[State, NearestNeighborList],
    model: View[State, TojaxedMliap] | TojaxedMliap,
    cutoffs_view: View[State, Table[SystemId, Array]],
    gradient_lens: Lens[JaxifiedInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> PotentialFromEnergy[State, JaxifiedInput, Gradients, Hessians, Any]:
    """Create a jaxified machine learning potential.

    Args:
        particles_view: Extracts particle data (positions, species).
        systems_view: Extracts system data (unit cell).
        neighborlist_view: Extracts neighbor list.
        model: Jaxified model instance or view to model in state.
        cutoffs_view: Extracts cutoffs as ``Table[SystemId, Array]``.
        gradient_lens: Lens specifying which gradients to compute.
        hessian_lens: Lens specifying which Hessians to compute.
        hessian_idx_view: View to hessian index structure.
        patch_idx_view: View to cached output index structure.
        out_cache_lens: Lens to cache location.

    Returns:
        Jaxified potential.
    """
    model_view = (lambda _: model) if isinstance(model, TojaxedMliap) else model
    radius_graph_fn = RadiusGraphConstructor(
        particles=particles_view,
        systems=systems_view,
        cutoffs=cutoffs_view,
        neighborlist=neighborlist_view,
        probe=None,
    )
    composer = FullGraphSumComposer(radius_graph_fn, model_view)
    return PotentialFromEnergy(
        composer=composer,
        energy_fn=tojaxed_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


class IsTojaxedState(Protocol):
    """Protocol for states providing all inputs for the jaxified potential."""

    @property
    def particles(self) -> Table[ParticleId, IsTojaxedParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...
    @property
    def neighborlist(self) -> NearestNeighborList: ...
    @property
    def jaxified_model(self) -> TojaxedMliap: ...


@overload
def make_tojaxed_from_state[State, InState: IsTojaxedState](
    state: Lens[State, InState],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Any]: ...


@overload
def make_tojaxed_from_state[State, InState: IsTojaxedState](
    state: Lens[State, InState],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, Any]: ...


def make_tojaxed_from_state(
    state: Any,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create a jaxified potential from a typed state.

    Args:
        state: Lens into the sub-state providing particles, unit cell,
            neighbor list, and jaxified model.
        compute_position_and_unitcell_gradients: When ``True``, compute
            gradients w.r.t. particle positions and unit cell
            (for forces / stress).

    Returns:
        Configured jaxified [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[JaxifiedInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.unitcell),
            )
        )
    return make_tojaxed_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        state.focus(lambda x: x.neighborlist),
        state.focus(lambda x: x.jaxified_model),
        state.focus(lambda x: x.jaxified_model.cutoff),
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
    )
