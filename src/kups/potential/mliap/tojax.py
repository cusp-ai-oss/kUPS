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
from typing import Any, Literal, Protocol, TypedDict

import jax
import jax.numpy as jnp
from jax import Array, export

from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import (
    NeighborList,
)
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import Energy, PotentialOut
from kups.core.typing import HasAtomicNumbers, HasCell, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, sequential_vmap_with_vjp
from kups.core.utils.kahan import KahanSummand
from kups.core.utils.msgpack import deserialize as msgpack_deserialize
from kups.potential.common.energy import PotentialFromEnergy
from kups.potential.common.graph import (
    FullGraphSumComposer,
    GraphConstructor,
    GraphPotentialInput,
    IsRadiusGraphPoints,
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
        kwargs: dict[str, Any] = {}
        leafes = self.model.in_tree.flatten_up_to((args, kwargs))
        leafes = jax.tree.map(
            jax.lax.convert_element_type,
            leafes,
            jax.tree.map(lambda x: x.dtype, list(self.model.in_avals)),
        )
        args, kwargs = self.model.in_tree.unflatten(leafes)
        return self.model.call(*args, **kwargs)


type JaxifiedInput = GraphPotentialInput[
    TojaxedMliap, IsTojaxedParticles, HasCell[AnyPeriodicity], Literal[2]
]


def tojaxed_energy(
    inp: JaxifiedInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Compute energy using a jaxified model.

    Prepares graph data and calls the exported model.

    Args:
        inp: Graph potential input containing the jaxified model and graph data.

    Returns:
        Per-system energies.
    """
    graph = inp.graph.sorted_by_system(sort_edges=True)

    n_sys = graph.systems.data.cell.vectors.shape[0] + 1

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
    cell = graph.systems.data.cell.vectors
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
    return WithPatch(
        graph.systems.set_data(energy[:-1]), IdPatch[Any]()
    )  # Remove padding


def make_tojaxed_potential[State, Gradients, Hessians](
    particles_view: View[State, Table[ParticleId, IsTojaxedParticles]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    model: View[State, TojaxedMliap] | TojaxedMliap,
    gradient_lens: Lens[JaxifiedInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> PotentialFromEnergy[State, JaxifiedInput, Gradients, Hessians, Patch[Any]]:
    """Create a jaxified machine learning potential.

    Args:
        particles_view: Extracts particle data (positions, species).
        systems_view: Extracts system data (cell).
        neighborlist_view: Extracts a cutoff-bound neighbor list.
        model: Jaxified model instance or view to model in state.
        gradient_lens: Lens specifying which gradients to compute.
        hessian_lens: Lens specifying which Hessians to compute.
        hessian_idx_view: View to hessian index structure.
        patch_idx_view: View to cached output index structure.
        out_cache_lens: Lens to cache location.

    Returns:
        Jaxified potential.
    """
    model_view = (lambda _: model) if isinstance(model, TojaxedMliap) else model
    radius_graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
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
