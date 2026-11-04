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
from typing import Any, Literal, Protocol, TypedDict, get_args

import jax
import jax.numpy as jnp
from jax import Array, export

from kups.core.assertion import FIRST_FAILURE, runtime_assert
from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import (
    NeighborList,
    NeighborListPoints,
)
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import Energy, PotentialOut
from kups.core.typing import HasAtomicNumbers, HasCell, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, sequential_vmap_with_vjp
from kups.core.utils.kahan import KahanSummand
from kups.core.utils.msgpack import deserialize as msgpack_deserialize
from kups.potential.common.energy import FullSumComposer, PotentialFromEnergy
from kups.potential.common.graph import (
    GraphConstructor,
    GraphInputConstructor,
    GraphPotentialInput,
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


class IsTojaxedParticles(NeighborListPoints, HasAtomicNumbers, Protocol): ...


# Same literal as ``kups.potential.mliap.torch.uma.UMATaskName``. This copy
# needs no torch; the simulation configs import it from here.
type UMATaskName = Literal["omat", "omol", "oc20", "odac", "omc"]
_UMA_TASK_NAMES: tuple[str, ...] = get_args(UMATaskName.__value__)


def _check_task_name(task_name: str | None) -> None:
    if task_name is not None and task_name not in _UMA_TASK_NAMES:
        raise ValueError(
            f"Unknown UMA task {task_name!r}; expected None or one of {_UMA_TASK_NAMES}."
        )


@dataclass
class TojaxedMliap:
    """Jaxified model container.

    Attributes:
        cutoff: Model cutoff radius [Angstrom].
        params: Model parameters as a list of arrays.
        model: Exported JAX model.
        task_name: UMA task the archive was exported for (tojax's
            ``--dataset``), or ``None`` for other models. kUPS cannot read it
            from the archive and cannot detect a wrong declaration. Only
            ``"omol"`` changes the inputs; see
            [tojaxed_energy][kups.potential.mliap.tojax.tojaxed_energy].
    """

    cutoff: Table[SystemId, Array]
    params: list[Array]
    model: export.Exported = field(static=True)
    task_name: UMATaskName | None = field(static=True, default=None)

    @staticmethod
    def from_zip_file(
        zip_file: str | Path, *, task_name: UMATaskName | None = None
    ) -> "TojaxedMliap":
        """Load a jaxified model from a zip archive.

        Expects the archive to contain ``model.jax``, ``metadata.json``
        (with a ``cutoff`` key), and ``params.msgpack``.

        Args:
            zip_file: Path to the ``.zip`` archive.
            task_name: UMA task the archive was exported for, or ``None``.

        Returns:
            Loaded jaxified model.

        Raises:
            ValueError: If ``task_name`` is not a UMA task.
        """
        _check_task_name(task_name)
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
            task_name=task_name,
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


def _omol_field(systems: Any, name: str, default: int, n_real: int) -> Array:
    value = getattr(systems, name, None)
    if value is None:
        value = jnp.full((n_real,), default, dtype=int)
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.signedinteger):
        raise TypeError(f"{name} must have a signed integer dtype; got {value.dtype}.")
    if value.shape != (n_real,):
        raise ValueError(f"{name} must have shape ({n_real},); got {value.shape}.")
    return value


def _omol_charge_spin(systems: Any, n_real: int) -> tuple[Array, Array]:
    """Charge and multiplicity per system, plus 0 and 1 for the padding system."""
    charge = _omol_field(systems, "total_charge", 0, n_real)
    spin = _omol_field(systems, "spin_multiplicity", 1, n_real)
    # One assertion for both fields: Result cannot combine a vmapped
    # predicate with an unbatched one.
    runtime_assert(
        jnp.all((charge >= -100) & (charge <= 100) & (spin >= 1) & (spin <= 100)),
        "omol needs total_charge in [-100, 100] and spin_multiplicity in [1, 100];"
        " got charge {charge_min} to {charge_max}, multiplicity {spin_min} to {spin_max}.",
        fmt_args={
            "charge_min": jnp.min(charge),
            "charge_max": jnp.max(charge),
            "spin_min": jnp.min(spin),
            "spin_max": jnp.max(spin),
        },
        exception_type=ValueError,
        loop_merge=FIRST_FAILURE,
    )
    return jnp.append(charge, 0), jnp.append(spin, 1)


def tojaxed_energy(
    inp: JaxifiedInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Compute energy using a jaxified model.

    Prepares graph data and calls the exported model. The model receives one
    padding system after the real ones.

    For ``task_name="omol"``, each system's ``total_charge`` and
    ``spin_multiplicity`` are passed to the model. A field that is missing or
    ``None`` defaults to charge 0 and multiplicity 1; the padding system gets
    the same defaults. The values stay traced, so changing them does not
    recompile. Every other ``task_name`` passes zeros.

    Charge must lie in ``[-100, 100]`` and multiplicity in ``[1, 100]``.
    Because the values are traced, this is a runtime assertion: the call
    still runs, and the ``ValueError`` is raised when the returned ``Result``
    is checked on the host, as ``propagate_and_fix`` does.

    Args:
        inp: Graph potential input containing the jaxified model and graph data.

    Returns:
        Per-system energies.

    Raises:
        ValueError: If ``task_name`` is not a UMA task, or for ``"omol"`` if a
            metadata field is not of shape ``(n_systems,)``.
        TypeError: For ``"omol"``, if a metadata field is not a signed
            integer array.
    """
    _check_task_name(inp.parameters.task_name)
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
        graph.particles.data.system.raw_indices,
        (0, 1),
        constant_values=graph.particles.data.system.num_labels,
    )
    cell = graph.systems.data.cell.vectors
    cell = jnp.concatenate([cell, jnp.zeros((1, 3, 3))], axis=0)

    edge_indices = graph.edges.indices.indices_in(graph.particles.keys)
    if inp.parameters.task_name == "omol":
        charge, spin = _omol_charge_spin(graph.systems.data, len(graph.systems))
    else:
        charge, spin = jnp.zeros(n_sys), jnp.zeros(n_sys)

    input_dict = AtomGraphInput(
        pos=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=jnp.ones((n_sys, 3), dtype=bool),
        edge_index=edge_indices.T,
        cell_offsets=graph.edges.shifts.squeeze(1),
        batch=batch,
        charge=charge,
        spin=spin,
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
    composer = FullSumComposer(GraphInputConstructor(radius_graph_fn, model_view))
    return PotentialFromEnergy(
        composer=composer,
        energy_fn=tojaxed_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
