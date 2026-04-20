# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MACE model integration via TorchModuleWrapper.

This module provides wrappers for using PyTorch MACE models directly from JAX
via the TorchModuleWrapper bridge, including full kUPS potential integration.

Example:
    ```python
    from kups.potential.mliap.torch import load_mace_wrapper

    # Load and wrap a MACE model (with forces)
    wrapper = load_mace_wrapper("path/to/mace.model")
    result = wrapper(node_attrs, positions, edge_index, batch, ptr, shifts, cell)
    energy, forces = result["energy"], result["forces"]
    ```

For kUPS integration:
    ```python
    from kups.potential.mliap.torch.mace import TorchMACEModel, make_torch_mace_potential

    # Create TorchMACEModel from wrapper
    model = TorchMACEModel(species_to_index, cutoff, num_species, wrapper)

    # Create kUPS potential (forces only)
    potential = make_torch_mace_potential(
        particles_view, systems_view, neighborlist_view, model, ...
    )

    # With virial/stress support
    potential = make_torch_mace_potential(
        ..., compute_virials=True, ...
    )
    ```

Requires the `torch_dev` dependency group: `uv sync --group torch_dev`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, overload

import jax
import jax.numpy as jnp
import torch  # pyright: ignore[reportMissingImports]
from jax import Array

from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import NearestNeighborList
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import EMPTY, EmptyType, Potential, PotentialOut
from kups.core.typing import (
    HasAtomicNumbers,
    HasUnitCell,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import TriclinicUnitCell
from kups.core.utils.functools import constant
from kups.core.utils.jax import dataclass, field
from kups.core.utils.torch import TorchModuleWrapper
from kups.potential.common.energy import PositionAndUnitCell
from kups.potential.common.graph import (
    GraphPotentialInput,
    HyperGraph,
    IsRadiusGraphPoints,
)
from kups.potential.mliap.interface import make_mliap_potential


class IsTorchMACEParticles(IsRadiusGraphPoints, HasAtomicNumbers, Protocol):
    """Particle protocol for PyTorch MACE models."""

    ...


__all__ = [
    "TorchMACEModel",
    "load_mace_wrapper",
    "make_torch_mace_potential",
]


class _PreparedMACEInputs(NamedTuple):
    """Prepared inputs for MACE model in PyG-style format.

    All tensors include padding (one extra atom/system) to work around
    MACE's inability to handle empty graphs.
    """

    node_attrs: Array
    positions: Array
    edge_index: Array
    batch: Array
    ptr: Array
    shifts: Array
    cell: Array | None


def _prepare_mace_inputs[
    P: IsTorchMACEParticles,
    S: HasUnitCell,
](
    graph: HyperGraph[P, S, Literal[2]],
    species_to_index: Array,
    num_mace_species: int,
) -> _PreparedMACEInputs:
    """Prepare MACE inputs from a sorted graph.

    Args:
        graph: HyperGraph with particles already sorted by system index.
        species_to_index: Mapping from atomic number to MACE species index.
        num_mace_species: Number of species the MACE model was trained on.

    Returns:
        Prepared inputs in PyG-style format for the MACE model.
    """
    ptr = jnp.cumulative_sum(
        graph.particles.data.system.counts.data, include_initial=True
    )
    species = jnp.pad(graph.particles.data.atomic_numbers, (0, 1), constant_values=0)
    positions = jnp.pad(
        graph.particles.data.positions,
        ((0, 1), (0, 0)),
        constant_values=0,
    )
    batch = jnp.pad(
        graph.particles.data.system.indices,
        (0, 1),
        constant_values=graph.particles.data.system.num_labels,
    )
    ptr = jnp.pad(ptr, (0, 1), constant_values=ptr[-1] + 1)
    edge_indices = graph.edges.indices.indices
    abs_shifts = (
        graph.edges.absolute_shifts(graph.particles, graph.systems)
        .squeeze(1)
        .astype(float)
    )
    node_attrs = jax.nn.one_hot(species_to_index[species], num_mace_species)

    return _PreparedMACEInputs(
        node_attrs=node_attrs,
        positions=positions,
        edge_index=edge_indices.T,
        batch=batch,
        ptr=ptr,
        shifts=abs_shifts,
        cell=None,
    )


class MACEModule(torch.nn.Module):
    """Wraps a MACE model for JAX interop via TorchModuleWrapper.

    Supports energy-only, energy+forces, and energy+forces+virials modes
    via the compute_force and compute_virials flags.
    """

    def __init__(
        self,
        mace_model: torch.nn.Module,
        compute_force: bool = True,
        compute_virials: bool = False,
    ) -> None:
        """Initialise MACEModule.

        Args:
            mace_model: Underlying PyTorch MACE model.
            compute_force: Whether to compute forces.
            compute_virials: Whether to compute virials.
        """
        super().__init__()
        self.mace = mace_model
        self.mace.eval()
        self.compute_force = compute_force
        self.compute_virials = compute_virials

    def forward(
        self,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        shifts: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run MACE forward pass.

        Args:
            node_attrs: One-hot species encoding.
            positions: Atom positions.
            edge_index: Edge index array (2, n_edges).
            batch: System index per atom.
            ptr: Cumulative atom counts per system.
            shifts: Absolute shift vectors per edge.
            cell: Unit cell lattice vectors (optional).

        Returns:
            Dictionary with ``"energy"`` and optionally ``"forces"`` /
            ``"virials"`` tensors.
        """
        input_dict = {
            "node_attrs": node_attrs,
            "positions": positions,
            "edge_index": edge_index,
            "batch": batch,
            "ptr": ptr,
            "shifts": shifts,
            "cell": cell,
        }
        output = self.mace(
            input_dict,
            compute_force=self.compute_force,
            compute_virials=self.compute_virials,
        )

        result: dict[str, torch.Tensor] = {"energy": output["energy"].detach()}
        if self.compute_force:
            result["forces"] = output["forces"].detach()
        if self.compute_virials:
            result["virials"] = output["virials"].detach()
        return result


def load_mace_wrapper(
    model_path: str | Path,
    device: str = "cuda",
    compute_force: bool = True,
    compute_virials: bool = False,
    dtype: Literal["float32", "float64"] = "float32",
) -> TorchModuleWrapper:
    """Load a PyTorch MACE model and wrap it for JAX computation.

    Args:
        model_path: Path to the MACE .model file
        device: Device to load the model onto (default: "cuda")
        compute_force: Whether to compute forces (default: True)
        compute_virials: Whether to compute virials for stress (default: False)
        dtype: Model precision - "float32" (default) or "float64"

    Returns:
        TorchModuleWrapper containing MACEModule.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Device 'cuda' requested but CUDA is not available. "
            "Use device='cpu' or ensure CUDA is properly installed."
        )

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"MACE model not found: {model_path}")

    model = torch.load(path, weights_only=False, map_location=device)
    model.eval()

    if dtype == "float64":
        model = model.double()
    else:
        model = model.float()

    module = MACEModule(
        model, compute_force=compute_force, compute_virials=compute_virials
    )
    return TorchModuleWrapper(module, requires_grad=compute_force)


@dataclass
class TorchMACEModel:
    """MACE model container for PyTorch models via TorchModuleWrapper.

    Attributes:
        species_to_index: Mapping from atomic number to MACE species index.
        cutoff: Model cutoff radius per system.
        num_mace_species: Number of species the MACE model was trained on.
        wrapper: TorchModuleWrapper bridging PyTorch to JAX.
        compute_virials: Whether to compute virials for stress.
    """

    species_to_index: Array
    cutoff: Table[SystemId, Array]
    num_mace_species: int = field(static=True)
    wrapper: TorchModuleWrapper = field(static=True)
    compute_virials: bool = field(static=True, default=False)


type TorchMACEInput[
    P: IsTorchMACEParticles,
    S: HasUnitCell,
] = GraphPotentialInput[TorchMACEModel, P, S, Literal[2]]


class _MACEWrapperResult(NamedTuple):
    """Result from calling MACE wrapper."""

    energy: Array
    forces: Array
    virials: Array | None


def _call_mace_wrapper[
    P: IsTorchMACEParticles,
    S: HasUnitCell,
](inp: TorchMACEInput[P, S]) -> _MACEWrapperResult:
    """Call the MACE wrapper and extract energy, forces, and optional virials.

    Args:
        inp: Graph potential input containing the MACE model and graph.

    Returns:
        Energy, forces (unsorted back to original order), and optional virials.
    """
    graph, sort_order = inp.graph.sorted_by_system(return_sort_order=True)
    unsort_order = jnp.argsort(sort_order)
    mace = inp.parameters

    inputs = _prepare_mace_inputs(graph, mace.species_to_index, mace.num_mace_species)

    cell = None
    if graph.systems is not None:
        cell = graph.systems.data.unitcell.lattice_vectors

    result = mace.wrapper(
        inputs.node_attrs,
        inputs.positions,
        inputs.edge_index,
        inputs.batch,
        inputs.ptr,
        inputs.shifts,
        cell,
    )

    energy = result["energy"][:-1]
    forces_sorted = result["forces"][:-1]
    forces = forces_sorted[unsort_order]
    virials = result.get("virials")
    if virials is not None:
        virials = virials[:-1]

    return _MACEWrapperResult(energy, forces, virials)


@overload
def torch_mace_model_fn[
    P: IsTorchMACEParticles,
    S: HasUnitCell,
](
    inp: TorchMACEInput[P, S],
    *,
    compute_virials: Literal[False] = False,
) -> WithPatch[PotentialOut[Array, EmptyType], IdPatch]: ...


@overload
def torch_mace_model_fn[
    P: IsTorchMACEParticles,
    S: HasUnitCell,
](
    inp: TorchMACEInput[P, S],
    *,
    compute_virials: Literal[True],
) -> WithPatch[PotentialOut[PositionAndUnitCell, EmptyType], IdPatch]: ...


def torch_mace_model_fn[
    P: IsTorchMACEParticles,
    S: HasUnitCell,
](
    inp: TorchMACEInput[P, S],
    *,
    compute_virials: bool = False,
) -> (
    WithPatch[PotentialOut[Array, EmptyType], IdPatch]
    | WithPatch[PotentialOut[PositionAndUnitCell, EmptyType], IdPatch]
):
    """Model function for PyTorch MACE models.

    Returns ``PotentialOut`` with forces (and optionally virials) computed by
    PyTorch.

    Args:
        inp: Graph potential input containing the MACE model and graph.
        compute_virials: Whether to include virial gradients in the output.

    Returns:
        ``WithPatch`` containing ``PotentialOut`` with energy, gradients, and
        an identity patch.
    """
    result = _call_mace_wrapper(inp)

    if compute_virials:
        assert result.virials is not None, "Model must have compute_virials=True"
        gradients = PositionAndUnitCell(
            positions=Table(inp.graph.particles.keys, -result.forces),
            unitcell=Table(
                inp.graph.systems.keys,
                TriclinicUnitCell.from_matrix(result.virials),
            ),
        )
        return WithPatch(
            PotentialOut(Table.arange(result.energy, label=SystemId), gradients, EMPTY),
            IdPatch(),
        )
    else:
        return WithPatch(
            PotentialOut(
                Table.arange(result.energy, label=SystemId), -result.forces, EMPTY
            ),
            IdPatch(),
        )


@overload
def make_torch_mace_potential[
    State,
    P: IsTorchMACEParticles,
    S: HasUnitCell,
    NNList: NearestNeighborList,
](
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NNList],
    model: View[State, TorchMACEModel] | TorchMACEModel,
    cutoffs_view: View[State, Table[SystemId, Array]],
    compute_virials: Literal[False] = False,
    patch_idx_view: View[State, PotentialOut[Array, EmptyType]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Array, EmptyType]] | None = None,
) -> Potential[State, Array, EmptyType, Patch[State]]: ...


@overload
def make_torch_mace_potential[
    State,
    P: IsTorchMACEParticles,
    S: HasUnitCell,
    NNList: NearestNeighborList,
](
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NNList],
    model: View[State, TorchMACEModel] | TorchMACEModel,
    cutoffs_view: View[State, Table[SystemId, Array]],
    compute_virials: Literal[True],
    patch_idx_view: View[State, PotentialOut[PositionAndUnitCell, EmptyType]]
    | None = None,
    out_cache_lens: Lens[State, PotentialOut[PositionAndUnitCell, EmptyType]]
    | None = None,
) -> Potential[State, PositionAndUnitCell, EmptyType, Patch[State]]: ...


def make_torch_mace_potential[
    State,
    P: IsTorchMACEParticles,
    S: HasUnitCell,
    NNList: NearestNeighborList,
](
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NNList],
    model: View[State, TorchMACEModel] | TorchMACEModel,
    cutoffs_view: View[State, Table[SystemId, Array]],
    compute_virials: bool = False,
    patch_idx_view: Any | None = None,
    out_cache_lens: Any | None = None,
) -> Any:
    """Create kUPS potential from PyTorch MACE model.

    Forces are computed by PyTorch natively (not JAX autodiff).
    Hessians are NOT supported (returns EmptyType).

    Args:
        particles_view: Extracts particle data from state
        systems_view: Extracts system data (unit cell) from state
        neighborlist_view: Extracts neighbor list from state
        model: TorchMACEModel instance or view to model in state
        cutoffs_view: Extracts cutoffs as ``Indexed[SystemId, Array]``
        compute_virials: Whether to compute virials for stress (default: False)
        patch_idx_view: Cached output index structure (optional)
        out_cache_lens: Cache location lens (optional)

    Returns:
        kUPS ``Potential`` backed by the PyTorch MACE model.
    """
    if isinstance(model, TorchMACEModel):
        model_view: View[State, TorchMACEModel] = constant(model)
    else:
        model_view = model

    if compute_virials:

        def virial_fn(inp):
            return torch_mace_model_fn(inp, compute_virials=True)

        return make_mliap_potential(
            model_fn=virial_fn,
            particles_view=particles_view,
            systems_view=systems_view,
            neighborlist_view=neighborlist_view,
            model_view=model_view,
            cutoffs_view=cutoffs_view,
            patch_idx_view=patch_idx_view,
            out_cache_lens=out_cache_lens,
        )

    def forces_fn(inp):
        return torch_mace_model_fn(inp, compute_virials=False)

    return make_mliap_potential(
        model_fn=forces_fn,
        particles_view=particles_view,
        systems_view=systems_view,
        neighborlist_view=neighborlist_view,
        model_view=model_view,
        cutoffs_view=cutoffs_view,
        patch_idx_view=patch_idx_view,
        out_cache_lens=out_cache_lens,
    )
