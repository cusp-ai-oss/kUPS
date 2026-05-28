# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Universal PyTorch MLFF interface.

Mirrors the JAX [tojax][kups.potential.mliap.tojax] interface for PyTorch
models. Each MLFF backend only needs to provide a ``torch.nn.Module`` whose
forward consumes the universal [AtomGraphInput][kups.potential.mliap.torch.interface.AtomGraphInput]
and returns a dict with ``"energy"``, ``"position_gradients"``, and
optionally ``"cell_gradients"``. All graph extraction, padding, and kUPS
``Potential`` wiring is handled here.

Example:
    ```python
    from kups.potential.mliap.torch.interface import (
        TorchMliap, make_torch_mliap_from_state,
    )

    # A backend provides a Module with the universal forward contract:
    model = TorchMliap.from_module(my_module, cutoff=6.0, compute_cell_gradients=True)

    # Wire into a kUPS Potential:
    potential = make_torch_mliap_from_state(
        state_lens, compute_position_and_cell_gradients=True,
    )
    ```

Requires the ``torch_dev`` dependency group: ``uv sync --group torch_dev``.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, overload

import jax.numpy as jnp
import torch  # pyright: ignore[reportMissingImports]
from jax import Array

from kups.core.data import Table
from kups.core.lens import Lens, View, bind
from kups.core.neighborlist import (
    IsAdaptiveCutoffNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
    adaptive_cutoff_neighborlist_from_state,
)
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import EMPTY, EmptyType, Potential, PotentialOut
from kups.core.typing import (
    HasAtomicNumbers,
    HasCell,
    ParticleId,
    SystemId,
)
from kups.core.utils.functools import constant
from kups.core.utils.jax import dataclass, field
from kups.core.utils.torch import TorchModuleWrapper
from kups.potential.common.energy import PositionAndCell
from kups.potential.common.graph import (
    GraphPotentialInput,
    IsRadiusGraphPoints,
)
from kups.potential.mliap.direct import make_direct_mliap_potential

__all__ = [
    "AtomGraphInput",
    "IsTorchMliapParticles",
    "IsTorchMliapState",
    "TorchMliap",
    "TorchMliapForward",
    "lattice_gradient_from_virial",
    "make_torch_mliap_from_state",
    "make_torch_mliap_potential",
    "torch_mliap_model_fn",
]


def lattice_gradient_from_virial(
    forces: "torch.Tensor",
    positions: "torch.Tensor",
    batch: "torch.Tensor",
    cell: "torch.Tensor",
    virial: "torch.Tensor",
) -> "torch.Tensor":
    """Recover ``∂E/∂h`` from a symmetric-strain virial.

    Many torch MLFF backends (MACE, UMA, …) return a virial or stress quantity
    that encodes the gradient of energy under a *symmetric infinitesimal
    strain* applied jointly to positions and cell:

        r_b → r_b + r_b @ ε          (per atom b)
        h_s → h_s + h_s @ ε          (per system s)

    The virial returned by the backend is then
    ``virial = sym(pos_virial + cell_virial)`` where:

        pos_virial[s, j, k]  = Σ_{b∈s} (∂E/∂r_b)_j · (r_b)_k
        cell_virial          = cell^T @ ∂E/∂h
        sym(M)               = (M + M^T) / 2

    Given forces (= ``-∂E/∂r``), positions, batch, cell, and the virial, this
    function reconstructs ``pos_virial`` from ``-forces ⊗ positions``, subtracts
    it, and solves ``cell^T @ (∂E/∂h) = cell_virial`` for the raw lattice
    gradient. Assumes ``cell^T @ ∂E/∂h`` is symmetric (rotational invariance
    of the energy); the antisymmetric part is unrecoverable from the
    symmetric-strain virial alone.

    Args:
        forces: ``(N, 3)`` ``= -∂E/∂r``.
        positions: ``(N, 3)``.
        batch: ``(N,)`` int system index per atom.
        cell: ``(B, 3, 3)``.
        virial: ``(B, 3, 3)`` symmetric strain virial as defined above.

    Returns:
        ``(B, 3, 3)`` ``∂E/∂h`` at fixed positions.
    """
    # Backends may emit ``forces``/``virial`` at a different precision than
    # ``cell``/``positions`` (e.g. UMA's predict-unit casts to its inference
    # dtype but normalizers/denorm steps can bump back). Unify on the highest
    # precision present so ``torch.linalg.solve`` doesn't reject a Float/Double
    # mix at the end.
    dtypes = (forces.dtype, positions.dtype, cell.dtype, virial.dtype)
    common_dtype = torch.float64 if torch.float64 in dtypes else torch.float32
    forces = forces.to(common_dtype)
    positions = positions.to(common_dtype)
    cell = cell.to(common_dtype)
    virial = virial.to(common_dtype)

    n_sys = cell.shape[0]
    g_r = -forces  # ∂E/∂r
    pos_virial_per_atom = g_r.unsqueeze(2) * positions.unsqueeze(1)  # (N, 3, 3)
    pos_virial = positions.new_zeros(n_sys, 3, 3)
    pos_virial = pos_virial.index_add(0, batch, pos_virial_per_atom)
    sym_pos_virial = 0.5 * (pos_virial + pos_virial.transpose(-1, -2))
    sym_cell_virial = virial - sym_pos_virial
    # Substitute identity for singular ``cell^T`` so ``torch.linalg.solve``
    # never raises on the all-zero mock tensors that ``TorchModuleWrapper``
    # uses for output-shape inference (CUDA's lstsq drivers also require full
    # rank, so we can't rely on them). The output values for singular cells
    # are meaningless and discarded by the wrapper's mock pass.
    cell_T = cell.transpose(-1, -2)
    det = torch.linalg.det(cell_T)
    eye = cell.new_zeros(3, 3)
    eye.fill_diagonal_(1.0)
    eye = eye.expand_as(cell_T)
    is_singular = (det.abs() < 1e-12).view(-1, 1, 1).expand_as(cell_T)
    safe_cell_T = cell_T.where(~is_singular, eye)
    return torch.linalg.solve(safe_cell_T, sym_cell_virial)


class AtomGraphInput(TypedDict):
    """Universal input schema shared by all torch MLFF backends.

    Mirrors the JAX [AtomGraphInput][kups.potential.mliap.tojax.AtomGraphInput].
    Shapes use ``N`` atoms, ``B`` systems, and ``E`` edges (each padded by one
    extra atom/system to work around backends that cannot handle empty graphs).
    """

    pos: Array  # (N, 3)
    atomic_numbers: Array  # (N,)
    cell: Array  # (B, 3, 3)
    pbc: Array  # (B, 3)
    edge_index: Array  # (2, E)
    cell_offsets: Array  # (E, 3) integer multiples of cell vectors
    batch: Array  # (N,)
    charge: Array  # (B,)
    spin: Array  # (B,)


class TorchMliapForward(Protocol):
    """Forward contract for a torch MLFF module.

    The module must accept an ``AtomGraphInput`` dict and return a dict with:

    - ``"energy"``: ``(B,)`` per-system total energies.
    - ``"position_gradients"``: ``(N, 3)`` :math:`\\partial E / \\partial r`.
    - ``"cell_gradients"``: ``(B, 3, 3)`` :math:`\\partial E / \\partial h`,
      required only when ``compute_cell_gradients=True``.

    Outputs are gradients (not forces); adapters around models that natively
    produce forces/virials negate appropriately inside the module.
    """

    def __call__(self, input: AtomGraphInput) -> dict[str, Array]: ...


class IsTorchMliapParticles(IsRadiusGraphPoints, HasAtomicNumbers, Protocol):
    """Particle protocol for torch MLFF models."""

    ...


@dataclass
class TorchMliap:
    """Container for a torch MLFF wired into JAX.

    Attributes:
        cutoff: Per-system cutoff radius [Å].
        wrapper: ``TorchModuleWrapper`` over the MLFF module.
        compute_cell_gradients: Whether the module returns ``"cell_gradients"``.
    """

    cutoff: Table[SystemId, Array]
    wrapper: TorchModuleWrapper = field(static=True)
    compute_cell_gradients: bool = field(static=True, default=False)

    @staticmethod
    def from_module(
        module: torch.nn.Module,
        cutoff: float,
        compute_cell_gradients: bool = False,
    ) -> "TorchMliap":
        """Wrap a torch.nn.Module that returns energy and gradients.

        Args:
            module: torch ``nn.Module`` satisfying ``TorchMliapForward``.
            cutoff: Interaction cutoff radius [Å].
            compute_cell_gradients: Whether the module returns
                ``"cell_gradients"`` for stress computation.

        Returns:
            Configured ``TorchMliap`` ready for use with the kUPS interface.
        """
        wrapper = TorchModuleWrapper(module, requires_grad=True)
        return TorchMliap(
            cutoff=Table((SystemId(0),), jnp.array([cutoff], float)),
            wrapper=wrapper,
            compute_cell_gradients=compute_cell_gradients,
        )

    def call(self, input: AtomGraphInput) -> dict[str, Array]:
        """Call the wrapped module on a prepared ``AtomGraphInput``."""
        return self.wrapper(input)


type TorchMliapInput[
    P: IsTorchMliapParticles,
    S: HasCell,
] = GraphPotentialInput[TorchMliap, P, S, Literal[2]]


def _prepare_torch_inputs(graph: Any) -> AtomGraphInput:
    """Convert a sorted kUPS graph to ``AtomGraphInput``.

    Unlike the JAX-exported [tojax][kups.potential.mliap.tojax] path — which
    pads to keep symbolic shapes stable across calls — the torch bridge
    operates on the raw real-system data. Torch handles dynamic shapes
    natively (no XLA-style recompilation per shape), and several backends
    (notably UMA's ``merge_mole`` mode) reject any padding-introduced phantom
    system because it changes the per-batch composition.

    Args:
        graph: ``HyperGraph`` already sorted by system.

    Returns:
        Prepared inputs in the universal schema (no padding).
    """
    n_sys = graph.systems.data.cell.vectors.shape[0]
    positions = graph.particles.data.positions
    atomic_numbers = graph.particles.data.atomic_numbers
    batch = graph.particles.data.system.indices
    cell = graph.systems.data.cell.vectors
    edge_indices = graph.edges.indices.indices_in(graph.particles.keys)

    return AtomGraphInput(
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


@overload
def torch_mliap_model_fn[
    P: IsTorchMliapParticles,
    S: HasCell,
](
    inp: TorchMliapInput[P, S],
    *,
    compute_cell_gradients: Literal[False] = False,
) -> WithPatch[PotentialOut[Array, EmptyType], IdPatch]: ...


@overload
def torch_mliap_model_fn[
    P: IsTorchMliapParticles,
    S: HasCell,
](
    inp: TorchMliapInput[P, S],
    *,
    compute_cell_gradients: Literal[True],
) -> WithPatch[PotentialOut[PositionAndCell, EmptyType], IdPatch]: ...


def torch_mliap_model_fn[
    P: IsTorchMliapParticles,
    S: HasCell,
](
    inp: TorchMliapInput[P, S],
    *,
    compute_cell_gradients: bool = False,
) -> (
    WithPatch[PotentialOut[Array, EmptyType], IdPatch]
    | WithPatch[PotentialOut[PositionAndCell, EmptyType], IdPatch]
):
    """Run a ``TorchMliap`` on a graph input and package the result.

    Args:
        inp: Graph potential input bundling the model and graph.
        compute_cell_gradients: Whether to wrap ``"cell_gradients"`` into a
            ``PositionAndCell`` gradients structure.

    Returns:
        ``WithPatch`` containing ``PotentialOut`` with energy, gradients, and
        an identity patch.
    """
    graph, sort_order = inp.graph.sorted_by_system(
        sort_edges=True, return_sort_order=True
    )
    unsort_order = jnp.argsort(sort_order)

    input_dict = _prepare_torch_inputs(graph)
    result = inp.parameters.call(input_dict)

    # Torch backends may run at a different (typically lower) precision than
    # the JAX side (e.g. UMA's predict-unit casts to float32 internally;
    # MACE may be loaded as float32 while JAX runs in x64). Pin every output
    # to the JAX input ``pos`` dtype here so adapters don't need to think
    # about precision and downstream ``lax.scan``/optax pipelines see
    # consistent types.
    out_dtype = input_dict["pos"].dtype
    energy = result["energy"].astype(out_dtype)
    pos_grad = result["position_gradients"][unsort_order].astype(out_dtype)
    energy_table = Table.arange(energy, label=SystemId)

    if compute_cell_gradients:
        cell_grad = result["cell_gradients"].astype(out_dtype)
        # Preserve the input cell/frame type: project the raw ∂E/∂h onto
        # the frame's parameter space via its ``from_matrix`` classmethod,
        # then swap in the new frame on a copy of the input cell.
        input_cell = inp.graph.systems.data.cell
        new_frame = input_cell.frame.from_matrix(cell_grad)
        new_cell = bind(input_cell, lambda c: c.frame).set(new_frame)
        gradients = PositionAndCell(
            positions=Table(inp.graph.particles.keys, pos_grad),
            cell=Table(inp.graph.systems.keys, new_cell),
        )
        return WithPatch(
            PotentialOut(energy_table, gradients, EMPTY),
            IdPatch(),
        )
    return WithPatch(
        PotentialOut(energy_table, pos_grad, EMPTY),
        IdPatch(),
    )


@overload
def make_torch_mliap_potential[
    State,
    P: IsTorchMliapParticles,
    S: HasCell,
    NNList: NeighborList[Literal[2]],
](
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NNList],
    model: View[State, TorchMliap] | TorchMliap,
    compute_cell_gradients: Literal[False] = False,
    patch_idx_view: View[State, PotentialOut[Array, EmptyType]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Array, EmptyType]] | None = None,
) -> Potential[State, Array, EmptyType, Patch[State]]: ...


@overload
def make_torch_mliap_potential[
    State,
    P: IsTorchMliapParticles,
    S: HasCell,
    NNList: NeighborList[Literal[2]],
](
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NNList],
    model: View[State, TorchMliap] | TorchMliap,
    compute_cell_gradients: Literal[True],
    patch_idx_view: View[State, PotentialOut[PositionAndCell, EmptyType]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[PositionAndCell, EmptyType]] | None = None,
) -> Potential[State, PositionAndCell, EmptyType, Patch[State]]: ...


def make_torch_mliap_potential(
    particles_view: Any,
    systems_view: Any,
    neighborlist_view: Any,
    model: Any,
    compute_cell_gradients: bool = False,
    patch_idx_view: Any | None = None,
    out_cache_lens: Any | None = None,
) -> Any:
    """Create a kUPS ``Potential`` from a ``TorchMliap``.

    Forces (and optionally stress) are computed inside the torch module; the
    kUPS side just routes the precomputed gradients through ``DirectPotential``.

    Args:
        particles_view: Extracts particle data from state.
        systems_view: Extracts system data (cell) from state.
        neighborlist_view: Extracts a cutoff-bound neighbor list from state.
        model: ``TorchMliap`` instance or view to model in state.
        compute_cell_gradients: When ``True``, exposes cell gradients
            (i.e. stress). The wrapped module must produce ``"cell_gradients"``.
        patch_idx_view: Cached output index structure (optional).
        out_cache_lens: Cache location lens (optional).

    Returns:
        Configured ``Potential`` backed by the torch MLFF.
    """
    model_view = constant(model) if isinstance(model, TorchMliap) else model
    if compute_cell_gradients:

        def cell_fn(inp: Any) -> Any:
            return torch_mliap_model_fn(inp, compute_cell_gradients=True)

        fn: Any = cell_fn
    else:

        def pos_fn(inp: Any) -> Any:
            return torch_mliap_model_fn(inp, compute_cell_gradients=False)

        fn = pos_fn
    return make_direct_mliap_potential(
        model_fn=fn,
        particles_view=particles_view,
        systems_view=systems_view,
        neighborlist_view=neighborlist_view,
        model_view=model_view,
        patch_idx_view=patch_idx_view,
        out_cache_lens=out_cache_lens,
    )


class IsTorchMliapState(
    IsAdaptiveCutoffNeighborListState[IsUniversalNeighborlistParams], Protocol
):
    """Protocol for states providing all inputs for a torch MLFF potential."""

    @property
    def particles(self) -> Table[ParticleId, IsTorchMliapParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasCell]: ...
    @property
    def torch_mliap_model(self) -> TorchMliap: ...


@overload
def make_torch_mliap_from_state[State](
    state: Lens[State, IsTorchMliapState],
    *,
    compute_position_and_cell_gradients: Literal[False] = ...,
    neighborlist_factory: NeighborListFactory[IsTorchMliapState] = ...,
) -> Potential[State, Array, EmptyType, Any]: ...


@overload
def make_torch_mliap_from_state[State](
    state: Lens[State, IsTorchMliapState],
    *,
    compute_position_and_cell_gradients: Literal[True],
    neighborlist_factory: NeighborListFactory[IsTorchMliapState] = ...,
) -> Potential[State, PositionAndCell, EmptyType, Any]: ...


def make_torch_mliap_from_state(
    state: Any,
    *,
    compute_position_and_cell_gradients: bool = False,
    neighborlist_factory: NeighborListFactory[
        Any
    ] = adaptive_cutoff_neighborlist_from_state,
) -> Any:
    """Create a torch MLFF potential from a typed state.

    Args:
        state: Lens into a sub-state providing particles, systems, neighbor
            list, and torch MLFF model.
        compute_position_and_cell_gradients: When ``True``, exposes both
            position and cell gradients. Requires the underlying
            ``TorchMliap.compute_cell_gradients`` to be ``True``.

    Returns:
        Configured torch MLFF ``Potential``.
    """
    model_view = state.focus(lambda x: x.torch_mliap_model)

    def neighborlist_view(s):
        return neighborlist_factory(state(s), model_view(s).cutoff)

    return make_torch_mliap_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        neighborlist_view,
        model_view,
        compute_cell_gradients=compute_position_and_cell_gradients,
    )
