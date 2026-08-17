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
    from kups.application.potential.filter import POSITIONS_AND_CELL
    from kups.application.potential.mliap.torch import make_torch_mliap_from_state
    from kups.potential.mliap.torch.interface import TorchMliap

    # A backend provides a Module with the universal forward contract:
    model = TorchMliap.from_module(my_module, cutoff=6.0, compute_cell_gradients=True)

    # Wire into a kUPS Potential:
    potential = make_torch_mliap_from_state(state_lens, gradient=POSITIONS_AND_CELL)
    ```

Requires the ``torch_dev`` dependency group: ``uv sync --group torch_dev``.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, overload

import jax.numpy as jnp
import torch  # pyright: ignore[reportMissingImports]
from jax import Array

from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Table
from kups.core.lens import Lens, View, bind
from kups.core.neighborlist import (
    NeighborList,
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
from kups.core.utils.kahan import KahanSummand
from kups.core.utils.torch import TorchModuleWrapper
from kups.potential.common.geometry import Geometry, PositionsAndCell
from kups.potential.common.graph import (
    GRAPH_GEOMETRY,
    GraphPotentialInput,
    IsRadiusGraphPoints,
)
from kups.potential.mliap.direct import (
    filter_pullback,
    make_direct_mliap_potential,
)

__all__ = [
    "AtomGraphInput",
    "IsTorchMliapParticles",
    "TorchMliap",
    "TorchMliapForward",
    "lattice_gradient_from_virial",
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
    that encodes the gradient of energy under a *symmetric infinitesimal strain*
    applied jointly to positions and cell. In kUPS's row convention
    (``r = frac @ h``; lattice vectors are the rows of ``h``) that virial is

        virial = pos_virial + cell_virial        (exactly symmetric)

    where

        pos_virial[s, j, k]  = Σ_{b∈s} (∂E/∂r_b)_j · (r_b)_k
        cell_virial          = (∂E/∂h)^T @ h

    Rotational invariance makes the *total* symmetric, but ``cell_virial`` on its
    own is not. Its antisymmetric part is pinned by ``pos_virial``, known exactly
    from forces and positions, so the raw lattice gradient (antisymmetric part
    included) is recovered by

        ∂E/∂h = h^-T @ (virial - pos_virial^T).

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
    # cell_virial = virial - pos_virial^T; ∂E/∂h = h^-T @ cell_virial (= solve(h^T, ·)).
    cell_virial = virial - pos_virial.transpose(-1, -2)
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
    return torch.linalg.solve(safe_cell_T, cell_virial)


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
    S: HasCell[AnyPeriodicity],
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
    pbc = jnp.broadcast_to(
        jnp.asarray(graph.systems.data.cell.periodic, dtype=bool), (n_sys, 3)
    )
    edge_indices = graph.edges.indices.indices_in(graph.particles.keys)

    return AtomGraphInput(
        pos=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        edge_index=edge_indices.T,
        cell_offsets=graph.edges.shifts.squeeze(1),
        batch=batch,
        charge=jnp.zeros(n_sys),
        spin=jnp.zeros(n_sys),
    )


def _project_grad_onto_frame[C: Cell[Any]](cell: C, cell_grad: Array) -> C:
    """Express a raw ``∂E/∂h`` matrix in ``cell``'s frame parameter space.

    Delegates to [`Frame.parameter_gradient`][kups.core.cell.Frame.parameter_gradient],
    which pulls the matrix gradient back onto the frame's own degrees of freedom
    (``tril`` for triclinic, ``lengths`` for orthogonal) via the ``vectors`` vjp,
    preserving the input frame type. Returns a copy of ``cell`` carrying the
    projected gradient as its frame.
    """
    return bind(cell, lambda c: c.frame).set(cell.frame.parameter_gradient(cell_grad))


def torch_mliap_model_fn[
    P: IsTorchMliapParticles,
    S: HasCell[AnyPeriodicity],
](
    inp: TorchMliapInput[P, S],
) -> WithPatch[PotentialOut[PositionsAndCell, EmptyType], IdPatch[Any]]:
    """Run a ``TorchMliap`` on a graph input and package the result.

    Always packages ``"cell_gradients"`` into a ``PositionsAndCell`` gradients
    structure (the module must produce them); downstream consumers that only
    need forces let XLA prune the unused cell-gradient ops.

    Args:
        inp: Graph potential input bundling the model and graph.

    Returns:
        ``WithPatch`` containing ``PotentialOut`` with energy, ``PositionsAndCell``
        gradients, and an identity patch.
    """
    graph, sort_order = inp.graph.sorted_by_system(
        sort_edges=True, return_sort_order=True
    )
    # Invert the permutation via scatter rather than a second argsort: XLA's
    # permutation_sort_simplifier miscompiles argsort-of-a-permutation with
    # int64 indices (x64 mode) on GPU.
    n = sort_order.shape[0]
    unsort_order = (
        jnp.zeros(n, dtype=sort_order.dtype)
        .at[sort_order]
        .set(jnp.arange(n, dtype=sort_order.dtype))
    )

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
    # Zero padded-particle force rows: their system index is the OOB sentinel,
    # which the downstream ``cell[system]`` gather (e.g. in the filter pullback)
    # silently clamps to a real system, contaminating its virial.
    valid = inp.graph.particles.data.system.valid_mask
    pos_grad = jnp.where(valid[:, None], pos_grad, 0.0)
    energy_table = Table.arange(energy, label=SystemId)

    cell_grad = result["cell_gradients"].astype(out_dtype)
    # Project the raw ∂E/∂h onto the input frame's parameter space,
    # preserving its type for downstream stress/relaxation consumers.
    new_cell = _project_grad_onto_frame(inp.graph.systems.data.cell, cell_grad)
    gradients = PositionsAndCell(
        positions=Table(inp.graph.particles.keys, pos_grad),
        cell=Table(inp.graph.systems.keys, new_cell),
    )
    return WithPatch(
        PotentialOut(energy_table, gradients, EMPTY),
        IdPatch[Any](),
    )


@overload
def make_torch_mliap_potential[
    State,
    P: IsTorchMliapParticles,
    S: HasCell[AnyPeriodicity],
    NNList: NeighborList[Literal[2]],
](
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NNList],
    model: View[State, TorchMliap] | TorchMliap,
    patch_idx_view: View[State, PotentialOut[PositionsAndCell, EmptyType]]
    | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[PositionsAndCell, EmptyType]]]
    | None = None,
) -> Potential[State, PositionsAndCell, EmptyType, Patch[State]]: ...


@overload
def make_torch_mliap_potential[
    State,
    P: IsTorchMliapParticles,
    S: HasCell[AnyPeriodicity],
    NNList: NeighborList[Literal[2]],
](
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    neighborlist_view: View[State, NNList],
    model: View[State, TorchMliap] | TorchMliap,
    patch_idx_view: View[State, PotentialOut[PositionsAndCell, EmptyType]]
    | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[PositionsAndCell, EmptyType]]]
    | None = None,
    *,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[State]]: ...


def make_torch_mliap_potential(
    particles_view: Any,
    systems_view: Any,
    neighborlist_view: Any,
    model: Any,
    patch_idx_view: Any | None = None,
    out_cache_lens: Any | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create a kUPS ``Potential`` from a ``TorchMliap``.

    Forces and stress are computed inside the torch module; the kUPS side just
    routes the precomputed ``PositionsAndCell`` gradients through
    ``DirectPotential``. Without a ``gradient`` the raw ``PositionsAndCell``
    gradients pass through; with one they are pulled back through ``gradient.set``
    into ``∂E/∂u`` — the pullback is hooked here, where the gradients are concretely
    ``PositionsAndCell``.

    Args:
        particles_view: Extracts particle data from state.
        systems_view: Extracts system data (cell) from state.
        neighborlist_view: Extracts a cutoff-bound neighbor list from state.
        model: ``TorchMliap`` instance or view to model in state.
        patch_idx_view: Cached output index structure (optional).
        out_cache_lens: Cache location lens (optional).
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]``
            selecting the optimizer DOFs.

    Returns:
        Configured ``Potential`` backed by the torch MLFF.
    """
    model_view = constant(model) if isinstance(model, TorchMliap) else model
    model_fn: Any
    if gradient is None:
        model_fn = torch_mliap_model_fn
    else:

        def model_fn[P: IsTorchMliapParticles, S: HasCell[AnyPeriodicity]](
            inp: TorchMliapInput[P, S],
        ) -> WithPatch[PotentialOut[PositionsAndCell, EmptyType], IdPatch[Any]]:
            result = torch_mliap_model_fn(inp)
            data = result.data
            geometry = GRAPH_GEOMETRY.get(inp)
            dof_gradient = filter_pullback(geometry, data.gradients, gradient)
            return WithPatch(
                PotentialOut(data.total_energies, dof_gradient, data.hessians),
                result.patch,
            )

    return make_direct_mliap_potential(
        model_fn=model_fn,
        particles_view=particles_view,
        systems_view=systems_view,
        neighborlist_view=neighborlist_view,
        model_view=model_view,
        patch_idx_view=patch_idx_view,
        out_cache_lens=out_cache_lens,
    )
