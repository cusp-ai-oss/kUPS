# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the universal PyTorch MLFF interface ``torch_mliap_model_fn``.

Requires the torch_dev dependency group: `uv sync --group torch_dev`.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

# Skip the whole module if torch isn't installed.
torch = pytest.importorskip("torch", minversion="2.0.0")

from kups.core import units  # noqa: E402
from kups.core.cell import PeriodicCell, TriclinicFrame  # noqa: E402
from kups.core.data import Index, Table  # noqa: E402
from kups.core.neighborlist import Edges  # noqa: E402
from kups.core.typing import (  # noqa: E402
    ExclusionId,
    InclusionId,
    ParticleId,
    SystemId,
)
from kups.core.units import testing  # noqa: E402
from kups.core.utils.jax import dataclass  # noqa: E402
from kups.potential.common.graph import GraphPotentialInput, HyperGraph  # noqa: E402
from kups.potential.mliap.torch.interface import (  # noqa: E402
    TorchMliap,
    torch_mliap_model_fn,
)
from test.units import FORCE, A  # noqa: E402


@pytest.fixture
def enable_x64():
    """Enable JAX x64 mode for the test, restoring the prior setting after."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@dataclass
class _Atoms:
    positions: units.Length
    atomic_numbers: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class _Systems:
    cell: PeriodicCell


class _IdentityGradModel:
    """Pure-JAX stand-in for a torch MLFF returning ``∂E/∂r`` = input positions.

    The interface feeds this the *sorted* positions, so the model's per-atom
    gradient is the sorted positions; the unsorted output equals the original
    (pre-sort) positions exactly iff the sort permutation is inverted correctly.
    """

    def call(self, input: dict[str, Array]) -> dict[str, Array]:
        n_sys = input["cell"].shape[0]
        return {
            "energy": jnp.zeros(n_sys, dtype=input["pos"].dtype),
            "position_gradients": input["pos"],
            "cell_gradients": jnp.zeros((n_sys, 3, 3), dtype=input["pos"].dtype),
        }


def _interleaved_graph() -> tuple[HyperGraph[Any, Any, Literal[2]], Array]:
    """Build a 3-system graph whose per-atom system order is non-monotonic.

    The non-trivial ordering forces ``sorted_by_system`` to produce a real
    (non-identity) permutation, so inverting it actually matters.

    Returns:
        The graph and the original (pre-sort) atom positions in Å.
    """
    sys_of_atom = [1, 0, 1, 0, 2, 1, 0, 2]
    n = len(sys_of_atom)
    n_sys = 3
    positions = jnp.arange(n * 3, dtype=float).reshape(n, 3)
    cell = PeriodicCell(
        TriclinicFrame.from_matrix(jnp.eye(3)[None].repeat(n_sys, 0) * 20.0 * A)
    )
    atoms = Table.arange(
        _Atoms(
            positions=positions * A,
            atomic_numbers=jnp.zeros(n, dtype=int),
            system=Index.new([SystemId(s) for s in sys_of_atom]),
            inclusion=Index.new([InclusionId(0)] * n),
            exclusion=Index.new([ExclusionId(0)] * n),
        ),
        label=ParticleId,
    )
    systems = Table.arange(_Systems(cell=cell), label=SystemId)
    edge_pairs = jnp.array([[0, 2], [1, 3]])
    edges = Edges(indices=Index(atoms.keys, edge_pairs), shifts=jnp.zeros((2, 1, 3)))
    return HyperGraph(atoms, systems, edges), positions


@pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="XLA argsort-of-a-permutation miscompile is GPU-only",
)
@pytest.mark.usefixtures("enable_x64")
def test_unsort_recovers_original_atom_order():
    """Position gradients are returned in the original (pre-sort) atom order.

    Regression for an XLA GPU miscompile: ``torch_mliap_model_fn`` sorts atoms
    by system and inverts that permutation to unsort the model's per-atom
    gradients. Inverting with ``jnp.argsort`` (argsort-of-a-permutation) trips
    XLA's ``permutation_sort_simplifier``, which emits invalid int64 scatter HLO
    under x64 mode and crashes at compile time on GPU. Inverting via scatter
    compiles and yields the correct order.
    """
    graph, positions = _interleaved_graph()

    @jax.jit
    def run(g: HyperGraph[Any, Any, Literal[2]]):
        inp = GraphPotentialInput(parameters=cast(Any, _IdentityGradModel()), graph=g)
        return torch_mliap_model_fn(inp).data.gradients

    pos_grad = run(graph).positions.data
    testing.assert_allclose(pos_grad, positions * FORCE, FORCE, rtol=0, atol=0)


class _SumOfSquares(torch.nn.Module):
    """Stand-in MLFF with ``E = Σ r²`` per system that keeps its last input."""

    def forward(self, input: dict[str, Any]) -> dict[str, Any]:
        self.seen = input
        pos, cell = input["pos"], input["cell"]
        energy = pos.new_zeros(cell.shape[0]).index_add(
            0, input["batch"], (pos**2).sum(-1)
        )
        return {
            "energy": energy,
            "position_gradients": 2 * pos,
            "cell_gradients": cell + 1,
        }


class _NeverCalled(torch.nn.Module):
    """Stand-in MLFF that fails when it is evaluated."""

    def forward(self, input: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("The model was called on an input without atoms.")


def _padded_input() -> dict[str, Any]:
    """Three systems, the middle one empty, followed by two padding atoms.

    The edges are two real pairs in system 0, a self-image pair in system 2, an
    unused buffer entry and a pair of padding atoms.
    """
    return {
        "pos": torch.arange(15.0).reshape(5, 3),
        "atomic_numbers": torch.tensor([29, 29, 29, 0, 0]),
        "cell": torch.eye(3) * torch.tensor([4.0, 5.0, 6.0]).view(3, 1, 1),
        "pbc": torch.tensor([[True] * 3, [False] * 3, [True, True, False]]),
        "edge_index": torch.tensor([[0, 1, 2, 5, 3], [1, 0, 2, 5, 4]]),
        "cell_offsets": torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
        ),
        "batch": torch.tensor([0, 0, 2, 3, 3]),
        "charge": torch.tensor([0.0, 1.0, 2.0]),
        "spin": torch.tensor([3.0, 4.0, 5.0]),
    }


def test_model_sees_only_occupied_systems_and_their_atoms():
    model = _SumOfSquares()
    inp = _padded_input()
    TorchMliap.from_module(model, 5.0 * A).wrapper.module(inp)
    seen = model.seen
    assert seen["pos"].tolist() == inp["pos"][:3].tolist()
    assert seen["atomic_numbers"].tolist() == [29, 29, 29]
    assert seen["batch"].tolist() == [0, 0, 1]
    assert seen["cell"].tolist() == inp["cell"][[0, 2]].tolist()
    assert seen["pbc"].tolist() == inp["pbc"][[0, 2]].tolist()
    assert seen["edge_index"].tolist() == [[0, 1, 2], [1, 0, 2]]
    assert seen["cell_offsets"].tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    assert seen["charge"].tolist() == [0.0, 2.0]
    assert seen["spin"].tolist() == [3.0, 5.0]


def test_padding_rows_and_empty_systems_get_zeros():
    inp = _padded_input()
    mliap = TorchMliap.from_module(
        _SumOfSquares(), 5.0 * A, compute_cell_gradients=True
    )
    out = mliap.wrapper.module(inp)
    assert out["energy"].tolist() == [55.0, 0.0, 149.0]
    assert out["position_gradients"][:3].tolist() == (2 * inp["pos"][:3]).tolist()
    assert out["position_gradients"][3:].tolist() == [[0.0] * 3] * 2
    assert out["cell_gradients"][[0, 2]].tolist() == (inp["cell"][[0, 2]] + 1).tolist()
    assert out["cell_gradients"][1].tolist() == [[0.0] * 3] * 3


@pytest.mark.parametrize("compute_cell_gradients", [False, True])
def test_input_without_atoms_gives_zeros_without_a_model_call(
    compute_cell_gradients: bool,
):
    inp = _padded_input()
    inp["batch"] = torch.full((5,), 3)
    mliap = TorchMliap.from_module(
        _NeverCalled(), 5.0 * A, compute_cell_gradients=compute_cell_gradients
    )
    out = mliap.wrapper.module(inp)
    assert out["energy"].tolist() == [0.0] * 3
    assert out["position_gradients"].tolist() == [[0.0] * 3] * 5
    assert ("cell_gradients" in out) == compute_cell_gradients
    if compute_cell_gradients:
        assert out["cell_gradients"].tolist() == [[[0.0] * 3] * 3] * 3
