# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the universal PyTorch MLFF interface ``torch_mliap_model_fn``.

Requires the torch_dev dependency group: `uv sync --group torch_dev`.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

# Skip the whole module if torch isn't installed.
torch = pytest.importorskip("torch", minversion="2.0.0")

from kups.core.cell import PeriodicCell, TriclinicFrame  # noqa: E402
from kups.core.data import Index, Table  # noqa: E402
from kups.core.neighborlist import Edges  # noqa: E402
from kups.core.typing import (  # noqa: E402
    ExclusionId,
    InclusionId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass  # noqa: E402
from kups.potential.common.graph import GraphPotentialInput, HyperGraph  # noqa: E402
from kups.potential.mliap.torch.interface import torch_mliap_model_fn  # noqa: E402


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
    positions: Array
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
        return {
            "energy": jnp.zeros(input["cell"].shape[0], dtype=input["pos"].dtype),
            "position_gradients": input["pos"],
        }


def _interleaved_graph() -> tuple[HyperGraph[Any, Any, Literal[2]], Array]:
    """Build a 3-system graph whose per-atom system order is non-monotonic.

    The non-trivial ordering forces ``sorted_by_system`` to produce a real
    (non-identity) permutation, so inverting it actually matters.

    Returns:
        The graph and the original (pre-sort) atom positions.
    """
    sys_of_atom = [1, 0, 1, 0, 2, 1, 0, 2]
    n = len(sys_of_atom)
    n_sys = 3
    positions = jnp.arange(n * 3, dtype=float).reshape(n, 3)
    cell = PeriodicCell(
        TriclinicFrame.from_matrix(jnp.eye(3)[None].repeat(n_sys, 0) * 20.0)
    )
    atoms = Table.arange(
        _Atoms(
            positions=positions,
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

    pos_grad = run(graph)
    np.testing.assert_array_equal(np.asarray(pos_grad), np.asarray(positions))
