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
from kups.potential.mliap.torch.interface import (  # noqa: E402
    TorchMliap,
    torch_mliap_model_fn,
)


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

    charge = 0.0
    spin = 0.0

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


@dataclass
class _ChargedSystems:
    cell: PeriodicCell
    total_charge: Array
    spin_multiplicity: Array


class _ChargeSpinEchoModel:
    """Stand-in model whose per-system energy encodes the conditioning it saw.

    ``energy = 1000 * charge + spin`` lets a test read back exactly which
    ``charge``/``spin`` values ``torch_mliap_model_fn`` handed to the module.
    """

    def __init__(self, charge: float = 0.0, spin: float = 0.0):
        self.charge = charge
        self.spin = spin
        self.seen: dict[str, Array] = {}

    def call(self, input: dict[str, Array]) -> dict[str, Array]:
        self.seen = {"charge": input["charge"], "spin": input["spin"]}
        n_sys = input["cell"].shape[0]
        return {
            "energy": 1000.0 * input["charge"] + input["spin"],
            "position_gradients": jnp.zeros_like(input["pos"]),
            "cell_gradients": jnp.zeros((n_sys, 3, 3), dtype=input["pos"].dtype),
        }


def _graph_with_systems(systems_data: Any) -> HyperGraph[Any, Any, Literal[2]]:
    graph, _ = _interleaved_graph()
    systems = Table.arange(systems_data, label=SystemId)
    return HyperGraph(graph.particles, systems, graph.edges)


def _energies(model: Any, graph: HyperGraph[Any, Any, Literal[2]]) -> np.ndarray:
    inp = GraphPotentialInput(parameters=cast(Any, model), graph=graph)
    return np.asarray(torch_mliap_model_fn(inp).data.total_energies.data)


def _cell(n_sys: int = 3) -> PeriodicCell:
    return PeriodicCell(
        TriclinicFrame.from_matrix(jnp.eye(3)[None].repeat(n_sys, 0) * 20.0)
    )


class TestChargeAndSpin:
    """Per-system charge / spin conditioning reaches the torch module (#236)."""

    def test_defaults_to_zero_without_any_source(self):
        model = _ChargeSpinEchoModel()
        energies = _energies(model, _graph_with_systems(_Systems(cell=_cell())))
        np.testing.assert_array_equal(energies, [0.0, 0.0, 0.0])

    def test_model_defaults_broadcast_to_every_system(self):
        model = _ChargeSpinEchoModel(charge=-1.0, spin=2.0)
        energies = _energies(model, _graph_with_systems(_Systems(cell=_cell())))
        np.testing.assert_array_equal(energies, [-998.0, -998.0, -998.0])
        assert model.seen["charge"].shape == (3,)
        assert model.seen["spin"].shape == (3,)

    def test_system_fields_override_model_defaults_per_system(self):
        model = _ChargeSpinEchoModel(charge=5.0, spin=7.0)
        systems = _ChargedSystems(
            cell=_cell(),
            total_charge=jnp.array([0.0, 1.0, -2.0]),
            spin_multiplicity=jnp.array([1.0, 2.0, 3.0]),
        )
        energies = _energies(model, _graph_with_systems(systems))
        # One value per system, in system order, despite the interleaved atoms.
        np.testing.assert_array_equal(energies, [1.0, 1002.0, -1997.0])

    def test_conditioning_dtype_follows_positions(self):
        model = _ChargeSpinEchoModel(charge=1.0, spin=1.0)
        graph = _graph_with_systems(_Systems(cell=_cell()))
        _energies(model, graph)
        dtype = graph.particles.data.positions.dtype
        assert model.seen["charge"].dtype == dtype
        assert model.seen["spin"].dtype == dtype

    def test_works_under_jit(self):
        model = _ChargeSpinEchoModel(charge=0.0, spin=1.0)
        systems = _ChargedSystems(
            cell=_cell(),
            total_charge=jnp.zeros(3),
            spin_multiplicity=jnp.array([1.0, 3.0, 1.0]),
        )

        @jax.jit
        def run(g: HyperGraph[Any, Any, Literal[2]]):
            inp = GraphPotentialInput(parameters=cast(Any, model), graph=g)
            return torch_mliap_model_fn(inp).data.total_energies.data

        np.testing.assert_array_equal(
            np.asarray(run(_graph_with_systems(systems))), [1.0, 3.0, 1.0]
        )


class _TorchChargeSpinEcho(torch.nn.Module):
    """Real torch module: ``energy = 1000 * charge + spin`` per system."""

    def forward(self, input):  # type: ignore[override]
        n_sys = input["cell"].shape[0]
        return {
            "energy": 1000.0 * input["charge"] + input["spin"],
            "position_gradients": torch.zeros_like(input["pos"]),
            "cell_gradients": input["cell"].new_zeros(n_sys, 3, 3),
        }


class TestTorchMliapFromModule:
    @pytest.mark.skipif(
        jax.default_backend() != "gpu",
        reason="TorchModuleWrapper's buffer callback needs the XLA GPU context",
    )
    def test_conditioning_reaches_a_real_torch_module(self):
        model = TorchMliap.from_module(
            _TorchChargeSpinEcho(),
            cutoff=5.0,
            compute_cell_gradients=True,
            charge=-1,
            spin=2,
        )
        systems = _ChargedSystems(
            cell=_cell(),
            total_charge=jnp.array([0.0, 1.0, -2.0]),
            spin_multiplicity=jnp.array([1.0, 2.0, 3.0]),
        )
        np.testing.assert_allclose(
            _energies(model, _graph_with_systems(systems)), [1.0, 1002.0, -1997.0]
        )
        np.testing.assert_allclose(
            _energies(model, _graph_with_systems(_Systems(cell=_cell()))),
            [-998.0, -998.0, -998.0],
        )

    def test_charge_and_spin_default_to_zero(self):
        model = TorchMliap.from_module(torch.nn.Identity(), cutoff=5.0)
        assert model.charge == 0.0
        assert model.spin == 0.0

    def test_charge_and_spin_are_stored(self):
        model = TorchMliap.from_module(
            torch.nn.Identity(), cutoff=5.0, charge=-1, spin=2
        )
        assert model.charge == -1.0
        assert model.spin == 2.0
