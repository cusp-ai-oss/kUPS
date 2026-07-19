# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Jaxified potential from kups.potential.mliap.tojax."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array, export

from kups.core.capacity import FixedCapacity
from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import bind, identity_lens, lens
from kups.core.neighborlist import AllDenseNearestNeighborList
from kups.core.potential import PotentialOut
from kups.core.result import as_result_function
from kups.core.typing import ExclusionId, InclusionId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.core.utils.msgpack import serialize as msgpack_serialize
from kups.potential.common.energy import PotentialFromEnergy
from kups.potential.mliap.tojax import (
    TojaxedMliap,
    make_tojaxed_evaluator,
    make_tojaxed_evaluator_from_state,
    make_tojaxed_potential,
)

from ..clear_cache import clear_cache  # noqa: F401

# --- Data structures ---


@dataclass
class AtomData:
    positions: Array
    atomic_numbers: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class SystemData:
    cell: Cell
    cutoff: Array


@dataclass
class State:
    atoms: Table[ParticleId, AtomData]
    systems: Table[SystemId, SystemData]
    jaxified_model: TojaxedMliap
    out: PotentialOut[tuple[()], tuple[()]]
    neighborlist: AllDenseNearestNeighborList

    @property
    def particles(self) -> Table[ParticleId, AtomData]:
        return self.atoms


# --- Mock helpers ---


def _make_mock_exported() -> tuple[export.Exported, list]:
    """Create a mock Jaxified exported function with symbolic shapes."""
    mock_params = [np.array(1.0, dtype=np.float32)]

    @jax.jit
    def mock_fn(params: list[jax.Array], x: dict[str, jax.Array]) -> jax.Array:
        return jnp.sum(x["pos"] ** 2) * params[0] * jnp.ones_like(x["charge"])

    concrete: dict[str, np.ndarray] = {
        "pos": np.zeros((4, 3), dtype=np.float32),
        "atomic_numbers": np.zeros((4,), dtype=np.int32),
        "cell": np.zeros((2, 3, 3), dtype=np.float32),
        "pbc": np.ones((2, 3), dtype=np.bool_),
        "edge_index": np.zeros((2, 6), dtype=np.int32),
        "cell_offsets": np.zeros((6, 3), dtype=np.float32),
        "charge": np.zeros((2,), dtype=np.float32),
        "spin": np.zeros((2,), dtype=np.float32),
        "batch": np.zeros((4,), dtype=np.int32),
    }
    shapes: dict[str, str] = {
        "pos": "n_atoms, _",
        "atomic_numbers": "n_atoms",
        "cell": "n_sys, _, _",
        "pbc": "n_sys, _",
        "edge_index": "_, n_edges",
        "cell_offsets": "n_edges, _",
        "charge": "n_sys",
        "spin": "n_sys",
        "batch": "n_atoms",
    }
    specs = export.symbolic_args_specs(
        (mock_params, concrete), shapes_specs=(None, shapes)
    )
    exported = export.export(mock_fn)(*specs)
    return exported, mock_params


def _make_mock_feature_exported() -> tuple[export.Exported, list]:
    """Create a structured energy and node-feature export."""
    mock_params = [np.array(2.0, dtype=np.float32)]

    @jax.jit
    def mock_fn(
        params: list[jax.Array], x: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        node_energies = jnp.sum(x["pos"] ** 2, axis=-1) * params[0]
        return {
            "energy": jnp.zeros_like(x["charge"]).at[x["batch"]].add(node_energies),
            "node_features": x["pos"] * params[0],
        }

    concrete: dict[str, np.ndarray] = {
        "pos": np.zeros((4, 3), dtype=np.float32),
        "atomic_numbers": np.zeros((4,), dtype=np.int32),
        "cell": np.zeros((2, 3, 3), dtype=np.float32),
        "pbc": np.ones((2, 3), dtype=np.bool_),
        "edge_index": np.zeros((2, 6), dtype=np.int32),
        "cell_offsets": np.zeros((6, 3), dtype=np.float32),
        "charge": np.zeros((2,), dtype=np.float32),
        "spin": np.zeros((2,), dtype=np.float32),
        "batch": np.zeros((4,), dtype=np.int32),
    }
    shapes: dict[str, str] = {
        "pos": "n_atoms, _",
        "atomic_numbers": "n_atoms",
        "cell": "n_sys, _, _",
        "pbc": "n_sys, _",
        "edge_index": "_, n_edges",
        "cell_offsets": "n_edges, _",
        "charge": "n_sys",
        "spin": "n_sys",
        "batch": "n_atoms",
    }
    specs = export.symbolic_args_specs(
        (mock_params, concrete), shapes_specs=(None, shapes)
    )
    exported = export.export(mock_fn)(*specs)
    return exported, mock_params


# --- Fixtures ---


@pytest.fixture(scope="module")
def mock_exported():
    return _make_mock_exported()


@pytest.fixture(scope="module")
def jaxified_model(mock_exported):
    exported, params = mock_exported
    return TojaxedMliap(
        cutoff=Table((SystemId(0),), jnp.array([5.0])),
        params=params,
        model=exported,
    )


@pytest.fixture(scope="module")
def jaxified_feature_model():
    exported, params = _make_mock_feature_exported()
    return TojaxedMliap(
        cutoff=Table((SystemId(0),), jnp.array([5.0])),
        params=params,
        model=exported,
        node_feature_dimension=3,
    )


@pytest.fixture(scope="module")
def simple_system(jaxified_model):
    """4 atoms in a 10 Angstrom cubic box."""
    n_atoms = 4
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]]
    )
    atomic_numbers = jnp.array([6, 8, 6, 8], dtype=int)
    system_ids = Index.new([SystemId(0)] * n_atoms)
    inclusion_ids = Index.new([InclusionId(0)] * n_atoms)
    exclusion_ids = Index.new([ExclusionId(i) for i in range(n_atoms)])
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))

    return State(
        atoms=Table.arange(
            AtomData(
                positions=positions,
                atomic_numbers=atomic_numbers,
                system=system_ids,
                inclusion=inclusion_ids,
                exclusion=exclusion_ids,
            ),
            label=ParticleId,
        ),
        systems=Table.arange(
            SystemData(cell=cell, cutoff=jnp.array([5.0])),
            label=SystemId,
        ),
        jaxified_model=jaxified_model,
        out=PotentialOut(Table.arange(jnp.zeros((1,)), label=SystemId), (), ()),
        neighborlist=AllDenseNearestNeighborList(
            avg_edges=FixedCapacity(n_atoms),
            avg_image_candidates=FixedCapacity(n_atoms),
        ),
    )


@pytest.fixture(scope="module")
def potential(jaxified_model):
    return make_tojaxed_potential(
        particles_view=lambda s: s.atoms,
        systems_view=lambda s: s.systems,
        neighborlist_view=lambda s: s.neighborlist,
        model=jaxified_model,
        cutoffs_view=lambda s: s.systems.map_data(lambda d: d.cutoff),
        gradient_lens=lens(lambda x: ()),
        hessian_lens=lens(lambda x: ()),
        hessian_idx_view=lambda x: (),
        out_cache_lens=lens(lambda x: x.out, cls=State),
        patch_idx_view=lambda x: PotentialOut(
            Table((SystemId(0),), Index.new((SystemId(0),)), _cls=SystemId), (), ()
        ),
    )


# --- Tests ---


class TestTojaxedMliapFromZipFile:
    def test_loads_model_correctly(self, tmp_path, mock_exported):
        """Test from_zip_file loads cutoff, params, and model."""
        exported, params = mock_exported
        cutoff = 5.0
        zip_path = tmp_path / "mock.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("model.jax", exported.serialize())
            zf.writestr("metadata.json", json.dumps({"cutoff": cutoff}))
            zf.writestr("params.msgpack", msgpack_serialize(params))

        model = TojaxedMliap.from_zip_file(zip_path)

        assert jnp.allclose(model.cutoff.data, jnp.array([cutoff]))
        assert model.params is not None
        assert model.model is not None

    def test_loads_structured_feature_contract(self, tmp_path):
        exported, params = _make_mock_feature_exported()
        zip_path = tmp_path / "mock-features.zip"
        metadata = {
            "cutoff": 5.0,
            "node_features": {"dimension": 3, "scope": "node"},
            "output_schema": {
                "energy": {"shape": ["S"], "dtype": "float32"},
                "node_features": {
                    "shape": ["N", 3],
                    "dtype": "float32",
                },
            },
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("model.jax", exported.serialize(vjp_order=1))
            zf.writestr("metadata.json", json.dumps(metadata))
            zf.writestr("params.msgpack", msgpack_serialize(params))

        model = TojaxedMliap.from_zip_file(zip_path)

        assert model.node_feature_dimension == 3

    def test_rejects_inconsistent_feature_metadata(self, tmp_path, mock_exported):
        exported, params = mock_exported
        zip_path = tmp_path / "bad-features.zip"
        metadata = {
            "cutoff": 5.0,
            "node_features": {"dimension": 3, "scope": "node"},
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("model.jax", exported.serialize())
            zf.writestr("metadata.json", json.dumps(metadata))
            zf.writestr("params.msgpack", msgpack_serialize(params))

        with pytest.raises(ValueError, match="output_schema"):
            TojaxedMliap.from_zip_file(zip_path)


class TestJaxifiedEnergy:
    def test_energy_value(self, simple_system, potential):
        """Test energy computation matches expected value from mock model.

        Mock model computes energy = sum(pos^2) per system.
        Positions: [[0,0,0],[3,0,0],[0,3,0],[3,3,0]] -> sum = 36.
        """
        state = simple_system
        potential_fn = as_result_function(potential)

        state = potential_fn(state).fix_or_raise(state)
        result = potential_fn(state)
        result.raise_assertion()

        assert result.value.data.total_energies.data.shape == (1,)
        npt.assert_allclose(result.value.data.total_energies.data, jnp.array([36.0]))

    def test_structured_model_remains_energy_potential_compatible(
        self, simple_system, jaxified_feature_model
    ):
        state = replace(simple_system, jaxified_model=jaxified_feature_model)
        potential = make_tojaxed_potential(
            particles_view=lambda s: s.atoms,
            systems_view=lambda s: s.systems,
            neighborlist_view=lambda s: s.neighborlist,
            model=jaxified_feature_model,
            cutoffs_view=lambda s: s.systems.map_data(lambda d: d.cutoff),
            gradient_lens=lens(lambda x: ()),
            hessian_lens=lens(lambda x: ()),
            hessian_idx_view=lambda x: (),
        )

        result = as_result_function(potential)(state)
        result.raise_assertion()

        npt.assert_allclose(
            result.value.data.total_energies.data,
            jnp.array([72.0]),
        )


class TestJaxifiedEvaluator:
    def test_returns_single_pass_energy_and_features_in_particle_order(
        self, simple_system, jaxified_feature_model
    ):
        state = replace(simple_system, jaxified_model=jaxified_feature_model)
        evaluator = make_tojaxed_evaluator(
            particles_view=lambda s: s.atoms,
            systems_view=lambda s: s.systems,
            neighborlist_view=lambda s: s.neighborlist,
            model=lambda s: s.jaxified_model,
            cutoffs_view=lambda s: s.systems.map_data(lambda d: d.cutoff),
        )

        result = evaluator(state)

        npt.assert_allclose(result.total_energies.data, jnp.array([72.0]))
        npt.assert_allclose(
            result.node_features.data,
            2.0 * state.atoms.data.positions,
        )
        assert result.node_features.keys == state.atoms.keys

        def feature_norm(positions):
            updated = bind(state).focus(lambda s: s.atoms.data.positions).set(positions)
            return jnp.sum(evaluator(updated).node_features.data ** 2)

        gradient = jax.jit(jax.grad(feature_norm))(state.atoms.data.positions)
        npt.assert_allclose(gradient, 8.0 * state.atoms.data.positions)

    def test_from_state_uses_the_application_state_contract(
        self, simple_system, jaxified_feature_model
    ):
        state = replace(simple_system, jaxified_model=jaxified_feature_model)
        evaluator = make_tojaxed_evaluator_from_state(identity_lens(State))

        result = jax.jit(evaluator)(state)

        npt.assert_allclose(result.total_energies.data, jnp.array([72.0]))
        npt.assert_allclose(
            result.node_features.data,
            2.0 * state.particles.data.positions,
        )

    def test_restores_interleaved_particle_order(
        self, simple_system, jaxified_feature_model
    ):
        system_ids = Index.new([SystemId(1), SystemId(0), SystemId(1), SystemId(0)])
        positions = jnp.array(
            [
                [7.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        atoms = bind(simple_system.atoms).focus(lambda x: x.data.system).set(system_ids)
        atoms = bind(atoms).focus(lambda x: x.data.positions).set(positions)
        cells = PeriodicCell(
            TriclinicFrame.from_matrix(jnp.tile(jnp.eye(3)[None] * 10.0, (2, 1, 1)))
        )
        systems = Table.arange(
            SystemData(cell=cells, cutoff=jnp.array([5.0, 5.0])),
            label=SystemId,
        )
        state = replace(
            simple_system,
            atoms=atoms,
            systems=systems,
            jaxified_model=jaxified_feature_model,
        )
        evaluator = make_tojaxed_evaluator(
            particles_view=lambda s: s.atoms,
            systems_view=lambda s: s.systems,
            neighborlist_view=lambda s: s.neighborlist,
            model=lambda s: s.jaxified_model,
            cutoffs_view=lambda s: s.systems.map_data(lambda d: d.cutoff),
        )

        result = evaluator(state)

        assert result.node_features.keys == state.atoms.keys
        npt.assert_allclose(result.total_energies.data, jnp.array([10.0, 226.0]))
        npt.assert_allclose(result.node_features.data, 2.0 * positions)

    def test_rejects_energy_only_artifact(self, simple_system):
        evaluator = make_tojaxed_evaluator(
            particles_view=lambda s: s.atoms,
            systems_view=lambda s: s.systems,
            neighborlist_view=lambda s: s.neighborlist,
            model=lambda s: s.jaxified_model,
            cutoffs_view=lambda s: s.systems.map_data(lambda d: d.cutoff),
        )

        with pytest.raises(ValueError, match="node features"):
            evaluator(simple_system)


class TestMakeJaxifiedPotential:
    def test_returns_potential_from_energy(self, potential):
        assert isinstance(potential, PotentialFromEnergy)
