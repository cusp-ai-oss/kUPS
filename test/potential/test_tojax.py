# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Jaxified potential from kups.potential.mliap.tojax."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from typing import Any, Literal

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
from kups.core.lens import lens
from kups.core.neighborlist import AllDenseNearestNeighborList, Edges
from kups.core.potential import PotentialOut
from kups.core.result import as_result_function
from kups.core.typing import ExclusionId, InclusionId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.core.utils.kahan import KahanSummand
from kups.core.utils.msgpack import serialize as msgpack_serialize
from kups.potential.common.energy import PotentialFromEnergy
from kups.potential.common.geometry import PositionsAndCell
from kups.potential.common.graph import (
    GRAPH_GEOMETRY,
    GraphPotentialInput,
    HyperGraph,
)
from kups.potential.mliap.tojax import (
    TojaxedMliap,
    make_tojaxed_potential,
    tojaxed_energy,
)

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
    out: KahanSummand[PotentialOut[tuple[()], tuple[()]]]
    neighborlist: AllDenseNearestNeighborList


# --- Mock helpers ---


_SHAPES: dict[str, str] = {
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


def _export(
    fn, params: list, float_dtype, int_dtype, metadata_dtype
) -> export.Exported:
    """Export ``fn(params, x)`` with symbolic atom, system and edge counts."""
    concrete: dict[str, np.ndarray] = {
        "pos": np.zeros((4, 3), dtype=float_dtype),
        "atomic_numbers": np.zeros((4,), dtype=int_dtype),
        "cell": np.zeros((2, 3, 3), dtype=float_dtype),
        "pbc": np.ones((2, 3), dtype=np.bool_),
        "edge_index": np.zeros((2, 6), dtype=int_dtype),
        "cell_offsets": np.zeros((6, 3), dtype=float_dtype),
        "charge": np.zeros((2,), dtype=metadata_dtype),
        "spin": np.zeros((2,), dtype=metadata_dtype),
        "batch": np.zeros((4,), dtype=int_dtype),
    }
    specs = export.symbolic_args_specs((params, concrete), shapes_specs=(None, _SHAPES))
    return export.export(jax.jit(fn))(*specs)


def _make_mock_exported() -> tuple[export.Exported, list]:
    """Create a mock Jaxified exported function with symbolic shapes."""
    mock_params = [np.array(1.0, dtype=np.float32)]

    def mock_fn(params: list[jax.Array], x: dict[str, jax.Array]) -> jax.Array:
        return jnp.sum(x["pos"] ** 2) * params[0] * jnp.ones_like(x["charge"])

    exported = _export(mock_fn, mock_params, np.float32, np.int32, np.float32)
    return exported, mock_params


def _write_zip(path, exported: export.Exported, params: list, cutoff: float = 5.0):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("model.jax", exported.serialize())
        zf.writestr("metadata.json", json.dumps({"cutoff": cutoff}))
        zf.writestr("params.msgpack", msgpack_serialize(params))
    return path


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
        out=KahanSummand.init(
            PotentialOut(Table.arange(jnp.zeros((1,)), label=SystemId), (), ())
        ),
        neighborlist=AllDenseNearestNeighborList(
            avg_edges=FixedCapacity(n_atoms),
            avg_image_candidates=FixedCapacity(n_atoms),
            cutoffs=Table((SystemId(0),), jnp.array([5.0])),
        ),
    )


@pytest.fixture(scope="module")
def potential(jaxified_model):
    return make_tojaxed_potential(
        particles_view=lambda s: s.atoms,
        systems_view=lambda s: s.systems,
        neighborlist_view=lambda s: s.neighborlist,
        model=jaxified_model,
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
        zip_path = _write_zip(tmp_path / "mock.zip", exported, params, cutoff)

        model = TojaxedMliap.from_zip_file(zip_path)

        assert jnp.allclose(model.cutoff.data, jnp.array([cutoff]))
        assert model.params is not None
        assert model.model is not None
        assert model.task_name is None

    def test_from_zip_file_task_name(self, tmp_path, mock_exported):
        zip_path = _write_zip(tmp_path / "mock.zip", *mock_exported)
        assert (
            TojaxedMliap.from_zip_file(zip_path, task_name="omol").task_name == "omol"
        )
        with pytest.raises(ValueError, match="omoll"):
            TojaxedMliap.from_zip_file(zip_path, task_name="omoll")  # type: ignore[arg-type]


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


class TestMakeJaxifiedPotential:
    def test_returns_potential_from_energy(self, potential):
        assert isinstance(potential, PotentialFromEnergy)


# --- Per-system charge and spin for UMA omol ---


@dataclass
class _GraphAtoms:
    positions: Array
    atomic_numbers: Array
    system: Index[SystemId]


@dataclass
class _MetaSystems:
    cell: Cell
    total_charge: Array | None
    spin_multiplicity: Array | None


@dataclass
class _ChargeOnlySystems:
    cell: Cell
    total_charge: Array


@dataclass
class _PlainSystems:
    cell: Cell


_KEYS = (SystemId(12), SystemId(37), SystemId(91))
_CHARGE = jnp.array([0, 1, 0])
_SPIN = jnp.array([1, 1, 2])
# Sum of squared coordinates of each system's atoms, in key order.
_SQ_SUMS = np.array([14.37, 9.24, 19.63])


def _cell(n_sys: int = 3) -> Cell:
    return Cell.from_pbc(
        TriclinicFrame.from_matrix(jnp.tile(jnp.eye(3)[None] * 20, (n_sys, 1, 1))),
        (False, False, False),
    )


def _meta_systems(charge=_CHARGE, spin=_SPIN) -> _MetaSystems:
    return _MetaSystems(_cell(), total_charge=charge, spin_multiplicity=spin)


def _metadata_graph(systems: Any) -> HyperGraph[Any, Any, Literal[2]]:
    """Interleaved atoms with unequal counts and non-consecutive system keys."""
    atoms = Table.arange(
        _GraphAtoms(
            jnp.arange(24, dtype=float).reshape(8, 3) / 10,
            jnp.ones(8, dtype=int),
            Index(_KEYS, jnp.array([1, 0, 1, 0, 2, 1, 0, 2])),
        ),
        label=ParticleId,
    )
    edges = Edges(
        Index(atoms.keys, jnp.array([[0, 2], [1, 3], [4, 7]])),
        jnp.zeros((3, 1, 3)),
    )
    return HyperGraph(atoms, Table(_KEYS, systems), edges)


def _make_metadata_exported(
    cell_term: bool = False, leak_padding: bool = False
) -> export.Exported:
    """Export a model whose energies depend on charge and spin per system.

    ``E_s = 0.5 * (1 + charge_s + 10 * spin_s) * sum(|r_i|^2)`` over the atoms
    of system ``s``. ``cell_term`` adds ``(charge_s + spin_s) * trace(cell_s)``.
    ``leak_padding`` adds ``1e3 * (charge_pad**2 + (spin_pad - 1)**2)`` to every
    system, so any padding row other than charge 0, spin 1 shows in real
    energies.
    Avals follow the UMA exporter: integer ``charge``, ``spin`` and indices.
    """

    def fn(params: list[Array], x: dict[str, Array]) -> Array:
        dtype = x["pos"].dtype
        charge = x["charge"].astype(dtype)
        spin = x["spin"].astype(dtype)
        sq = jax.ops.segment_sum(
            jnp.sum(x["pos"] ** 2, axis=-1),
            x["batch"],
            num_segments=x["charge"].shape[0],
        )
        energy = 0.5 * (1.0 + charge + 10.0 * spin) * sq * params[0]
        if cell_term:
            energy = energy + (charge + spin) * jnp.trace(x["cell"], axis1=1, axis2=2)
        if leak_padding:
            energy = energy + 1e3 * (charge[-1] ** 2 + (spin[-1] - 1.0) ** 2)
        return energy

    params = [np.array(1.0, dtype=np.float64)]
    return _export(fn, params, np.float64, np.int64, np.int64)


@pytest.fixture(scope="module")
def metadata_exported() -> export.Exported:
    return _make_metadata_exported()


def _model(exported: export.Exported, task_name: Any = "omol") -> TojaxedMliap:
    return TojaxedMliap(
        cutoff=Table((SystemId(0),), jnp.array([5.0])),
        params=[jnp.array(1.0)],
        model=exported,
        task_name=task_name,
    )


def _energy_fn(model: TojaxedMliap):
    def energy(graph: HyperGraph[Any, Any, Literal[2]]) -> Table[SystemId, Array]:
        return tojaxed_energy(GraphPotentialInput(model, graph)).data

    return energy


def _energies(model: TojaxedMliap, systems: Any) -> np.ndarray:
    result = as_result_function(_energy_fn(model))(_metadata_graph(systems))
    result.raise_assertion()
    assert result.value.keys == _KEYS
    return np.asarray(result.value.data)


def _weights(charge, spin) -> np.ndarray:
    return 1.0 + np.asarray(charge) + 10.0 * np.asarray(spin)


class TestOmolSystemMetadata:
    @pytest.mark.parametrize("dtype", [jnp.int8, jnp.int32, jnp.int64])
    def test_omol_forwards_system_metadata(self, metadata_exported, dtype):
        model = _model(metadata_exported)
        systems = _meta_systems(_CHARGE.astype(dtype), _SPIN.astype(dtype))
        energies = _energies(model, systems)
        npt.assert_allclose(energies, [79.035, 55.44, 206.115], rtol=1e-12)

    @pytest.mark.parametrize(
        "systems",
        [_PlainSystems(_cell()), _MetaSystems(_cell(), None, None)],
        ids=["absent", "none"],
    )
    def test_omol_defaults_without_fields(self, metadata_exported, systems):
        energies = _energies(_model(metadata_exported), systems)
        npt.assert_allclose(energies, [79.035, 50.82, 107.965], rtol=1e-12)

    def test_omol_field_defaults_are_independent(self, metadata_exported):
        systems = _ChargeOnlySystems(_cell(), total_charge=_CHARGE)
        energies = _energies(_model(metadata_exported), systems)
        npt.assert_allclose(energies, 0.5 * _weights(_CHARGE, 1) * _SQ_SUMS)

    @pytest.mark.parametrize("task_name", [None, "omat"])
    def test_generic_ignores_system_metadata(self, metadata_exported, task_name):
        systems = _meta_systems(
            charge=jnp.array([0.5, 1.0, 0.0]), spin=jnp.array([0, 0, 0])
        )
        energies = _energies(_model(metadata_exported, task_name), systems)
        npt.assert_allclose(energies, [7.185, 4.62, 9.815], rtol=1e-12)

    @pytest.mark.parametrize(
        ("systems", "weights"),
        [
            (_PlainSystems(_cell()), _weights(0, 1)),
            (_meta_systems(), _weights(_CHARGE, _SPIN)),
        ],
        ids=["defaults", "fields"],
    )
    def test_omol_padding_row_is_neutral_singlet(self, systems, weights):
        model = _model(_make_metadata_exported(leak_padding=True))
        npt.assert_allclose(_energies(model, systems), 0.5 * weights * _SQ_SUMS)

    def test_omol_metadata_is_traced(self, metadata_exported):
        traces = []

        @jax.jit
        def energy(graph):
            traces.append(None)
            return _energy_fn(_model(metadata_exported))(graph).data

        first = energy(_metadata_graph(_meta_systems()))
        second = energy(_metadata_graph(_meta_systems(charge=jnp.array([1, 1, 0]))))
        assert len(traces) == 1
        assert not np.allclose(first, second)

    @pytest.mark.parametrize("dtype", [jnp.uint8, jnp.bool_, jnp.float64])
    @pytest.mark.parametrize("field", ["total_charge", "spin_multiplicity"])
    def test_omol_rejects_metadata_dtype(self, metadata_exported, dtype, field):
        meta = _meta_systems()
        systems = replace(meta, **{field: getattr(meta, field).astype(dtype)})
        with pytest.raises(TypeError, match=field):
            _energies(_model(metadata_exported), systems)

    @pytest.mark.parametrize("field", ["total_charge", "spin_multiplicity"])
    def test_omol_rejects_metadata_shape(self, metadata_exported, field):
        # ``Table`` checks and broadcasts the leading axis; trailing axes reach omol.
        meta = _meta_systems()
        systems = replace(meta, **{field: getattr(meta, field)[:, None]})
        with pytest.raises(ValueError, match=field):
            _energies(_model(metadata_exported), systems)

    def test_omol_rejects_out_of_bounds_under_jit(self, metadata_exported):
        traces = []

        def energy(graph):
            traces.append(None)
            return _energy_fn(_model(metadata_exported))(graph)

        compiled = jax.jit(as_result_function(energy))
        for charge, spin in (([-100, 0, 100], [1, 100, 1]), (_CHARGE, _SPIN)):
            graph = _metadata_graph(_meta_systems(jnp.array(charge), jnp.array(spin)))
            compiled(graph).raise_assertion()
        for charge, spin, reported in (
            ([0, 0, 0], [1, 0, 1], "multiplicity 0 to 1"),
            ([0, 0, 0], [1, 101, 1], "multiplicity 1 to 101"),
            ([-101, 0, 0], [1, 1, 1], "charge -101 to 0"),
            ([0, 0, 101], [1, 1, 1], "charge 0 to 101"),
        ):
            graph = _metadata_graph(_meta_systems(jnp.array(charge), jnp.array(spin)))
            with pytest.raises(ValueError, match=reported):
                compiled(graph).raise_assertion()
        assert len(traces) == 1

    def test_unknown_task_rejected_at_trace(self, metadata_exported):
        with pytest.raises(ValueError, match="omoll"):
            _energies(_model(metadata_exported, "omoll"), _meta_systems())

    def test_task_name_survives_pytree_roundtrip(self, metadata_exported):
        model = _model(metadata_exported)
        leaves, treedef = jax.tree.flatten(model)
        assert jax.tree.unflatten(treedef, leaves).task_name == "omol"
        assert jax.jit(lambda m: m)(model).task_name == "omol"


@dataclass
class _PotentialState:
    atoms: Table[ParticleId, AtomData]
    systems: Table[SystemId, _MetaSystems]
    neighborlist: AllDenseNearestNeighborList


class TestTojaxedPotentialMetadata:
    def test_potential_gradients_follow_metadata(self):
        """Metadata reaches the model through the public potential with gradients."""
        graph = _metadata_graph(_meta_systems())
        n_atoms = 8
        state = _PotentialState(
            atoms=Table.arange(
                AtomData(
                    positions=graph.particles.data.positions,
                    atomic_numbers=graph.particles.data.atomic_numbers,
                    system=graph.particles.data.system,
                    inclusion=Index.new([InclusionId(0)] * n_atoms),
                    exclusion=Index.new([ExclusionId(i) for i in range(n_atoms)]),
                ),
                label=ParticleId,
            ),
            systems=graph.systems,
            neighborlist=AllDenseNearestNeighborList(
                avg_edges=FixedCapacity(n_atoms * n_atoms),
                avg_image_candidates=FixedCapacity(n_atoms * n_atoms),
                cutoffs=Table(_KEYS, jnp.full((3,), 5.0)),
            ),
        )
        positions_and_cell = lens(
            lambda g: PositionsAndCell(
                g.particles.map_data(lambda p: p.positions), g.systems
            )
        )
        potential = make_tojaxed_potential(
            particles_view=lambda s: s.atoms,
            systems_view=lambda s: s.systems,
            neighborlist_view=lambda s: s.neighborlist,
            model=_model(_make_metadata_exported(cell_term=True)),
            gradient_lens=GRAPH_GEOMETRY.nest(positions_and_cell),
            hessian_lens=lens(lambda x: ()),
            hessian_idx_view=lambda x: (),
        )
        potential_fn = as_result_function(potential)
        state = potential_fn(state).fix_or_raise(state)
        result = potential_fn(state)
        result.raise_assertion()
        out = result.value.data

        weights = _weights(_CHARGE, _SPIN)
        q = np.asarray(_CHARGE + _SPIN, dtype=float)
        npt.assert_allclose(
            out.total_energies.data, 0.5 * weights * _SQ_SUMS + q * 60.0, rtol=1e-12
        )
        system_rows = np.asarray(state.atoms.data.system.raw_indices)
        npt.assert_allclose(
            out.gradients.positions.data,
            weights[system_rows, None] * np.asarray(state.atoms.data.positions),
            rtol=1e-12,
        )
        cell_gradients = state.systems.data.cell.frame.vectors_gradient(
            out.gradients.cell.data.frame
        )
        npt.assert_allclose(
            cell_gradients, q[:, None, None] * np.eye(3)[None], rtol=1e-12, atol=1e-12
        )
