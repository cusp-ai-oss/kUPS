# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UMA adapter on the universal torch MLFF interface.

UMA requires fairchem-core>=2.0 which is not installable on Python 3.14
(fairchem v2 caps at Python 3.13). The forward-pass test stubs
``fairchem.core.datasets.atomic_data`` via ``sys.modules`` so the adapter's
behaviour can be exercised without the real dependency.

Requires the torch_dev dependency group: `uv sync --group torch_dev`.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch", minversion="2.0.0")

from kups.application.relaxation.data import RelaxSystems  # noqa: E402
from kups.core.cell import Cell, TriclinicFrame  # noqa: E402
from kups.core.data import Index, Table  # noqa: E402
from kups.core.neighborlist import Edges  # noqa: E402
from kups.core.typing import ParticleId, SystemId  # noqa: E402
from kups.core.utils.jax import dataclass, no_post_init, tree_zeros_like  # noqa: E402
from kups.potential.common.graph import GraphPotentialInput, HyperGraph  # noqa: E402
from kups.potential.mliap.torch.interface import (  # noqa: E402
    TorchMliap,
    torch_mliap_model_fn,
)
from kups.potential.mliap.torch.uma import UMAModule  # noqa: E402


def _atom_graph_input(n_real_atoms: int = 3, n_real_systems: int = 1):
    """Build a universal AtomGraphInput dict (padded by +1 atom and +1 system)."""
    n_atoms = n_real_atoms + 1
    n_sys = n_real_systems + 1
    species = torch.tensor(
        [1, 2, 3][:n_real_atoms] + [0] * (n_atoms - n_real_atoms),
        dtype=torch.int64,
    )
    positions = torch.randn(n_atoms, 3)
    cell = torch.eye(3).unsqueeze(0).repeat(n_sys, 1, 1) * 10.0
    batch = torch.zeros(n_atoms, dtype=torch.int64)
    batch[-1] = n_real_systems
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    cell_offsets = torch.zeros(6, 3, dtype=torch.int64)
    return {
        "pos": positions,
        "atomic_numbers": species,
        "cell": cell,
        "pbc": torch.ones(n_sys, 3, dtype=torch.bool),
        "edge_index": edge_index,
        "cell_offsets": cell_offsets,
        "batch": batch,
        "charge": torch.zeros(n_sys),
        "spin": torch.zeros(n_sys),
    }


class _FakeAtomicData(dict):
    """Minimal stand-in for ``fairchem.core.datasets.atomic_data.AtomicData``."""

    def __init__(self, **kwargs):
        super().__init__()
        self.update(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def stub_fairchem(monkeypatch):
    """Install minimal fairchem stub modules in sys.modules.

    Lets UMAModule.forward import ``AtomicData`` without the real fairchem
    being importable in the dev env (Py3.14 has no fairchem-core>=2 wheels).
    """
    fc = ModuleType("fairchem")
    fc_core = ModuleType("fairchem.core")
    fc_ds = ModuleType("fairchem.core.datasets")
    fc_ad = ModuleType("fairchem.core.datasets.atomic_data")
    fc_ad.AtomicData = _FakeAtomicData  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "fairchem", fc)
    monkeypatch.setitem(sys.modules, "fairchem.core", fc_core)
    monkeypatch.setitem(sys.modules, "fairchem.core.datasets", fc_ds)
    monkeypatch.setitem(sys.modules, "fairchem.core.datasets.atomic_data", fc_ad)


class _FakePredictUnit:
    """Mock UMA predict-unit returning zero energy/forces/stress of the right shape."""

    def __init__(self, consistent_with_pos_squared_energy: bool = False):
        """Either a trivial mock (zeros), or one that emulates UMA's outputs
        for ``E = ½·Σ|r|²`` (forces, stress consistent so that the recovered
        ``∂E/∂h`` is exactly 0).
        """
        self.last_data = None
        self.consistent = consistent_with_pos_squared_energy
        self.device = "cpu"  # matches fairchem.MLIPPredictUnit's str-typed device
        # mimic fairchem's predict-unit settings — UMAModule reads
        # ``base_precision_dtype`` to cast inputs.
        self.inference_settings = type(
            "_S", (), {"base_precision_dtype": torch.float32}
        )()

    def predict(self, data, undo_element_references: bool = True):
        self.last_data = data
        pos = data["pos"]
        cell = data["cell"]
        batch = data["batch"]
        n_atoms = pos.shape[0]
        n_sys = cell.shape[0]
        if not self.consistent:
            return {
                "energy": torch.zeros(n_sys),
                "forces": torch.ones(n_atoms, 3),
                "stress": torch.zeros(n_sys, 3, 3),
            }
        # Consistent with E = ½·Σ|r|² (no cell term):
        #   forces = -∂E/∂r = -r
        #   virial = sym(g_r ⊗ r) summed per system = sym(r ⊗ r)
        #   stress = virial / V
        forces = -pos
        g_r = pos
        pos_virial_per_atom = g_r.unsqueeze(2) * pos.unsqueeze(1)
        pos_virial = pos.new_zeros(n_sys, 3, 3).index_add(0, batch, pos_virial_per_atom)
        sym_pos_virial = 0.5 * (pos_virial + pos_virial.transpose(-1, -2))
        volume = torch.linalg.det(cell).abs().view(-1, 1, 1)
        stress = sym_pos_virial / volume
        pos_sq = 0.5 * (pos**2).sum(dim=-1)
        energy = pos.new_zeros(n_sys).index_add(0, batch, pos_sq)
        return {"energy": energy, "forces": forces, "stress": stress}


class TestUMAModule:
    """UMAModule wiring around a mock predict-unit (fairchem stubbed)."""

    def test_forward_returns_energy_and_position_gradients(self, stub_fairchem):
        from kups.potential.mliap.torch import UMAModule

        module = UMAModule(_FakePredictUnit(), task_name="omat")
        result = module(_atom_graph_input())
        assert set(result) == {"energy", "position_gradients"}
        assert result["energy"].shape == (2,)
        assert result["position_gradients"].shape == (4, 3)
        # forces=1 ⇒ position_gradients=-1
        assert torch.allclose(result["position_gradients"], -torch.ones(4, 3))

    def test_forward_with_cell_gradients(self, stub_fairchem):
        from kups.potential.mliap.torch import UMAModule

        module = UMAModule(
            _FakePredictUnit(), task_name="omat", compute_cell_gradients=True
        )
        result = module(_atom_graph_input())
        assert set(result) == {"energy", "position_gradients", "cell_gradients"}
        assert result["cell_gradients"].shape == (2, 3, 3)

    def test_cell_gradients_zero_for_position_only_energy(self, stub_fairchem):
        """Stress emulating ``E = ½·Σ|r|²`` ⇒ recovered ``∂E/∂h`` is exactly 0."""
        from kups.potential.mliap.torch import UMAModule

        module = UMAModule(
            _FakePredictUnit(consistent_with_pos_squared_energy=True),
            task_name="omat",
            compute_cell_gradients=True,
        )
        result = module(_atom_graph_input())
        assert torch.allclose(
            result["cell_gradients"],
            torch.zeros_like(result["cell_gradients"]),
            atol=1e-5,
        )

    def test_atomic_data_fields(self, stub_fairchem):
        from kups.potential.mliap.torch import UMAModule

        pu = _FakePredictUnit()
        module = UMAModule(pu, task_name="omol")
        inputs = _atom_graph_input()
        inputs["charge"] = torch.zeros(2, dtype=torch.int64)
        inputs["spin"] = torch.ones(2, dtype=torch.int64)
        module(inputs)

        data = pu.last_data
        assert data is not None
        # Required AtomicData fields propagated
        for k in [
            "pos",
            "atomic_numbers",
            "cell",
            "pbc",
            "natoms",
            "edge_index",
            "cell_offsets",
            "nedges",
            "charge",
            "spin",
            "fixed",
            "tags",
            "batch",
            "sid",
            "dataset",
        ]:
            assert k in data, f"missing {k} on AtomicData"
        assert data["dataset"] == ["omol", "omol"]
        assert data["sid"] == ["", ""]
        # natoms = 3 real + 1 padding
        assert data["natoms"].tolist() == [3, 1]
        # nedges: 6 real edges, all on system 0 (real)
        assert data["nedges"].tolist() == [6, 0]
        # Last atom is pinned to the last (padding) system so that
        # ``batch.max() + 1 == n_sys`` even under mock-zero shape inference.
        assert int(data["batch"][-1]) == 1

    def test_outputs_are_detached(self, stub_fairchem):
        from kups.potential.mliap.torch import UMAModule

        module = UMAModule(_FakePredictUnit(), task_name="omat")
        result = module(_atom_graph_input())
        assert not result["energy"].requires_grad
        assert not result["position_gradients"].requires_grad


class TestLoadUMA:
    """``load_uma`` error paths."""

    def test_load_nonexistent_file_raises(self):
        from kups.potential.mliap.torch import load_uma

        with pytest.raises(FileNotFoundError):
            load_uma("/nonexistent/path/uma.pt", device="cpu")

    def test_load_without_fairchem_raises_import_error(self, tmp_path):
        """When fairchem-core>=2 is unavailable, surfaces a clear ImportError."""
        from kups.potential.mliap.torch import load_uma

        # Force the lazy import of fairchem.core.units.mlip_unit to fail by
        # stubbing it with a module that does not expose load_predict_unit.
        bad = ModuleType("fairchem.core.units.mlip_unit")
        sys.modules["fairchem.core.units.mlip_unit"] = bad
        try:
            model_path = tmp_path / "fake.pt"
            model_path.touch()
            with pytest.raises(ImportError, match="fairchem-core"):
                load_uma(model_path, device="cpu")
        finally:
            sys.modules.pop("fairchem.core.units.mlip_unit", None)

    @pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="Only runs when CUDA is not available",
    )
    def test_cuda_unavailable_raises_runtime_error(self, tmp_path):
        from kups.potential.mliap.torch import load_uma

        model_path = tmp_path / "fake.pt"
        model_path.touch()
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            load_uma(model_path, device="cuda")


@dataclass
class _GraphAtoms:
    positions: jax.Array
    atomic_numbers: jax.Array
    system: Index[SystemId]


@dataclass
class _LegacySystems:
    cell: Cell


def _metadata_graph(legacy=False):
    """Build interleaved atoms with unequal counts and noncanonical system keys."""
    keys = (SystemId(12), SystemId(37), SystemId(91))
    cell = Cell.from_pbc(
        TriclinicFrame.from_matrix(jnp.tile(jnp.eye(3)[None] * 20, (3, 1, 1))),
        (False, False, False),
    )
    atoms = Table.arange(
        _GraphAtoms(
            jnp.arange(24, dtype=float).reshape(8, 3) / 10,
            jnp.ones(8, dtype=int),
            Index(keys, jnp.array([1, 0, 1, 0, 2, 1, 0, 2])),
        ),
        label=ParticleId,
    )
    data = (
        _LegacySystems(cell)
        if legacy
        else RelaxSystems(
            cell,
            tree_zeros_like(cell),
            jnp.zeros(3),
            total_charge=jnp.array([0, 1, 0]),
            spin_multiplicity=jnp.array([1, 1, 2]),
        )
    )
    edges = Edges(
        Index(atoms.keys, jnp.array([[0, 2], [1, 3], [4, 7]])),
        jnp.zeros((3, 1, 3)),
    )
    return HyperGraph(atoms, Table(keys, data), edges)


class _CPUTransport:
    """Replace only CUDA transport, retaining the selected builder and torch model."""

    def __init__(self, module, requires_grad=True):
        self.module = module
        self.inputs = None

    def __call__(self, inputs):
        from kups.core.utils.torch import _jax_to_torch_dtype

        self.inputs = inputs
        tensors = {
            key: torch.as_tensor(
                np.array(value), dtype=_jax_to_torch_dtype(value.dtype)
            )
            for key, value in inputs.items()
        }
        return {
            key: jnp.asarray(value.detach().numpy())
            for key, value in self.module(tensors).items()
        }


class _MetadataPredictUnit(_FakePredictUnit):
    """Use a quadratic energy whose coefficient depends on charge and spin."""

    def predict(self, data, undo_element_references=True):
        self.last_data = data
        pos, batch = data["pos"], data["batch"]
        n_sys = data["cell"].shape[0]
        factor = 2 * data["charge"] + data["spin"]
        gradients = factor[batch, None] * pos
        energy = 10 * data["charge"] + data["spin"]
        energy = energy.index_add(0, batch, 0.5 * (pos * gradients).sum(-1))
        virial = pos.new_zeros(n_sys, 3, 3).index_add(
            0, batch, gradients[:, :, None] * pos[:, None, :]
        )
        volume = torch.linalg.det(data["cell"]).abs()[:, None, None]
        return {"energy": energy, "forces": -gradients, "stress": virial / volume}


@pytest.fixture
def cpu_transport(monkeypatch):
    """Keep the production graph path executable without CUDA."""
    monkeypatch.setattr(
        "kups.potential.mliap.torch.interface.TorchModuleWrapper", _CPUTransport
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_omol_graph_to_output_metadata_and_unsorting(
    stub_fairchem, cpu_transport, legacy
):
    """Selected metadata changes real outputs while keys and atom order survive."""
    graph = _metadata_graph(legacy)
    predictor = _MetadataPredictUnit()
    module = UMAModule(predictor, "omol", compute_cell_gradients=True)
    model = TorchMliap.from_module(
        module, 6, True, input_builder=module._prepare_inputs
    )
    output = torch_mliap_model_fn(GraphPotentialInput(model, graph)).data
    charge = [0, 0, 0] if legacy else [0, 1, 0]
    spin = [1, 1, 1] if legacy else [1, 1, 2]
    assert predictor.last_data["charge"].tolist() == charge
    assert predictor.last_data["spin"].tolist() == spin
    assert output.total_energies.keys == graph.systems.keys
    assert output.gradients.cell.keys == graph.systems.keys
    assert output.gradients.positions.keys == graph.particles.keys
    # Hand-derived sums of squared coordinates by system: 14.37, 9.24, 19.63.
    expected_energy = [8.185, 5.62, 10.815] if legacy else [8.185, 24.86, 21.63]
    np.testing.assert_allclose(output.total_energies.data, expected_energy, rtol=1e-6)
    factors = np.ones(8) if legacy else np.array([3, 1, 3, 1, 2, 3, 1, 2])
    np.testing.assert_allclose(
        output.gradients.positions.data,
        np.asarray(graph.particles.data.positions) * factors[:, None],
        rtol=1e-6,
    )
    np.testing.assert_allclose(output.gradients.cell.data.vectors, 0, atol=1e-6)


@pytest.mark.parametrize("task", [None, "omat", "odac", "oc20", "omc"])
def test_generic_and_nonmolecular_zero_conditioning(stub_fairchem, cpu_transport, task):
    """Generic and non-omol callers retain floating zeros despite stored metadata."""
    graph = _metadata_graph()
    predictor = _MetadataPredictUnit()
    module = UMAModule(predictor, task or "omat", compute_cell_gradients=True)
    builder = None if task is None else module._prepare_inputs
    model = TorchMliap.from_module(module, 6, True, input_builder=builder)
    result = torch_mliap_model_fn(GraphPotentialInput(model, graph)).data
    for field in ("charge", "spin"):
        values = model.wrapper.inputs[field]
        assert jnp.issubdtype(values.dtype, jnp.floating)
        np.testing.assert_array_equal(values, [0, 0, 0])
    assert result.total_energies.keys == graph.systems.keys
    np.testing.assert_array_equal(result.total_energies.data, [0, 0, 0])


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("total_charge", jnp.array([True, False, False]), TypeError),
        ("total_charge", jnp.array([0.0, 1.0, 0.0]), TypeError),
        ("spin_multiplicity", jnp.array([1.0, 1.0, 2.0]), TypeError),
        ("total_charge", jnp.array(1), ValueError),
        ("spin_multiplicity", jnp.array([[1, 1, 2]]), ValueError),
    ],
)
def test_direct_system_metadata_dtype_and_shape(field, value, error):
    """The omol input boundary rejects metadata that bypassed public builders."""
    graph = _metadata_graph()
    with no_post_init():
        graph = replace(
            graph,
            systems=graph.systems.set_data(
                replace(graph.systems.data, **{field: value})
            ),
        )
    with pytest.raises(error, match=field):
        UMAModule(_FakePredictUnit(), "omol")._prepare_inputs(graph)


@pytest.mark.parametrize(
    "field,value", [("charge", -101), ("charge", 101), ("spin", 0), ("spin", 101)]
)
def test_omol_values_checked_before_predict(stub_fairchem, field, value):
    """Invalid real omol conditioning never reaches the predictor."""
    inputs = _atom_graph_input()
    inputs["charge"] = torch.zeros(2, dtype=torch.int64)
    inputs["spin"] = torch.ones(2, dtype=torch.int64)
    inputs[field][0] = value
    predictor = _FakePredictUnit()
    with pytest.raises(ValueError, match=field):
        UMAModule(predictor, "omol")(inputs)
    assert predictor.last_data is None


def test_omol_mock_zero_shape_probe_bypasses_validation(stub_fairchem):
    """All-zero shape probes return finite shapes without initializing UMA."""
    inputs = _atom_graph_input()
    inputs["atomic_numbers"].zero_()
    predictor = _FakePredictUnit()
    output = UMAModule(predictor, "omol", compute_cell_gradients=True)(inputs)
    assert predictor.last_data is None
    assert output["energy"].shape == (2,)
    assert output["cell_gradients"].shape == (2, 3, 3)
    assert all(torch.isfinite(value).all() for value in output.values())


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.complex64])
def test_real_omol_requires_integer_metadata_before_cast(stub_fairchem, dtype):
    """Model precision casts cannot hide invalid real molecular metadata dtypes."""
    inputs = _atom_graph_input()
    inputs["charge"] = torch.zeros(2, dtype=dtype)
    inputs["spin"] = torch.ones(2, dtype=torch.int64)
    with pytest.raises(TypeError, match="charge"):
        UMAModule(_FakePredictUnit(), "omol")(inputs)


def test_omol_inclusive_bounds_and_model_precision(stub_fairchem):
    """Integer metadata at FAIRChem's limits is cast to model precision afterward."""
    inputs = _atom_graph_input()
    inputs["charge"] = torch.tensor([-100, 100])
    inputs["spin"] = torch.tensor([1, 100])
    predictor = _FakePredictUnit()
    UMAModule(predictor, "omol")(inputs)
    assert predictor.last_data["charge"].tolist() == [-100, 100]
    assert predictor.last_data["spin"].tolist() == [1, 100]
    assert predictor.last_data["charge"].dtype == torch.float32


def test_omol_preparation_is_jittable():
    """Preparation keeps integer metadata as dynamic leaves without host conversion."""
    graph = _metadata_graph().sorted_by_system()
    prepare = jax.jit(UMAModule(_FakePredictUnit(), "omol")._prepare_inputs)
    inputs = prepare(graph)
    np.testing.assert_array_equal(inputs["charge"], [0, 1, 0])
    np.testing.assert_array_equal(inputs["spin"], [1, 1, 2])
    assert inputs["cell"].shape[0] == 3


@pytest.mark.parametrize("dtype", [jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64])
def test_unsigned_metadata_survives_transport(stub_fairchem, cpu_transport, dtype):
    """Valid unsigned batch metadata reaches UMA through supported transport dtypes."""
    graph = _metadata_graph()
    graph = replace(
        graph,
        systems=graph.systems.set_data(
            replace(
                graph.systems.data,
                total_charge=jnp.array([0, 1, 0], dtype=dtype),
                spin_multiplicity=jnp.array([1, 1, 2], dtype=dtype),
            )
        ),
    )
    module = UMAModule(_MetadataPredictUnit(), "omol", compute_cell_gradients=True)
    model = TorchMliap.from_module(
        module, 6, True, input_builder=module._prepare_inputs
    )
    output = torch_mliap_model_fn(GraphPotentialInput(model, graph)).data
    np.testing.assert_allclose(
        output.total_energies.data, [8.185, 24.86, 21.63], rtol=1e-6
    )


@pytest.mark.parametrize("field", ["total_charge", "spin_multiplicity"])
def test_unsigned_overflow_cannot_wrap_into_valid_metadata(
    stub_fairchem, cpu_transport, field
):
    """Large uint64 values are rejected before prediction instead of wrapping."""
    graph = _metadata_graph()
    value = jnp.array([np.iinfo(np.uint64).max, 1, 1], dtype=jnp.uint64)
    graph = replace(
        graph,
        systems=graph.systems.set_data(replace(graph.systems.data, **{field: value})),
    )
    predictor = _FakePredictUnit()
    module = UMAModule(predictor, "omol", compute_cell_gradients=True)
    model = TorchMliap.from_module(
        module, 6, True, input_builder=module._prepare_inputs
    )
    with pytest.raises(ValueError, match="must be in"):
        torch_mliap_model_fn(GraphPotentialInput(model, graph))
    assert predictor.last_data is None


def test_direct_unsigned_tensor_bounds(stub_fairchem):
    """A negative charge bound does not wrap when direct tensors are unsigned."""
    inputs = _atom_graph_input()
    inputs["charge"] = torch.tensor([0, 100], dtype=torch.uint8)
    inputs["spin"] = torch.tensor([1, 100], dtype=torch.uint8)
    predictor = _FakePredictUnit()
    UMAModule(predictor, "omol")(inputs)
    assert predictor.last_data["charge"].tolist() == [0, 100]


def test_load_uma_binds_task_input_builder(
    stub_fairchem, cpu_transport, tmp_path, monkeypatch
):
    """The public loader selects omol preparation without a second task field."""
    from dataclasses import dataclass as plain_dataclass

    from kups.potential.mliap.torch import load_uma

    @plain_dataclass
    class Settings:
        external_graph_gen: bool = False

    predictor = _MetadataPredictUnit()
    unit = ModuleType("fairchem.core.units.mlip_unit")
    unit.load_predict_unit = lambda **_: predictor
    inference = ModuleType("fairchem.core.units.mlip_unit.api.inference")
    inference.InferenceSettings = lambda **kw: Settings()
    inference.guess_inference_settings = lambda _: Settings()
    monkeypatch.setitem(sys.modules, unit.__name__, unit)
    monkeypatch.setitem(sys.modules, inference.__name__, inference)
    path = tmp_path / "fake.pt"
    path.touch()
    model = load_uma(path, device="cpu", task_name="omol", compute_cell_gradients=True)
    output = torch_mliap_model_fn(GraphPotentialInput(model, _metadata_graph())).data
    np.testing.assert_allclose(
        output.total_energies.data, [8.185, 24.86, 21.63], rtol=1e-6
    )
