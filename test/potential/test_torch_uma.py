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
from types import ModuleType

import pytest

torch = pytest.importorskip("torch", minversion="2.0.0")


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
        module(_atom_graph_input())

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
