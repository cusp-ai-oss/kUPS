# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MACE adapter on the universal torch MLFF interface.

Requires the torch_dev dependency group: `uv sync --group torch_dev`.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

# Skip entire module if torch not available
torch = pytest.importorskip("torch", minversion="2.0.0")


class MockMACEModel(torch.nn.Module):
    """Mock MACE model with the minimal attribute surface used by ``load_mace``.

    Implements a trivial energy ``E = ½ Σ |r|²`` so that ``forces = -r``.
    """

    def __init__(
        self,
        num_species: int = 5,
        atomic_numbers: list[int] | None = None,
        r_max: float = 5.0,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(num_species, 1)
        self.num_species = num_species
        if atomic_numbers is None:
            atomic_numbers = list(range(1, num_species + 1))
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max))

    def forward(
        self,
        input_dict: dict,
        compute_force: bool = False,
        compute_virials: bool = False,
    ) -> dict:
        node_attrs = input_dict["node_attrs"]
        positions = input_dict["positions"]
        ptr = input_dict["ptr"]

        if compute_force or compute_virials:
            positions = positions.detach().requires_grad_(True)

        per_atom_energy = self.linear(node_attrs).squeeze(-1)
        per_atom_energy = per_atom_energy + 0.5 * (positions**2).sum(dim=-1)

        n_systems = len(ptr) - 1
        energies = torch.stack(
            [
                per_atom_energy[ptr[i].item() : ptr[i + 1].item()].sum()
                for i in range(n_systems)
            ]
        )
        result: dict = {"energy": energies}

        if compute_force:
            if energies.grad_fn is not None:
                (grad,) = torch.autograd.grad(
                    energies.sum(), positions, retain_graph=compute_virials
                )
                result["forces"] = -grad
            else:
                result["forces"] = torch.zeros_like(positions)
        if compute_virials:
            # Consistent with ``E = linear(node_attrs) + 0.5·Σ|pos|²``:
            # only the position² term contributes to the symmetric strain.
            # Under the perturbation r → r·(I+ε), the contribution to
            # ∂E/∂ε is Σ_b pos_b ⊗ pos_b per system. MACE convention
            # returns ``-virials`` so that virials = -sym(pos_virial+cell_virial).
            pos_outer = positions.unsqueeze(2) * positions.unsqueeze(1)
            virials = torch.stack(
                [
                    pos_outer[ptr[i].item() : ptr[i + 1].item()].sum(dim=0)
                    for i in range(n_systems)
                ]
            )
            result["virials"] = -virials
        return result


def _atom_graph_input(
    n_atoms: int = 3,
    n_systems: int = 1,
    n_edges: int = 6,
) -> dict:
    """Build a synthetic universal AtomGraphInput dict (padded by +1 atom/sys)."""
    species = torch.tensor([1, 2, 3] + [0] * (n_atoms - 3 + 1), dtype=torch.int64)
    positions = torch.randn(n_atoms + 1, 3)
    cell = torch.eye(3).unsqueeze(0).repeat(n_systems + 1, 1, 1) * 10.0
    batch = torch.zeros(n_atoms + 1, dtype=torch.int64)
    batch[-1] = n_systems  # padding row goes into the padding system
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    cell_offsets = torch.zeros(n_edges, 3, dtype=torch.int64)
    return {
        "pos": positions,
        "atomic_numbers": species,
        "cell": cell,
        "pbc": torch.ones(n_systems + 1, 3, dtype=torch.bool),
        "edge_index": edge_index,
        "cell_offsets": cell_offsets,
        "batch": batch,
        "charge": torch.zeros(n_systems + 1),
        "spin": torch.zeros(n_systems + 1),
    }


class TestMACEModule:
    """Adapter behaviour: AtomGraphInput → MACE → energy + gradients."""

    def _make(self, compute_cell_gradients: bool = False):
        from kups.potential.mliap.torch import MACEModule

        mock = MockMACEModel(num_species=5)
        species_to_index = torch.zeros(6, dtype=torch.int64)
        species_to_index[mock.atomic_numbers] = torch.arange(5, dtype=torch.int64)
        return MACEModule(
            mock,
            species_to_index=species_to_index,
            num_species=5,
            compute_cell_gradients=compute_cell_gradients,
        )

    def test_forward_returns_energy_and_position_gradients(self):
        module = self._make()
        result = module(_atom_graph_input())
        assert set(result) == {"energy", "position_gradients"}
        assert result["energy"].shape == (2,)  # 1 real + 1 padding
        assert result["position_gradients"].shape == (4, 3)  # 3 real + 1 padding

    def test_forward_with_cell_gradients(self):
        module = self._make(compute_cell_gradients=True)
        result = module(_atom_graph_input())
        assert set(result) == {"energy", "position_gradients", "cell_gradients"}
        assert result["cell_gradients"].shape == (2, 3, 3)

    def test_cell_gradients_zero_for_position_only_energy(self):
        """E = ½·Σ|r|² has no cell dependence ⇒ ∂E/∂h must be 0.

        Validates that the virial→lattice-gradient inversion strips out the
        position contribution rather than passing MACE's virials through raw.
        """
        module = self._make(compute_cell_gradients=True)
        result = module(_atom_graph_input())
        assert torch.allclose(
            result["cell_gradients"],
            torch.zeros_like(result["cell_gradients"]),
            atol=1e-5,
        )

    def test_position_gradients_match_negative_force(self):
        """E = ½ Σ |r|² ⇒ position_gradient = ∂E/∂r = r."""
        module = self._make()
        inp = _atom_graph_input()
        result = module(inp)
        assert torch.allclose(result["position_gradients"], inp["pos"], atol=1e-6)

    def test_outputs_are_detached(self):
        module = self._make()
        result = module(_atom_graph_input())
        assert not result["energy"].requires_grad
        assert not result["position_gradients"].requires_grad

    def test_module_eval_mode(self):
        from kups.potential.mliap.torch import MACEModule

        mock = MockMACEModel(num_species=5)
        mock.train()
        assert mock.training
        module = MACEModule(mock, species_to_index=torch.arange(6), num_species=5)
        assert not module.mace.training


class TestLoadMACE:
    """`load_mace` returns a fully-populated container."""

    def test_load_returns_torch_mliap(self, tmp_path):
        from kups.potential.mliap.torch import MACEModule, TorchMliap, load_mace

        mock = MockMACEModel(num_species=5, r_max=4.5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock, model_path)

        mliap = load_mace(model_path, device="cpu", compute_cell_gradients=True)
        assert isinstance(mliap, TorchMliap)
        assert isinstance(mliap.wrapper.module, MACEModule)
        assert mliap.compute_cell_gradients
        assert float(mliap.cutoff.data[0]) == pytest.approx(4.5)

    def test_load_overrides_cutoff(self, tmp_path):
        from kups.potential.mliap.torch import load_mace

        mock = MockMACEModel(num_species=5, r_max=4.5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock, model_path)

        mliap = load_mace(model_path, device="cpu", cutoff=6.0)
        assert float(mliap.cutoff.data[0]) == pytest.approx(6.0)

    def test_load_nonexistent_file_raises(self):
        from kups.potential.mliap.torch import load_mace

        with pytest.raises(FileNotFoundError):
            load_mace("/nonexistent/path/to/model.model", device="cpu")

    @pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="Only runs when CUDA is not available",
    )
    def test_cuda_unavailable_raises_runtime_error(self, tmp_path):
        from kups.potential.mliap.torch import load_mace

        mock = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock, model_path)
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            load_mace(model_path, device="cuda")


class TestLatticeGradientFromVirial:
    """Direct numerical checks on the virial→∂E/∂h inversion."""

    def test_position_only_energy_recovers_zero(self):
        """E = ½·Σ|r|² has no cell dependence ⇒ ∂E/∂h = 0."""
        from kups.potential.mliap.torch.interface import (
            lattice_gradient_from_virial,
        )

        pos = torch.randn(5, 3, dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1) * 10.0
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)
        forces = -pos  # g_r = pos
        # Analytic virial = sym(pos_virial), cell_virial = 0:
        pos_virial_per_atom = pos.unsqueeze(2) * pos.unsqueeze(1)
        virial = torch.zeros(2, 3, 3, dtype=torch.float64).index_add(
            0, batch, pos_virial_per_atom
        )
        cell_grad = lattice_gradient_from_virial(forces, pos, batch, cell, virial)
        assert torch.allclose(cell_grad, torch.zeros_like(cell_grad), atol=1e-10)

    def test_cell_only_energy_recovers_analytic(self):
        """E = Σ_s Σ_ij h_ij². g_h = 2·h. Verify recovery matches."""
        from kups.potential.mliap.torch.interface import (
            lattice_gradient_from_virial,
        )

        torch.manual_seed(0)
        pos = torch.randn(5, 3, dtype=torch.float64)
        # Use random invertible cells
        cell = torch.randn(2, 3, 3, dtype=torch.float64) + 3 * torch.eye(
            3, dtype=torch.float64
        )
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)
        forces = torch.zeros_like(pos)  # no pos dependence
        g_h = 2 * cell
        # cell^T @ g_h = 2·cell^T·cell, which is symmetric ⇒ recovery is exact
        cell_virial = cell.transpose(-1, -2) @ g_h
        virial = 0.5 * (cell_virial + cell_virial.transpose(-1, -2))

        cell_grad = lattice_gradient_from_virial(forces, pos, batch, cell, virial)
        assert torch.allclose(cell_grad, g_h, atol=1e-10)


class TestProjectGradOntoFrame:
    """``∂E/∂h`` matrix → frame-parameter-space projection used for cell gradients."""

    def test_preserves_triclinic_frame_and_projects_lower_triangle(self):
        from kups.core.cell import PeriodicCell, TriclinicFrame
        from kups.potential.mliap.torch.interface import _project_grad_onto_frame

        cell = PeriodicCell(
            TriclinicFrame.from_matrix(
                jnp.array([[2.0, 0, 0], [0.5, 3.0, 0], [0.1, 0.2, 4.0]])
            )
        )
        cell_grad = jnp.arange(9.0).reshape(3, 3)
        out = _project_grad_onto_frame(cell, cell_grad)

        # Frame type (not a vjp cotangent tuple) must survive for downstream
        # ``gradients.cell.data.vectors`` consumers.
        assert isinstance(out, PeriodicCell)
        assert isinstance(out.frame, TriclinicFrame)
        assert jnp.allclose(out.frame.tril, jnp.array([0.0, 3.0, 4.0, 6.0, 7.0, 8.0]))

    def test_preserves_orthogonal_frame_and_projects_diagonal(self):
        from kups.core.cell import OrthogonalFrame, PeriodicCell
        from kups.potential.mliap.torch.interface import _project_grad_onto_frame

        cell = PeriodicCell(OrthogonalFrame(jnp.array([2.0, 3.0, 4.0])))
        cell_grad = jnp.arange(9.0).reshape(3, 3)
        out = _project_grad_onto_frame(cell, cell_grad)

        assert isinstance(out.frame, OrthogonalFrame)
        assert jnp.allclose(out.frame.lengths, jnp.array([0.0, 4.0, 8.0]))


class TestUniversalInterfaceAPI:
    """Smoke checks for the universal interface surface."""

    def test_imports(self):
        from kups.potential.mliap.torch import (
            AtomGraphInput,
            TorchMliap,
            lattice_gradient_from_virial,
            make_torch_mliap_from_state,
            make_torch_mliap_potential,
            torch_mliap_model_fn,
        )

        assert AtomGraphInput is not None
        assert TorchMliap is not None
        assert callable(lattice_gradient_from_virial)
        assert callable(make_torch_mliap_from_state)
        assert callable(make_torch_mliap_potential)
        assert callable(torch_mliap_model_fn)

    def test_torch_mliap_from_module(self):
        from kups.potential.mliap.torch import MACEModule, TorchMliap

        mock = MockMACEModel(num_species=5)
        module = MACEModule(mock, species_to_index=torch.arange(6), num_species=5)
        mliap = TorchMliap.from_module(module, cutoff=5.0)
        assert isinstance(mliap, TorchMliap)
        assert float(mliap.cutoff.data[0]) == pytest.approx(5.0)
        assert mliap.wrapper.requires_grad
        assert jnp.allclose(mliap.cutoff.data, jnp.array([5.0]))
