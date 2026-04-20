# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for MACE model integration via TorchModuleWrapper.

Requires the torch_dev dependency group: `uv sync --group torch_dev`
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from kups.core.data.table import Table
from kups.core.typing import SystemId

# Skip entire module if torch not available
torch = pytest.importorskip("torch", minversion="2.0.0")


class MockMACEModel(torch.nn.Module):
    """Mock MACE model that mimics the expected interface."""

    def __init__(self, num_species: int = 10):
        super().__init__()
        self.linear = torch.nn.Linear(num_species, 1)
        self.num_species = num_species

    def forward(
        self,
        input_dict: dict,
        compute_force: bool = False,
        compute_virials: bool = False,
    ) -> dict:
        """Mock forward pass matching MACE signature."""
        node_attrs = input_dict["node_attrs"]
        positions = input_dict["positions"]
        ptr = input_dict["ptr"]
        cell = input_dict.get("cell")

        if compute_force or compute_virials:
            positions = positions.detach().requires_grad_(True)
            if cell is not None:
                cell = cell.detach().requires_grad_(True)

        per_atom_energy = self.linear(node_attrs).squeeze(-1)
        position_energy = 0.5 * (positions**2).sum(dim=-1)
        per_atom_energy = per_atom_energy + position_energy

        n_systems = len(ptr) - 1
        energies = []
        for i in range(n_systems):
            start, end = ptr[i].item(), ptr[i + 1].item()
            system_energy = per_atom_energy[start:end].sum()
            energies.append(system_energy)

        energy = torch.stack(energies)
        result = {"energy": energy}

        if compute_force:
            if energy.grad_fn is not None:
                total_energy = energy.sum()
                (grad,) = torch.autograd.grad(
                    total_energy, positions, retain_graph=compute_virials
                )
                result["forces"] = -grad
            else:
                result["forces"] = torch.zeros_like(positions)

        if compute_virials:
            result["virials"] = torch.zeros(n_systems, 3, 3)

        return result


class TestMACEModule:
    """Tests for MACEModule wrapper (energy + forces)."""

    def test_module_eval_mode(self):
        """Test MACEModule sets model to eval mode."""
        from kups.potential.mliap.torch.mace import MACEModule

        mock_mace = MockMACEModel()
        mock_mace.train()
        assert mock_mace.training

        module = MACEModule(mock_mace)
        assert not module.mace.training

    def test_module_to_device(self):
        """Test MACEModule.to() moves model to device."""
        from kups.potential.mliap.torch.mace import MACEModule

        mock_mace = MockMACEModel()
        module = MACEModule(mock_mace)

        module.to(torch.device("cpu"))
        param_device = next(module.parameters()).device
        assert param_device == torch.device("cpu")

    def test_module_forward_returns_dict(self):
        """Test MACEModule forward returns dict with energy and forces.

        Note: Uses MockMACEModel which provides a trivial energy function
        E = 0.5 * sum(positions²). This validates the interface contract
        (return type, shapes) but not numerical correctness with real MACE models.
        """
        from kups.potential.mliap.torch.mace import MACEModule

        mock_mace = MockMACEModel(num_species=5)
        module = MACEModule(mock_mace)

        n_atoms = 3
        n_types = 5
        n_edges = 6
        n_systems = 1

        node_attrs = torch.nn.functional.one_hot(
            torch.tensor([0, 1, 2]), n_types
        ).float()
        positions = torch.randn(n_atoms, 3)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        batch = torch.zeros(n_atoms, dtype=torch.long)
        ptr = torch.tensor([0, n_atoms])
        shifts = torch.zeros(n_edges, 3)

        result = module(node_attrs, positions, edge_index, batch, ptr, shifts, None)

        assert isinstance(result, dict)
        assert "energy" in result
        assert "forces" in result
        assert result["energy"].shape == (n_systems,)
        assert result["forces"].shape == (n_atoms, 3)

    def test_forces_are_negative_gradient(self):
        """Test that forces are the negative gradient of energy w.r.t. positions.

        Note: Uses MockMACEModel with trivial energy E = 0.5 * sum(positions²),
        so forces = -positions. This validates the interface contract (forces
        equal negative gradient) but not numerical correctness with real MACE models.
        """
        from kups.potential.mliap.torch.mace import MACEModule

        mock_mace = MockMACEModel(num_species=5)
        module = MACEModule(mock_mace)

        n_atoms = 3
        n_types = 5
        n_edges = 6

        node_attrs = torch.nn.functional.one_hot(
            torch.tensor([0, 1, 2]), n_types
        ).float()
        positions = torch.randn(n_atoms, 3)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        batch = torch.zeros(n_atoms, dtype=torch.long)
        ptr = torch.tensor([0, n_atoms])
        shifts = torch.zeros(n_edges, 3)

        result = module(node_attrs, positions, edge_index, batch, ptr, shifts, None)

        # For mock model with E = 0.5 * sum(positions^2), forces = -positions
        assert torch.allclose(result["forces"], -positions, atol=1e-6)

    def test_outputs_are_detached(self):
        """Test that outputs don't require gradients."""
        from kups.potential.mliap.torch.mace import MACEModule

        mock_mace = MockMACEModel(num_species=5)
        module = MACEModule(mock_mace)

        n_atoms = 3
        n_types = 5
        n_edges = 6

        node_attrs = torch.nn.functional.one_hot(
            torch.tensor([0, 1, 2]), n_types
        ).float()
        positions = torch.randn(n_atoms, 3)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        batch = torch.zeros(n_atoms, dtype=torch.long)
        ptr = torch.tensor([0, n_atoms])
        shifts = torch.zeros(n_edges, 3)

        result = module(node_attrs, positions, edge_index, batch, ptr, shifts, None)

        assert not result["energy"].requires_grad
        assert not result["forces"].requires_grad


class TestMACEEnergyOnly:
    """Tests for MACEModule with compute_force=False (energy only)."""

    def test_module_forward_returns_dict(self):
        """Test MACEModule(compute_force=False) forward returns dict with energy only."""
        from kups.potential.mliap.torch.mace import MACEModule

        mock_mace = MockMACEModel(num_species=5)
        module = MACEModule(mock_mace, compute_force=False)

        n_atoms = 3
        n_types = 5
        n_edges = 6
        n_systems = 1

        node_attrs = torch.nn.functional.one_hot(
            torch.tensor([0, 1, 2]), n_types
        ).float()
        positions = torch.randn(n_atoms, 3)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        batch = torch.zeros(n_atoms, dtype=torch.long)
        ptr = torch.tensor([0, n_atoms])
        shifts = torch.zeros(n_edges, 3)

        result = module(node_attrs, positions, edge_index, batch, ptr, shifts, None)

        assert isinstance(result, dict)
        assert "energy" in result
        assert "forces" not in result
        assert result["energy"].shape == (n_systems,)


class TestMACEWrapperCreation:
    """Tests for MACE wrapper creation."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="TorchModuleWrapper requires CUDA",
    )
    def test_wrapper_creation_cuda(self, tmp_path):
        """Test load_mace_wrapper creates a working wrapper on CUDA."""
        from kups.potential.mliap.torch import load_mace_wrapper

        mock_mace = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock_mace, model_path)

        wrapper = load_mace_wrapper(model_path, device="cuda")
        assert wrapper is not None

    def test_wrapper_creation_cpu(self, tmp_path):
        """Test load_mace_wrapper creates a wrapper (CPU loading)."""
        from kups.potential.mliap.torch import load_mace_wrapper

        mock_mace = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock_mace, model_path)

        wrapper = load_mace_wrapper(model_path, device="cpu")
        assert wrapper is not None

    def test_wrapper_creation_energy_only(self, tmp_path):
        """Test load_mace_wrapper with compute_force=False uses MACEModule."""
        from kups.potential.mliap.torch.mace import MACEModule, load_mace_wrapper

        mock_mace = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock_mace, model_path)

        wrapper = load_mace_wrapper(model_path, device="cpu", compute_force=False)
        assert isinstance(wrapper.module, MACEModule)
        assert not wrapper.module.compute_force

    def test_wrapper_creation_with_forces(self, tmp_path):
        """Test load_mace_wrapper with compute_force=True uses MACEModule."""
        from kups.potential.mliap.torch.mace import MACEModule, load_mace_wrapper

        mock_mace = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock_mace, model_path)

        wrapper = load_mace_wrapper(model_path, device="cpu", compute_force=True)
        assert isinstance(wrapper.module, MACEModule)

    def test_wrapper_output_shapes_with_forces(self, tmp_path):
        """Test wrapper correctly determines output shapes with forces."""
        from kups.potential.mliap.torch import load_mace_wrapper

        mock_mace = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock_mace, model_path)

        wrapper = load_mace_wrapper(model_path, device="cpu", compute_force=True)

        n_atoms = 3
        n_types = 5
        n_edges = 6

        node_attrs = jnp.ones((n_atoms, n_types), dtype=jnp.float32)
        positions = jnp.zeros((n_atoms, 3), dtype=jnp.float32)
        edge_index = jnp.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        batch = jnp.zeros(n_atoms, dtype=jnp.int64)
        ptr = jnp.array([0, n_atoms], dtype=jnp.int64)
        shifts = jnp.zeros((n_edges, 3), dtype=jnp.float32)

        args_flat = [node_attrs, positions, edge_index, batch, ptr, shifts]
        import jax

        from kups.core.utils.torch import ScalarSpec, _infer_spec

        _, in_tree = jax.tree.flatten((args_flat, {}))
        flat_spec = _infer_spec(args_flat)
        array_leaves = [
            a for a, s in zip(args_flat, flat_spec) if not isinstance(s, ScalarSpec)
        ]
        scalar_vals = [
            a for a, s in zip(args_flat, flat_spec) if isinstance(s, ScalarSpec)
        ]
        output_shapes, out_tree = wrapper._get_output_info(
            array_leaves, flat_spec, scalar_vals, in_tree
        )

        assert len(output_shapes) == 2
        assert output_shapes[0].shape == (1,)  # energy
        assert output_shapes[1].shape == (n_atoms, 3)  # forces

    def test_wrapper_output_shapes_energy_only(self, tmp_path):
        """Test wrapper correctly determines output shapes without forces."""
        from kups.potential.mliap.torch import load_mace_wrapper

        mock_mace = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock_mace, model_path)

        wrapper = load_mace_wrapper(model_path, device="cpu", compute_force=False)

        n_atoms = 3
        n_types = 5
        n_edges = 6

        node_attrs = jnp.ones((n_atoms, n_types), dtype=jnp.float32)
        positions = jnp.zeros((n_atoms, 3), dtype=jnp.float32)
        edge_index = jnp.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        batch = jnp.zeros(n_atoms, dtype=jnp.int64)
        ptr = jnp.array([0, n_atoms], dtype=jnp.int64)
        shifts = jnp.zeros((n_edges, 3), dtype=jnp.float32)

        args_flat = [node_attrs, positions, edge_index, batch, ptr, shifts]
        import jax

        from kups.core.utils.torch import ScalarSpec, _infer_spec

        _, in_tree = jax.tree.flatten((args_flat, {}))
        flat_spec = _infer_spec(args_flat)
        array_leaves = [
            a for a, s in zip(args_flat, flat_spec) if not isinstance(s, ScalarSpec)
        ]
        scalar_vals = [
            a for a, s in zip(args_flat, flat_spec) if isinstance(s, ScalarSpec)
        ]
        output_shapes, out_tree = wrapper._get_output_info(
            array_leaves, flat_spec, scalar_vals, in_tree
        )

        assert len(output_shapes) == 1
        assert output_shapes[0].shape == (1,)  # energy


@pytest.mark.skip(
    reason="Requires JAX FFI GPU context (run manually without pytest-xdist)"
)
class TestMACEWrapperCUDAExecution:
    """Tests for full CUDA execution."""

    def test_wrapper_forward_jax(self, tmp_path):
        """Test wrapper can compute energy and forces from JAX arrays."""
        from kups.potential.mliap.torch import load_mace_wrapper

        mock_mace = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock_mace, model_path)

        wrapper = load_mace_wrapper(model_path, device="cuda")

        n_atoms = 3
        n_types = 5
        n_edges = 6
        n_systems = 1

        species = jnp.array([0, 1, 2])
        node_attrs = jnp.eye(n_types)[species].astype(jnp.float32)
        positions = jnp.zeros((n_atoms, 3), dtype=jnp.float32)
        edge_index = jnp.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        batch = jnp.zeros(n_atoms, dtype=jnp.int64)
        ptr = jnp.array([0, n_atoms], dtype=jnp.int64)
        shifts = jnp.zeros((n_edges, 3), dtype=jnp.float32)

        result = wrapper(node_attrs, positions, edge_index, batch, ptr, shifts, None)

        assert result["energy"].shape == (n_systems,)
        assert result["forces"].shape == (n_atoms, 3)


class TestMultiSystemBatching:
    """Tests for multi-system batch handling."""

    def test_two_systems_different_sizes(self):
        """Test with 2 systems of different atom counts.

        Note: Uses MockMACEModel which provides a trivial energy function.
        This validates the batching interface (shapes, ptr handling) but not
        numerical correctness with real MACE models.
        """
        from kups.potential.mliap.torch.mace import MACEModule

        mock_mace = MockMACEModel(num_species=5)
        module = MACEModule(mock_mace)

        # System 1: 3 atoms, System 2: 5 atoms
        n_atoms = 8
        n_systems = 2
        n_types = 5

        # Create species: atoms 0-2 for system 0, atoms 3-7 for system 1
        species = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
        node_attrs = torch.nn.functional.one_hot(species, n_types).float()
        positions = torch.randn(n_atoms, 3)

        # Batch and ptr encode which atoms belong to which system
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        ptr = torch.tensor([0, 3, 8])  # boundaries

        # Simple edge list (intra-system edges only)
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 0, 4, 5, 6, 7, 3],
            ]
        )
        shifts = torch.zeros(8, 3)

        result = module(node_attrs, positions, edge_index, batch, ptr, shifts, None)

        assert result["energy"].shape == (n_systems,)
        assert result["forces"].shape == (n_atoms, 3)

    def test_periodic_cell(self):
        """Test with periodic boundary conditions.

        Note: Uses MockMACEModel which ignores cell in energy calculation.
        This validates the interface (cell passed through) but not numerical
        correctness of PBC handling.
        """
        from kups.potential.mliap.torch.mace import MACEModule

        mock_mace = MockMACEModel(num_species=5)
        module = MACEModule(mock_mace)

        n_atoms = 3
        n_types = 5
        n_systems = 1

        node_attrs = torch.nn.functional.one_hot(
            torch.tensor([0, 1, 2]), n_types
        ).float()
        positions = torch.randn(n_atoms, 3)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        batch = torch.zeros(n_atoms, dtype=torch.long)
        ptr = torch.tensor([0, n_atoms])
        shifts = torch.randn(3, 3)  # Non-zero shifts for PBC

        # 10Å cubic cell
        cell = torch.eye(3).unsqueeze(0) * 10.0

        result = module(node_attrs, positions, edge_index, batch, ptr, shifts, cell)

        assert result["energy"].shape == (n_systems,)
        assert result["forces"].shape == (n_atoms, 3)


class TestLoadMACEErrors:
    """Tests for error handling in load_mace_wrapper."""

    def test_load_nonexistent_file_raises(self):
        """Test loading a nonexistent file raises appropriate error."""
        from kups.potential.mliap.torch import load_mace_wrapper

        with pytest.raises(FileNotFoundError):
            load_mace_wrapper("/nonexistent/path/to/model.model")

    @pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="Test only runs when CUDA is NOT available",
    )
    def test_cuda_unavailable_raises_runtime_error(self, tmp_path):
        """Test requesting CUDA when unavailable raises RuntimeError."""
        from kups.potential.mliap.torch import load_mace_wrapper

        mock_mace = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock_mace, model_path)

        with pytest.raises(RuntimeError, match="CUDA is not available"):
            load_mace_wrapper(model_path, device="cuda")


class TestTorchMACEEnergyAndForces:
    """Tests for torch_mace_model_fn integration.

    These tests validate the torch_mace_model_fn function which
    is the main kUPS-facing API for PyTorch MACE integration.

    Note: Tests are skipped on CPU because TorchModuleWrapper requires
    GPU context for the JAX FFI bridge. Tests using MockMACEModel validate
    the interface contract but not numerical correctness with real MACE models.
    """

    def test_function_signature(self):
        """Test torch_mace_model_fn is importable and callable."""
        from kups.potential.mliap.torch.mace import torch_mace_model_fn

        assert callable(torch_mace_model_fn)

    def test_torch_mace_model_creation(self, tmp_path):
        """Test TorchMACEModel can be created with wrapper."""
        from kups.potential.mliap.torch import TorchMACEModel, load_mace_wrapper

        mock_mace = MockMACEModel(num_species=5)
        model_path = tmp_path / "mock_mace.model"
        torch.save(mock_mace, model_path)

        wrapper = load_mace_wrapper(model_path, device="cpu", compute_force=True)

        model = TorchMACEModel(
            species_to_index=jnp.arange(10),
            cutoff=Table((SystemId(0),), jnp.array([5.0])),
            num_mace_species=5,
            wrapper=wrapper,
        )

        assert model.num_mace_species == 5
        assert jnp.allclose(model.cutoff.data, jnp.array([5.0]))
        assert model.wrapper is wrapper
        assert model.species_to_index.shape == (10,)

    def test_make_torch_mace_potential_signature(self):
        """Test make_torch_mace_potential has expected signature."""
        from kups.potential.mliap.torch import make_torch_mace_potential

        # Function should be importable and callable
        assert callable(make_torch_mace_potential)

    def test_potential_out_and_patch_types_exist(self):
        """Test PotentialOut and related types are importable."""
        from kups.core.patch import WithPatch
        from kups.core.potential import PotentialOut

        # These types should be importable (used in return type)
        assert PotentialOut is not None
        assert WithPatch is not None
