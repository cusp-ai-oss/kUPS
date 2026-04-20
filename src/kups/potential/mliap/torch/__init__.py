# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model bridge for ML interatomic potentials.

This module provides a bridge for using PyTorch-based ML potentials from JAX
via TorchModuleWrapper.

Example:
    ```python
    from kups.potential.mliap.torch import load_mace_wrapper

    # Load MACE model with forces
    wrapper = load_mace_wrapper("model.model")
    result = wrapper(node_attrs, positions, edge_index, batch, ptr, shifts, cell)
    energy, forces = result["energy"], result["forces"]

    # Energy only (faster for MC)
    wrapper = load_mace_wrapper("model.model", compute_force=False)
    result = wrapper(...)
    energy = result["energy"]
    ```

Requires the `torch_dev` dependency group: `uv sync --group torch_dev`
"""

from kups.potential.mliap.torch.mace import (
    TorchMACEModel,
    load_mace_wrapper,
    make_torch_mace_potential,
)

__all__ = [
    "TorchMACEModel",
    "load_mace_wrapper",
    "make_torch_mace_potential",
]
