# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ML interatomic potentials.

A universal interface mirroring [tojax][kups.potential.mliap.tojax]:
each torch MLFF backend only fills in a ``torch.nn.Module`` whose forward
consumes [AtomGraphInput][kups.potential.mliap.torch.interface.AtomGraphInput]
and returns ``{"energy", "position_gradients", "cell_gradients"}``. All graph
extraction, padding, and kUPS ``Potential`` wiring is shared.

Example:
    ```python
    from kups.application.potential.filter import POSITIONS_AND_CELL
    from kups.application.potential.mliap.torch import make_torch_mliap_from_state
    from kups.potential.mliap.torch import load_mace

    model = load_mace("mace.model", compute_cell_gradients=True)
    potential = make_torch_mliap_from_state(state_lens, gradient=POSITIONS_AND_CELL)
    ```

Requires the ``torch_dev`` dependency group: ``uv sync --group torch_dev``.
"""

from kups.potential.mliap.torch.interface import (
    AtomGraphInput,
    IsTorchMliapParticles,
    TorchMliap,
    TorchMliapForward,
    lattice_gradient_from_virial,
    make_torch_mliap_potential,
    torch_mliap_model_fn,
)
from kups.potential.mliap.torch.mace import MACEModule, load_mace
from kups.potential.mliap.torch.uma import (
    UMAInferenceSettings,
    UMAModule,
    UMATaskName,
    load_uma,
)

__all__ = [
    "AtomGraphInput",
    "IsTorchMliapParticles",
    "MACEModule",
    "TorchMliap",
    "TorchMliapForward",
    "UMAInferenceSettings",
    "UMAModule",
    "UMATaskName",
    "lattice_gradient_from_virial",
    "load_mace",
    "load_uma",
    "make_torch_mliap_potential",
    "torch_mliap_model_fn",
]
