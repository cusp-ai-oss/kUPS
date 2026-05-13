# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MACE adapter for the universal torch MLFF interface.

Provides a thin ``torch.nn.Module`` ([MACEModule][kups.potential.mliap.torch.mace.MACEModule])
that translates the universal
[AtomGraphInput][kups.potential.mliap.torch.interface.AtomGraphInput] into
MACE's PyG-style input format, plus a [load_mace][kups.potential.mliap.torch.mace.load_mace]
loader returning a [TorchMliap][kups.potential.mliap.torch.interface.TorchMliap].

Example:
    ```python
    from kups.potential.mliap.torch import load_mace, make_torch_mliap_from_state

    model = load_mace("mace.model", compute_cell_gradients=True)
    potential = make_torch_mliap_from_state(
        state_lens, compute_position_and_cell_gradients=True,
    )
    ```

Requires the ``torch`` dependency group: ``uv sync --group torch``.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]

from kups.potential.mliap.torch.interface import (
    TorchMliap,
    lattice_gradient_from_virial,
)

__all__ = ["MACEModule", "load_mace"]


class MACEModule(torch.nn.Module):
    """Adapter: ``AtomGraphInput`` → MACE PyG-style input → energy + gradients.

    Wraps a MACE ``nn.Module`` and translates the universal graph input into
    the (``node_attrs``, ``positions``, ``edge_index``, ``batch``, ``ptr``,
    ``shifts``, ``cell``) tuple that MACE expects. Returns gradients of energy
    w.r.t. positions (and optionally cell vectors).

    Attributes:
        mace: Underlying MACE ``nn.Module``.
        species_to_index: Buffer mapping atomic number ``Z`` → MACE species
            index (0..``num_species``-1).
        num_species: Number of species the MACE model was trained on.
        compute_cell_gradients: Whether to compute cell gradients (stress).
    """

    species_to_index: torch.Tensor

    def __init__(
        self,
        mace_model: torch.nn.Module,
        species_to_index: torch.Tensor,
        num_species: int,
        compute_cell_gradients: bool = False,
    ) -> None:
        """Initialise ``MACEModule``.

        Args:
            mace_model: Underlying MACE ``nn.Module``.
            species_to_index: Tensor mapping ``Z`` → MACE index. Indexed by
                atomic number; entries for unsupported ``Z`` are ignored.
            num_species: Number of species the MACE model was trained on.
            compute_cell_gradients: Whether to compute cell gradients.
        """
        super().__init__()
        self.mace = mace_model
        self.mace.eval()
        self.register_buffer("species_to_index", species_to_index.to(dtype=torch.int64))
        self.num_species = num_species
        self.compute_cell_gradients = compute_cell_gradients

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run MACE on a universal ``AtomGraphInput`` and return gradients.

        Args:
            input: Dict matching the universal ``AtomGraphInput`` schema.

        Returns:
            Dict with ``"energy"`` ``(B,)``, ``"position_gradients"`` ``(N, 3)``,
            and optionally ``"cell_gradients"`` ``(B, 3, 3)``.
        """
        # MACE is loaded at a fixed precision (float32 or float64 — see
        # ``load_mace(dtype=...)``); align every float input to that dtype
        # rather than whatever JAX hands us.
        model_dtype = next(self.mace.parameters()).dtype

        pos = input["pos"].to(model_dtype)
        species = input["atomic_numbers"]
        cell = input["cell"].to(model_dtype)
        batch = input["batch"]
        edge_index = input["edge_index"]
        cell_offsets = input["cell_offsets"].to(model_dtype)

        n_atoms = pos.shape[0]
        n_sys = cell.shape[0]

        # kUPS's neighbor list is a fixed-size buffer; ``indices_in`` maps
        # unused slots to the OOB sentinel ``len(keys) == n_atoms``. Drop
        # padded edges before indexing into atom-shaped tensors.
        valid_edge = (edge_index < n_atoms).all(dim=0)
        edge_index = edge_index[:, valid_edge]
        cell_offsets = cell_offsets[valid_edge]

        counts = torch.bincount(batch, minlength=n_sys)
        ptr = torch.cat([batch.new_zeros(1), counts.cumsum(0)])
        # ``species_to_index`` is registered as a CPU buffer at construction
        # time. ``TorchModuleWrapper`` calls us once with mock tensors on the
        # device of the wrapped MACE model (typically cuda) without first
        # calling ``self.to(device)``, so the indexing would straddle devices.
        # Co-locate the lookup table with the input on every call.
        species_to_index = self.species_to_index.to(species.device)
        node_attrs = F.one_hot(species_to_index[species], self.num_species).to(
            pos.dtype
        )
        cell_per_edge = cell[batch[edge_index[0]]]
        # cell_offsets (E,3) integer multiples → absolute Å via per-edge cell:
        # shifts[e, j] = Σ_i cell_offsets[e, i] * cell_per_edge[e, i, j]
        shifts = (cell_offsets.to(pos.dtype).unsqueeze(1) @ cell_per_edge).squeeze(1)

        out = self.mace(
            {
                "node_attrs": node_attrs,
                "positions": pos,
                "edge_index": edge_index,
                "batch": batch,
                "ptr": ptr,
                "shifts": shifts,
                # MACE's ``prepare_graph`` reads ``unit_shifts`` (the integer
                # cell-offset multiples) when ``compute_virials=True`` to build
                # the strain-perturbed graph in ``get_symmetric_displacement``.
                "unit_shifts": cell_offsets.to(pos.dtype),
                "cell": cell,
            },
            compute_force=True,
            compute_virials=self.compute_cell_gradients,
        )

        forces = out["forces"]
        result: dict[str, torch.Tensor] = {
            "energy": out["energy"].detach(),
            "position_gradients": (-forces).detach(),
        }
        if self.compute_cell_gradients:
            # MACE's ``virials`` = -sym(pos_virial + cell^T @ ∂E/∂h).
            # Negate to get the symmetric-strain virial, then invert.
            virial = -out["virials"]
            cell_grad = lattice_gradient_from_virial(
                forces=forces,
                positions=pos,
                batch=batch,
                cell=cell,
                virial=virial,
            )
            result["cell_gradients"] = cell_grad.detach()
        return result


def _build_species_to_index(model: torch.nn.Module) -> tuple[torch.Tensor, int]:
    """Build ``species_to_index`` lookup and ``num_species`` from a MACE model.

    Expects ``model.atomic_numbers`` to enumerate the supported atomic numbers
    in MACE-index order. The returned tensor has length ``max(Z) + 1`` so that
    indexing ``species_to_index[Z]`` gives the MACE species index for
    supported ``Z`` (unsupported entries default to 0).
    """
    atomic_numbers = cast(torch.Tensor, model.atomic_numbers).to(
        dtype=torch.int64, device="cpu"
    )
    num_species = int(atomic_numbers.numel())
    table_size = int(atomic_numbers.max().item()) + 1
    table = atomic_numbers.new_zeros(table_size)
    table[atomic_numbers] = torch.arange(num_species, dtype=torch.int64)
    return table, num_species


def load_mace(
    model_path: str | Path,
    device: str = "cuda",
    dtype: Literal["float32", "float64"] = "float32",
    compute_cell_gradients: bool = False,
    cutoff: float | None = None,
) -> TorchMliap:
    """Load a PyTorch MACE ``.model`` into a universal ``TorchMliap``.

    Args:
        model_path: Path to a MACE ``.model`` checkpoint.
        device: Device to load the model onto.
        dtype: Model precision — ``"float32"`` (default) or ``"float64"``.
        compute_cell_gradients: Whether to also compute virials/stress.
        cutoff: Cutoff radius [Å]. When ``None``, read from ``model.r_max``.

    Returns:
        ``TorchMliap`` ready to be wired into the kUPS interface.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Device 'cuda' requested but CUDA is not available. "
            "Use device='cpu' or ensure CUDA is properly installed."
        )

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"MACE model not found: {model_path}")

    model = torch.load(path, weights_only=False, map_location=device)
    model.eval()
    model = model.double() if dtype == "float64" else model.float()
    # Re-broadcast to the target device: some MACE/e3nn TorchScript submodules
    # carry Wigner-3j buffers that ``map_location`` and ``.float()`` don't
    # consistently move, and ``TorchModuleWrapper``'s mock-inference call
    # invokes the module before any ``.to(device)`` could rectify this.
    model = model.to(device)

    species_to_index, num_species = _build_species_to_index(model)
    if cutoff is None:
        cutoff = float(cast(torch.Tensor, model.r_max).item())

    module = MACEModule(
        model,
        species_to_index=species_to_index,
        num_species=num_species,
        compute_cell_gradients=compute_cell_gradients,
    )
    return TorchMliap.from_module(
        module, cutoff=cutoff, compute_cell_gradients=compute_cell_gradients
    )
