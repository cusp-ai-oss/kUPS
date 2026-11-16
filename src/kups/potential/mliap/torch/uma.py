# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""UMA adapter for the universal torch MLFF interface.

Wraps Meta FAIR Chemistry's [UMA](https://huggingface.co/facebook/UMA) models
(via [fairchem-core](https://github.com/facebookresearch/fairchem) ≥ 2.0).
Public loader: [load_uma][kups.potential.mliap.torch.uma.load_uma].

UMA is a mixture-of-experts model with several dataset-specific inference
heads — pick one with ``task_name`` (``"omat"`` for inorganic materials,
``"omol"`` for molecules, ``"oc20"`` for catalysis, ``"odac"`` for
MOFs/direct-air-capture, ``"omc"`` for molecular crystals).

Example:
    ```python
    from kups.application.potential.filter import POSITIONS_AND_CELL
    from kups.application.potential.mliap.torch import make_torch_mliap_from_state
    from kups.potential.mliap.torch import load_uma

    model = load_uma(
        "uma-s-1.2.pt", task_name="omat", compute_cell_gradients=True,
    )
    potential = make_torch_mliap_from_state(state_lens, gradient=POSITIONS_AND_CELL)
    ```

Requires the ``uma`` extras group: ``uv sync --extra uma``.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, override

import torch  # pyright: ignore[reportMissingImports]

from kups.potential.mliap.torch.interface import (
    TorchMliap,
    lattice_gradient_from_virial,
)

__all__ = ["UMAInferenceSettings", "UMAModule", "UMATaskName", "load_uma"]


type UMATaskName = Literal["omat", "omol", "oc20", "odac", "omc"]
type UMAInferenceSettings = Literal["default", "turbo", "turbo_f64"]


class UMAModule(torch.nn.Module):
    """Adapter: ``AtomGraphInput`` → fairchem ``AtomicData`` → energy + gradients.

    Wraps a fairchem ``MLIPPredictUnit`` and translates the universal graph
    input into the ``AtomicData`` object UMA expects. Returns gradients of
    energy w.r.t. positions (and optionally w.r.t. cell vectors).

    The wrapped predict-unit holds its own torch module and manages its own
    device placement; this adapter intentionally does not register it as a
    submodule (it is not an ``nn.Module``).

    Attributes:
        predict_unit: fairchem ``MLIPPredictUnit`` (held by reference).
        task_name: UMA inference head to route every system to.
        compute_cell_gradients: Whether to also return ``"cell_gradients"``.

    Note:
        UMA's ``stress`` is the symmetrized strain virial ``V_ij /
        volume`` from a joint symmetric strain on positions and cell
        (cf. ``fairchem.core.models.uma.outputs.compute_forces_and_stress``).
        We subtract the (fully known) position virial and apply the ``h^-T``
        factor to recover the raw lattice gradient ``∂E/∂h`` see
        ``lattice_gradient_from_virial``.
    """

    def __init__(
        self,
        predict_unit: Any,  # noqa: ANN401
        task_name: UMATaskName | str = "omat",
        compute_cell_gradients: bool = False,
    ) -> None:
        """Initialise ``UMAModule``.

        Args:
            predict_unit: fairchem ``MLIPPredictUnit`` (already loaded onto a
                device).
            task_name: UMA inference head (e.g. ``"omat"``, ``"omol"``).
            compute_cell_gradients: Whether to compute cell gradients (stress).
        """
        super().__init__()
        # PredictUnit is not an nn.Module; keep as plain attribute so
        # ``module.to(device)`` does not try to traverse it.
        self.predict_unit = predict_unit
        self.task_name = str(task_name)
        self.compute_cell_gradients = compute_cell_gradients

    @override
    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # type: ignore
        """Run UMA on a universal ``AtomGraphInput`` and return gradients.

        Args:
            input: Dict matching the universal ``AtomGraphInput`` schema.

        Returns:
            Dict with ``"energy"`` ``(B,)``, ``"position_gradients"`` ``(N, 3)``,
            and optionally ``"cell_gradients"`` ``(B, 3, 3)``.
        """
        from fairchem.core.datasets.atomic_data import (  # pyright: ignore[reportMissingImports]
            AtomicData,
        )

        # Leave inputs on the device JAX handed us; the first-call branch below
        # covers the lazy-init device issue. Cast floats to the inference dtype
        # now, because turbo's MOLE merge runs before UMA's own dtype cast and
        # fails on float64 data with float32 weights.
        inf_dtype = self.predict_unit.inference_settings.base_precision_dtype
        pos = input["pos"].to(dtype=inf_dtype)
        species = input["atomic_numbers"].to(dtype=torch.int64)
        cell = input["cell"].to(dtype=inf_dtype)
        batch = input["batch"].to(dtype=torch.int64).clone()
        edge_index = input["edge_index"].to(dtype=torch.int64)
        cell_offsets = -input["cell_offsets"].to(dtype=inf_dtype)
        pbc = input["pbc"]
        charge = input["charge"].to(dtype=inf_dtype)
        spin = input["spin"].to(dtype=inf_dtype)

        n_atoms = pos.shape[0]
        n_sys = cell.shape[0]

        # ``TorchModuleWrapper._get_output_info`` probes with all-zero species.
        # Turbo mode merges MOLE experts for the composition of the first real
        # ``predict()`` call, so answer the probe with zeros of the right shape
        # instead of letting it fix an all-hydrogen composition.
        if int(species.abs().sum().item()) == 0:
            out_dtype = self.predict_unit.inference_settings.base_precision_dtype
            dev = pos.device
            result: dict[str, torch.Tensor] = {
                "energy": torch.zeros(n_sys, dtype=out_dtype, device=dev),
                "position_gradients": torch.zeros(
                    n_atoms, 3, dtype=out_dtype, device=dev
                ),
            }
            if self.compute_cell_gradients:
                result["cell_gradients"] = torch.zeros(
                    n_sys, 3, 3, dtype=out_dtype, device=dev
                )
            return result

        # kUPS's neighbor list is a fixed-size buffer; ``indices_in`` maps
        # unused slots to the OOB sentinel ``len(keys) == n_atoms``. Drop
        # those padded edges before handing the graph to UMA — otherwise the
        # species/position gather hits true out-of-bounds indices and CUDA
        # asserts.
        valid_edge = (edge_index < n_atoms).all(dim=0)
        edge_index = edge_index[:, valid_edge]
        cell_offsets = cell_offsets[valid_edge]

        # Pin the last atom to the last system so ``batch.max() + 1 == n_sys``
        # (validates AtomicData). For ``sorted_by_system`` real input this is
        # a no-op (the last atom is already in the last system).
        batch[-1] = n_sys - 1

        # ``_lazy_init`` prepares the model on cpu before moving it to its
        # device, so the first call needs cpu data or the MOLE merge mixes cuda
        # indices with cpu weights. ``predict`` moves data back for the forward.
        if not getattr(self.predict_unit, "lazy_model_intialized", True):
            pos = pos.cpu()
            species = species.cpu()
            cell = cell.cpu()
            batch = batch.cpu()
            edge_index = edge_index.cpu()
            cell_offsets = cell_offsets.cpu()
            pbc = pbc.cpu()
            charge = charge.cpu()
            spin = spin.cpu()

        natoms = torch.bincount(batch, minlength=n_sys)
        edge_batch = batch[edge_index[0]]
        nedges = torch.bincount(edge_batch, minlength=n_sys)
        fixed = torch.zeros(n_atoms, dtype=torch.int64, device=pos.device)
        tags = torch.zeros(n_atoms, dtype=torch.int64, device=pos.device)

        data = AtomicData(
            pos=pos,
            atomic_numbers=species,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets.to(pos.dtype),
            nedges=nedges,
            charge=charge,
            spin=spin,
            fixed=fixed,
            tags=tags,
            batch=batch,
            sid=[""] * n_sys,
            dataset=[self.task_name] * n_sys,
        )

        preds = self.predict_unit.predict(data)

        forces = preds["forces"]
        # ``predict_unit`` places outputs on its own device, which may differ
        # from the input device (cpu mock vs cuda predict-unit, single- vs
        # multi-gpu). Pin our post-processing tensors to the output device so
        # ``stress * volume`` and ``lattice_gradient_from_virial`` stay on a
        # single device.
        out_device = forces.device
        result: dict[str, torch.Tensor] = {
            "energy": preds["energy"].detach(),
            "position_gradients": (-forces).detach(),
        }
        if self.compute_cell_gradients:
            stress = preds["stress"]
            # UMA flattens stress to (B, 9); reshape to (B, 3, 3) if needed.
            if stress.dim() == 2 and stress.shape[-1] == 9:
                stress = stress.view(-1, 3, 3)
            cell_d = cell.to(out_device)
            volume = torch.linalg.det(cell_d).abs()
            virial = stress * volume.view(-1, 1, 1)
            cell_grad = lattice_gradient_from_virial(
                forces=forces,
                positions=pos.to(out_device),
                batch=batch.to(out_device),
                cell=cell_d,
                virial=virial,
            )
            result["cell_gradients"] = cell_grad.detach()
        return result


def load_uma(
    model_path: str | Path,
    device: Literal["cpu", "cuda"] = "cuda",
    task_name: UMATaskName | str = "omat",
    compute_cell_gradients: bool = False,
    cutoff: float = 6.0,
    inference_settings: UMAInferenceSettings = "default",
) -> TorchMliap:
    """Load a Meta FAIR Chemistry UMA checkpoint into a ``TorchMliap``.

    Args:
        model_path: Path to a UMA ``.pt`` checkpoint (e.g. ``uma-s-1.2.pt``).
        device: Device to load the model onto.
        task_name: UMA inference head — ``"omat"`` (materials),
            ``"omol"`` (molecules), ``"oc20"`` (catalysis),
            ``"odac"`` (MOFs / DAC), ``"omc"`` (molecular crystals).
        compute_cell_gradients: Whether to also return cell gradients
            (stress). See ``UMAModule`` for convention caveats.
        cutoff: Cutoff radius [Å]. UMA-s-1.2 defaults to 6.0.
        inference_settings: ``"default"`` or ``"turbo"`` (resolved by
            fairchem's ``guess_inference_settings``), or ``"turbo_f64"`` for
            a compiled, MOLE-merged float64 preset.

    Returns:
        ``TorchMliap`` ready to be wired into the kUPS interface.

    Raises:
        ImportError: If ``fairchem-core>=2.0`` is not installed.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Device 'cuda' requested but CUDA is not available. "
            "Use device='cpu' or ensure CUDA is properly installed."
        )

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"UMA model not found: {model_path}")

    try:
        from dataclasses import replace

        from fairchem.core.units.mlip_unit import (  # pyright: ignore[reportMissingImports]
            load_predict_unit,
        )
        from fairchem.core.units.mlip_unit.api.inference import (  # pyright: ignore[reportMissingImports]
            InferenceSettings,
            guess_inference_settings,
        )
    except ImportError as exc:
        raise ImportError(
            "Loading UMA requires fairchem-core>=2.0. "
            "Install with `uv sync --extra uma`."
        ) from exc

    # kUPS supplies the radius graph with the same cutoff, so force external
    # graph generation; this also avoids compile issues in UMA's own graph code.
    # ``turbo_f64`` is a custom preset beyond fairchem's named ones.
    custom_settings = {
        "turbo_f64": InferenceSettings(
            base_precision_dtype=torch.float64,
            tf32=False,
            activation_checkpointing=False,
            merge_mole=True,
            compile=True,
            external_graph_gen=False,
            internal_graph_gen_version=2,
            execution_mode="general",
        ),
    }
    settings = custom_settings.get(inference_settings) or guess_inference_settings(
        inference_settings
    )
    settings = replace(settings, external_graph_gen=True)

    predict_unit = load_predict_unit(
        path=str(path),
        device=device,
        inference_settings=settings,
    )
    module = UMAModule(
        predict_unit,
        task_name=task_name,
        compute_cell_gradients=compute_cell_gradients,
    )
    return TorchMliap.from_module(
        module, cutoff=cutoff, compute_cell_gradients=compute_cell_gradients
    )
