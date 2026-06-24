# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Force-field selection and potential construction for the unified apps.

A discriminated [PotentialConfig][kups.application.simulations.potentials.PotentialConfig]
selects a force field; each config's ``build`` constructs the corresponding
[Potential][kups.core.potential.Potential] directly with its parameters (so the
parameters need not live on the simulation state) and returns the per-system
cutoff used to size the neighbor list.

The torch backends (MACE/UMA) import torch lazily inside ``build`` so importing
this module never requires the optional ``torch`` dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, Field

from kups.application.potential.classical.lennard_jones import (
    IsLJGraphState,
    make_lennard_jones_from_state,
)
from kups.application.potential.mliap.tojax import (
    IsTojaxedGraphState,
    make_tojaxed_from_state,
)
from kups.application.utils.path import get_model_path
from kups.core.lens import Lens
from kups.core.patch import Patch
from kups.core.potential import EmptyType, Potential
from kups.potential.classical.lennard_jones import LennardJonesParameters, MixingRule
from kups.potential.common.geometry import Geometry, PositionsAndCell
from kups.potential.mliap.tojax import TojaxedMliap

if TYPE_CHECKING:
    # Type-only: referencing this in annotations never triggers a torch import
    # (annotations are lazy strings via ``from __future__ import annotations``).
    from kups.application.potential.mliap.torch import IsTorchMliapGraphState

# Inlined here (not imported from kups.potential.mliap.torch.uma) so this module
# stays importable without the optional ``torch`` dependency.
type UMATaskName = Literal["omat", "omol", "oc20", "odac", "omc"]
type UMAInferenceSettings = Literal["default", "turbo", "turbo_f64"]

type BuiltPotential[State] = tuple[
    Potential[State, PositionsAndCell, EmptyType, Patch[Any]], float
]
"""``(potential, cutoff)`` where ``cutoff`` sizes the neighbor list."""


class LjPotentialConfig(BaseModel):
    """Lennard-Jones force field."""

    backend: Literal["lj"] = "lj"
    cutoff: float
    parameters: dict[str, tuple[float | None, float | None]]
    mixing_rule: MixingRule

    def build[State](
        self,
        state_lens: Lens[State, IsLJGraphState],
        gradient: Lens[Geometry, PositionsAndCell],
    ) -> BuiltPotential[State]:
        """Build the LJ potential with its parameters bound in."""
        params = LennardJonesParameters.from_dict(
            cutoff=self.cutoff,
            parameters=self.parameters,
            mixing_rule=self.mixing_rule,
        )
        potential = make_lennard_jones_from_state(
            state_lens, parameters=params, gradient=gradient
        )
        return potential, params.cutoff


class TojaxPotentialConfig(BaseModel):
    """Jaxified MLFF (a model exported to JAX via tojax)."""

    backend: Literal["tojax"] = "tojax"
    model_path: str | Path

    def build[State](
        self,
        state_lens: Lens[State, IsTojaxedGraphState],
        gradient: Lens[Geometry, PositionsAndCell],
    ) -> BuiltPotential[State]:
        """Build the jaxified potential from the exported model."""
        model = TojaxedMliap.from_zip_file(get_model_path(self.model_path))
        potential = make_tojaxed_from_state(
            state_lens, parameters=model, gradient=gradient
        )
        return potential, model.cutoff


class MaceConfig(BaseModel):
    """PyTorch MACE checkpoint."""

    backend: Literal["mace"] = "mace"
    model_path: str | Path
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: Literal["float32", "float64"] = "float32"

    def build[State](
        self,
        state_lens: Lens[State, IsTorchMliapGraphState],
        gradient: Lens[Geometry, PositionsAndCell],
    ) -> BuiltPotential[State]:
        """Load the MACE checkpoint and build the potential (lazy torch import)."""
        from kups.application.potential.mliap.torch import make_torch_mliap_from_state
        from kups.potential.mliap.torch import load_mace

        model = load_mace(
            get_model_path(self.model_path),
            device=self.device,
            dtype=self.dtype,
            compute_cell_gradients=True,
        )
        potential = make_torch_mliap_from_state(
            state_lens, parameters=model, gradient=gradient
        )
        return potential, model.cutoff


class UmaConfig(BaseModel):
    """Meta FAIR Chemistry UMA checkpoint."""

    backend: Literal["uma"] = "uma"
    model_path: str | Path
    device: Literal["cpu", "cuda"] = "cuda"
    task_name: UMATaskName = "omat"
    inference_settings: UMAInferenceSettings = "default"

    def build[State](
        self,
        state_lens: Lens[State, IsTorchMliapGraphState],
        gradient: Lens[Geometry, PositionsAndCell],
    ) -> BuiltPotential[State]:
        """Load the UMA checkpoint and build the potential (lazy torch import)."""
        from kups.application.potential.mliap.torch import make_torch_mliap_from_state
        from kups.potential.mliap.torch import load_uma

        model = load_uma(
            get_model_path(self.model_path),
            device=self.device,
            task_name=self.task_name,
            compute_cell_gradients=True,
            inference_settings=self.inference_settings,
        )
        potential = make_torch_mliap_from_state(
            state_lens, parameters=model, gradient=gradient
        )
        return potential, model.cutoff


type PotentialConfig = Annotated[
    LjPotentialConfig | TojaxPotentialConfig | MaceConfig | UmaConfig,
    Field(discriminator="backend"),
]
