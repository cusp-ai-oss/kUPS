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
from kups.application.potential.classical.lennard_jones_ewald import (
    IsLJEwaldGraphState,
    make_lennard_jones_ewald_from_state,
)
from kups.application.potential.mliap.tojax import (
    IsTojaxedGraphState,
    make_tojaxed_from_state,
)
from kups.application.simulations.mcmc_rigid import EwaldConfig
from kups.application.utils.data import Scalar
from kups.application.utils.path import get_model_path
from kups.core import units
from kups.core.cell import Periodic3D
from kups.core.data import Table
from kups.core.geometry import Geometry, PositionsAndCell
from kups.core.lens import Lens
from kups.core.neighborlist import AdaptiveNeighborList, NeighborListFactory
from kups.core.patch import Patch
from kups.core.potential import EmptyType, Potential
from kups.core.typing import HasCell, ParticleId, SystemId
from kups.potential.classical.ewald import (
    EwaldParameters,
    IsEwaldPointData,
    PMEParameters,
)
from kups.potential.classical.lennard_jones import LennardJonesParameters, MixingRule
from kups.potential.mliap.tojax import TojaxedMliap

if TYPE_CHECKING:
    # Type-only: referencing this in annotations never triggers a torch import
    # (annotations are lazy strings via ``from __future__ import annotations``).
    from kups.application.potential.mliap.torch import IsTorchMliapGraphState

# Inlined here (not imported from kups.potential.mliap.torch.uma) so this module
# stays importable without the optional ``torch`` dependency.
type UMATaskName = Literal["omat", "omol", "oc20", "odac", "omc"]
type UMAInferenceSettings = Literal["default", "turbo", "turbo_f64"]

# ``(potential, cutoff)``; the cutoff sizes the neighbor list.
type BuiltPotential[State] = tuple[
    Potential[State, PositionsAndCell[units.Force], EmptyType, Patch[Any]],
    Table[SystemId, units.Length],
]


class LjPotentialConfig(BaseModel):
    """Lennard-Jones force field."""

    backend: Literal["lj"] = "lj"
    cutoff: Annotated[units.Length, Scalar]
    parameters: dict[
        str,
        tuple[
            Annotated[units.Length, Scalar] | None,
            Annotated[units.Energy, Scalar] | None,
        ],
    ]
    """Mapping label -> (sigma, epsilon); a null pair means no interaction."""
    mixing_rule: MixingRule

    def build[State, Focus: IsLJGraphState](
        self,
        state_lens: Lens[State, Focus],
        gradient: Lens[Geometry, PositionsAndCell[units.Length]],
        neighborlist_factory: NeighborListFactory[
            Focus
        ] = AdaptiveNeighborList.from_state,
    ) -> BuiltPotential[State]:
        """Build the LJ potential with its parameters bound in."""
        params = LennardJonesParameters.from_dict(
            cutoff=self.cutoff,
            parameters=self.parameters,
            mixing_rule=self.mixing_rule,
        )
        potential = make_lennard_jones_from_state(
            state_lens,
            parameters=params,
            gradient=gradient,
            neighborlist_factory=neighborlist_factory,
        )
        return potential, params.cutoff


class LjEwaldPotentialConfig(BaseModel):
    """Lennard-Jones plus the Coulomb interactions of the input charges."""

    backend: Literal["lj_ewald"] = "lj_ewald"
    cutoff: Annotated[units.Length, Scalar]
    parameters: dict[
        str,
        tuple[
            Annotated[units.Length, Scalar] | None,
            Annotated[units.Energy, Scalar] | None,
        ],
    ]
    """Mapping label -> (sigma, epsilon); a null pair means no interaction."""
    mixing_rule: MixingRule
    ewald: EwaldConfig
    reciprocal: Literal["pme", "ewald"] = "pme"
    """Reciprocal sum on a particle mesh (``pme``) or over k-vectors (``ewald``)."""

    def build[State, Focus: IsLJEwaldGraphState](
        self,
        state_lens: Lens[State, Focus],
        gradient: Lens[Geometry, PositionsAndCell[units.Length]],
        particles: Table[ParticleId, IsEwaldPointData],
        systems: Table[SystemId, HasCell[Periodic3D]],
        neighborlist_factory: NeighborListFactory[
            Focus
        ] = AdaptiveNeighborList.from_state,
    ) -> BuiltPotential[State]:
        """Build both potentials with their parameters bound in.

        Screening, cutoffs and the mesh are estimated once from ``particles``
        and ``systems``, the loaded input. They do not follow a changing cell.

        Raises:
            ValueError: If the input has no charges.
        """
        if not units.absolute(particles.data.charges).sum().value > 0:
            raise ValueError(
                "The lj_ewald backend needs charges; CIF files give them as "
                "_atom_site_charge."
            )
        lj = LennardJonesParameters.from_dict(
            cutoff=self.cutoff,
            parameters=self.parameters,
            mixing_rule=self.mixing_rule,
        )
        make = PMEParameters.make if self.reciprocal == "pme" else EwaldParameters.make
        ewald = make(
            particles, systems, self.ewald.epsilon_total, self.ewald.real_cutoff
        )
        potential = make_lennard_jones_ewald_from_state(
            state_lens,
            lennard_jones=lj,
            ewald=ewald,
            gradient=gradient,
            neighborlist_factory=neighborlist_factory,
        )
        cutoff = units.maximum(
            Table.broadcast_to(lj.cutoff, ewald.cutoff).data, ewald.cutoff.data
        )
        return potential, ewald.cutoff.set_data(cutoff)


class TojaxPotentialConfig(BaseModel):
    """Jaxified MLFF (a model exported to JAX via tojax)."""

    backend: Literal["tojax"] = "tojax"
    model_path: str | Path

    def build[State, Focus: IsTojaxedGraphState](
        self,
        state_lens: Lens[State, Focus],
        gradient: Lens[Geometry, PositionsAndCell[units.Length]],
        neighborlist_factory: NeighborListFactory[
            Focus
        ] = AdaptiveNeighborList.from_state,
    ) -> BuiltPotential[State]:
        """Build the jaxified potential from the exported model."""
        model = TojaxedMliap.from_zip_file(get_model_path(self.model_path))
        potential = make_tojaxed_from_state(
            state_lens,
            parameters=model,
            gradient=gradient,
            neighborlist_factory=neighborlist_factory,
        )
        return potential, model.cutoff


class MaceConfig(BaseModel):
    """PyTorch MACE checkpoint."""

    backend: Literal["mace"] = "mace"
    model_path: str | Path
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: Literal["float32", "float64"] = "float32"

    def build[State, Focus: IsTorchMliapGraphState](
        self,
        state_lens: Lens[State, Focus],
        gradient: Lens[Geometry, PositionsAndCell[units.Length]],
        neighborlist_factory: NeighborListFactory[
            Focus
        ] = AdaptiveNeighborList.from_state,
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
            state_lens,
            parameters=model,
            gradient=gradient,
            neighborlist_factory=neighborlist_factory,
        )
        return potential, model.cutoff


class UmaConfig(BaseModel):
    """Meta FAIR Chemistry UMA checkpoint."""

    backend: Literal["uma"] = "uma"
    model_path: str | Path
    device: Literal["cpu", "cuda"] = "cuda"
    task_name: UMATaskName = "omat"
    inference_settings: UMAInferenceSettings = "default"

    def build[State, Focus: IsTorchMliapGraphState](
        self,
        state_lens: Lens[State, Focus],
        gradient: Lens[Geometry, PositionsAndCell[units.Length]],
        neighborlist_factory: NeighborListFactory[
            Focus
        ] = AdaptiveNeighborList.from_state,
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
            state_lens,
            parameters=model,
            gradient=gradient,
            neighborlist_factory=neighborlist_factory,
        )
        return potential, model.cutoff


type PotentialConfig = Annotated[
    LjPotentialConfig | TojaxPotentialConfig | MaceConfig | UmaConfig,
    Field(discriminator="backend"),
]

# ``kups_md`` builds its potential from the loaded input, as Ewald needs.
type MdPotentialConfig = Annotated[
    LjPotentialConfig
    | LjEwaldPotentialConfig
    | TojaxPotentialConfig
    | MaceConfig
    | UmaConfig,
    Field(discriminator="backend"),
]
