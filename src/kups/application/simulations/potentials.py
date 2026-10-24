# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Force-field selection and potential construction for the unified apps.

A discriminated [PotentialConfig][kups.application.simulations.potentials.PotentialConfig]
selects a force field; each config's ``build`` constructs the corresponding
[Potential][kups.core.potential.Potential] directly with its parameters (so the
parameters need not live on the simulation state) over the state's skin list and
returns the per-system cutoff the skin list serves. ``build`` gets the loaded
particle and system tables to size parameters from; it keeps no reference to them,
since the apps donate the state built from them.

The torch backends (MACE/UMA) import torch lazily inside ``build`` so importing
this module never requires the optional ``torch`` dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

import jax.numpy as jnp
import numpy as np
from jax import Array
from pydantic import BaseModel, Field

from kups.application.potential.classical.ewald import (
    IsEwaldSkinListState,
    make_ewald_from_state,
)
from kups.application.potential.classical.lennard_jones import (
    IsLJRowState,
    make_lennard_jones_row_from_state,
)
from kups.application.potential.classical.pme import make_pme_from_state
from kups.application.potential.mliap.tojax import (
    IsTojaxedSkinListState,
    make_tojaxed_from_state,
)
from kups.application.utils.path import get_model_path
from kups.core.cell import Periodic3D, require_periodic_3d
from kups.core.data import Table
from kups.core.lens import Lens
from kups.core.neighborlist import IsSkinListState, SkinNeighborList
from kups.core.patch import Patch
from kups.core.potential import EmptyType, Potential, sum_potentials
from kups.core.typing import HasCell, IsState, ParticleId, SystemId
from kups.core.utils.jax import astype_floating, tree_map
from kups.potential.classical.ewald import (
    EwaldParameters,
    IsEwaldPointData,
    PMEParameters,
)
from kups.potential.classical.lennard_jones import (
    IsLennardJonesParticles,
    LennardJonesParameters,
    MixingRule,
)
from kups.potential.common.geometry import Geometry, PositionsAndCell
from kups.potential.mliap.tojax import TojaxedMliap

if TYPE_CHECKING:
    # Type-only: referencing this in annotations never triggers a torch import
    # (annotations are lazy strings via ``from __future__ import annotations``).
    from kups.application.potential.mliap.torch import IsTorchMliapSkinListState

# Inlined here (not imported from kups.potential.mliap.torch.uma) so this module
# stays importable without the optional ``torch`` dependency.
type UMATaskName = Literal["omat", "omol", "oc20", "odac", "omc"]
type UMAInferenceSettings = Literal["default", "turbo", "turbo_f64"]

type BuiltPotential[State] = tuple[
    Potential[State, PositionsAndCell, EmptyType, Patch[Any]], Table[SystemId, Array]
]
"""``(potential, cutoff)``; the skin list is built at ``cutoff + skin``."""


class LjPotentialConfig(BaseModel):
    """Lennard-Jones force field."""

    backend: Literal["lj"] = "lj"
    cutoff: float
    parameters: dict[str, tuple[float | None, float | None]]
    mixing_rule: MixingRule
    kernel_dtype: Literal["float64", "float32"] = "float64"
    """Precision of the pair terms and of each atom's sums; per-system energies and
    stress are summed in float64. ``float32`` runs a Pallas kernel on CUDA GPUs of
    compute capability 8.0+. With ``float32``, pairs within about 2e-7 (relative) of
    the cutoff may fall on either side of it, and the energies carry round-off that
    line searches see near convergence, so keep ``float64`` for tight relaxations."""

    def build[State](
        self,
        state_lens: Lens[State, IsLJRowState],
        gradient: Lens[Geometry, PositionsAndCell],
        particles: Table[ParticleId, object],
        systems: Table[SystemId, object],
    ) -> BuiltPotential[State]:
        """Build LJ over the skin list's neighbor rows with its parameters bound in."""
        del particles, systems
        params = LennardJonesParameters.from_dict(
            cutoff=self.cutoff,
            parameters=self.parameters,
            mixing_rule=self.mixing_rule,
        )
        potential = make_lennard_jones_row_from_state(
            state_lens, parameters=params, gradient=gradient, dtype=self.kernel_dtype
        )
        return potential, params.cutoff


class CoulombConfig(BaseModel):
    """Coulomb interactions of the input charges, split by Ewald screening."""

    method: Literal["ewald", "pme"] = "pme"
    """Reciprocal sum on a particle mesh (``pme``) or over k-vectors (``ewald``,
    faster only for cells of up to a few hundred atoms)."""
    precision: float = Field(default=1e-6, gt=0, lt=1)
    """Target error of the real- and reciprocal-space sums."""

    def build[State](
        self,
        state_lens: Lens[State, IsEwaldSkinListState],
        gradient: Lens[Geometry, PositionsAndCell],
        particles: Table[ParticleId, IsEwaldPointData],
        systems: Table[SystemId, HasCell[Periodic3D]],
        cutoff: float,
    ) -> BuiltPotential[State]:
        """Build the Coulomb terms over the skin list's pairs."""
        require_periodic_3d(systems.data.cell)
        charged = particles.data.system.max_over(jnp.abs(particles.data.charges))
        uncharged = np.flatnonzero(~(np.asarray(charged.data) > 0)).tolist()
        if uncharged:
            raise ValueError(
                f"Systems {uncharged} have no charges. Give them as _atom_site_charge "
                "or _atom_type_partial_charge in CIF files, or as the atoms' initial "
                "charges, as extxyz files store them."
            )
        dtype = particles.data.positions.dtype
        match self.method:
            case "ewald":
                ewald = EwaldParameters.make(
                    particles, systems, self.precision, cutoff, reciprocal_dtype=dtype
                )
                ewald = astype_floating(ewald, dtype)
                potential = make_ewald_from_state(
                    state_lens,
                    parameters=ewald,
                    gradient=gradient,
                    neighborlist_factory=SkinNeighborList.from_state,
                )
                return potential, ewald.cutoff
            case "pme":
                pme = PMEParameters.make(
                    particles, systems, self.precision, cutoff, reciprocal_dtype=dtype
                )
                pme = astype_floating(pme, dtype)
                potential = make_pme_from_state(
                    state_lens,
                    parameters=pme,
                    gradient=gradient,
                    neighborlist_factory=SkinNeighborList.from_state,
                )
                return potential, pme.cutoff


class _IsLjCoulombParticles(IsLennardJonesParticles, IsEwaldPointData, Protocol):
    """Particles with species labels and charges."""


class _IsLjCoulombState(
    IsState[_IsLjCoulombParticles, HasCell[Periodic3D]], IsSkinListState, Protocol
):
    """Charged particles in periodic systems, and a skin list."""


class LjCoulombPotentialConfig(BaseModel):
    """Lennard-Jones plus the Coulomb interactions of the input charges."""

    backend: Literal["lj_coulomb"] = "lj_coulomb"
    lj: LjPotentialConfig
    """Lennard-Jones terms; ``lj.cutoff`` is also the real-space Coulomb cutoff."""
    coulomb: CoulombConfig

    def build[State](
        self,
        state_lens: Lens[State, _IsLjCoulombState],
        gradient: Lens[Geometry, PositionsAndCell],
        particles: Table[ParticleId, IsEwaldPointData],
        systems: Table[SystemId, HasCell[Periodic3D]],
    ) -> BuiltPotential[State]:
        """Build LJ over the skin list's neighbor rows and Coulomb over its pairs."""
        lj, lj_cutoff = self.lj.build(state_lens, gradient, particles, systems)
        coulomb, coulomb_cutoff = self.coulomb.build(
            state_lens, gradient, particles, systems, self.lj.cutoff
        )
        # In float32, the Coulomb cutoff can round up past the LJ one.
        cutoff = tree_map(jnp.maximum, lj_cutoff, coulomb_cutoff)
        return sum_potentials(lj, coulomb), cutoff


class TojaxPotentialConfig(BaseModel):
    """Jaxified MLFF (a model exported to JAX via tojax)."""

    backend: Literal["tojax"] = "tojax"
    model_path: str | Path

    def build[State](
        self,
        state_lens: Lens[State, IsTojaxedSkinListState],
        gradient: Lens[Geometry, PositionsAndCell],
        particles: Table[ParticleId, object],
        systems: Table[SystemId, object],
    ) -> BuiltPotential[State]:
        """Build the jaxified potential from the exported model."""
        del particles, systems
        model = TojaxedMliap.from_zip_file(get_model_path(self.model_path))
        potential = make_tojaxed_from_state(
            state_lens,
            parameters=model,
            gradient=gradient,
            neighborlist_factory=SkinNeighborList.from_state,
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
        state_lens: Lens[State, IsTorchMliapSkinListState],
        gradient: Lens[Geometry, PositionsAndCell],
        particles: Table[ParticleId, object],
        systems: Table[SystemId, object],
    ) -> BuiltPotential[State]:
        """Load the MACE checkpoint and build the potential (lazy torch import)."""
        del particles, systems
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
            neighborlist_factory=SkinNeighborList.from_state,
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
        state_lens: Lens[State, IsTorchMliapSkinListState],
        gradient: Lens[Geometry, PositionsAndCell],
        particles: Table[ParticleId, object],
        systems: Table[SystemId, object],
    ) -> BuiltPotential[State]:
        """Load the UMA checkpoint and build the potential (lazy torch import)."""
        del particles, systems
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
            neighborlist_factory=SkinNeighborList.from_state,
        )
        return potential, model.cutoff


type PotentialConfig = Annotated[
    LjPotentialConfig
    | LjCoulombPotentialConfig
    | TojaxPotentialConfig
    | MaceConfig
    | UmaConfig,
    Field(discriminator="backend"),
]
