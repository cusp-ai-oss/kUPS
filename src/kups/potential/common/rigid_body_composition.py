# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Inputs for energy terms that depend only on rigid-body counts."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.lens import Lens, View, const_lens
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import Potential, PotentialOut
from kups.core.typing import MotifId, SystemId
from kups.core.utils.kahan import KahanSummand
from kups.potential.common.energy import (
    FullSumComposer,
    InputConstructor,
    PotentialFromEnergy,
)


@dataclass(frozen=True)
class RigidBodyComposition[State, Template]:
    """Rigid templates and counts for energy terms independent of body poses.

    Templates fix internal geometry and particle properties; flexible internal
    motion is unsupported. Translation, rotation and reinsertion preserve the
    coefficients. Rebuild after changing templates, background, parameters or cells.

    ``counts`` supplies one row per system, starting with a constant 1 for the
    background, followed by template counts. ``fixed_system`` masks out mobile
    particles in the initial state. Each potential prepares its own coefficients.
    """

    initial_state: State
    fixed_system: Index[SystemId]
    templates: Template
    template_motif: Index[MotifId]
    counts: InputConstructor[State, Table[SystemId, Array], Patch[State]]

    def sum_particles(self, fixed_values: Array, template_values: Array) -> Array:
        """Sum particle properties into background and template coefficients."""
        fixed = self.fixed_system.sum_over(fixed_values).data
        templates = self.template_motif.sum_over(template_values).data
        return jnp.concatenate(
            (
                fixed[:, None],
                jnp.broadcast_to(templates, (len(fixed), *templates.shape)),
            ),
            axis=1,
        )

    def potential[Gradients, Hessians](
        self,
        coefficients: Array,
        gradients: Gradients,
        hessian_lens: Lens[Gradients, Hessians],
        cache: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]] | None,
        patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None,
    ) -> Potential[State, Gradients, Hessians, Patch[State]]:
        """Use ordinary potential machinery for a linear or quadratic count sum."""
        if jax.tree.leaves(gradients):
            raise ValueError("Rigid-body composition requires energy-only evaluation")
        if coefficients.ndim not in (2, 3):
            raise ValueError("Composition coefficients must be linear or quadratic")

        def energy(
            inp: Table[SystemId, Array],
        ) -> WithPatch[Table[SystemId, Array], IdPatch[State]]:
            values = (
                jnp.einsum("gi,gij,gj->g", inp.data, coefficients, inp.data)
                if coefficients.ndim == 3
                else jnp.einsum("gi,gi->g", inp.data, coefficients)
            )
            return WithPatch(inp.set_data(values), IdPatch())

        return PotentialFromEnergy(
            energy_fn=energy,
            composer=FullSumComposer(self.counts),
            gradient_lens=const_lens(gradients),
            hessian_lens=hessian_lens,
            hessian_idx_view=const_lens(hessian_lens(gradients)),
            cache_lens=cache,
            patch_idx_view=patch_idx_view,
        )
