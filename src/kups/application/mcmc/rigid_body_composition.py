# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Bind rigid-body composition inputs to MCMC state."""

import jax.numpy as jnp
from jax import Array

from kups.application.mcmc.data import (
    MCMCGroup,
    MCMCParticles,
    MCMCSystems,
    MotifParticles,
)
from kups.core.data import Buffered, Table
from kups.core.lens import Lens
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.typing import GroupId, MotifParticleId, SystemId
from kups.mcmc.probability import motif_counts
from kups.potential.common.graph import PointCloud
from kups.potential.common.rigid_body_composition import RigidBodyComposition


def make_rigid_body_composition[State](
    state: State,
    cloud: PointCloud[MCMCParticles, MCMCSystems],
    motifs: Table[MotifParticleId, MotifParticles],
    groups: Lens[State, Buffered[GroupId, MCMCGroup]],
) -> RigidBodyComposition[State, MotifParticles]:
    """Describe rigid bodies that can be inserted into a fixed host."""
    systems = cloud.systems
    accept = systems.set_data(jnp.ones(len(systems), dtype=bool))

    def counts(
        state: State, patch: Patch[State] | None, old_input: bool = False
    ) -> WithPatch[Table[SystemId, Array], IdPatch[State]]:
        if patch is not None and not old_input:
            state = patch(state, accept)
        values = motif_counts(groups(state))
        rows = jnp.concatenate((jnp.ones((len(systems), 1)), values), axis=1)
        return WithPatch(systems.set_data(rows), IdPatch[State]())

    particles = cloud.particles.data
    return RigidBodyComposition(
        state,
        particles.system.apply_mask(~particles.group.valid_mask),
        motifs.data,
        motifs.data.motif,
        counts,
    )
