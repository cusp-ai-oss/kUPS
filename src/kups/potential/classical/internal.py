# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Internal energy corrections for rigid molecular motifs.

This module provides potential energy corrections accounting for intramolecular
energies of rigid molecules. Useful when only intermolecular interactions should
be computed (e.g., rigid body Monte Carlo with pre-optimized geometries).
"""

import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.lens import View
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import EMPTY, EmptyType, Energy, Potential, PotentialOut
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field


@dataclass
class MotifData:
    """Motif data with system assignment.

    Attributes:
        values: Motif index values (integers)
        system: System assignment for each motif entry
    """

    values: Array
    system: Index[SystemId]


@dataclass
class InternalEnergies[State, StatePatch: Patch](
    Potential[State, EmptyType, EmptyType, StatePatch]
):
    """Potential providing fixed internal energies for molecular motifs.

    Computes total energy by summing precomputed motif energies for all molecules
    in each system. Used to add/subtract intramolecular contributions in rigid
    body simulations where internal geometries are fixed.

    Type Parameters:
        State: Simulation state type
        StatePatch: Patch type for state updates

    Attributes:
        motifs: Lens to indexed motif data
        motif_potential_out: Lens to precomputed motif energies

    Note:
        Currently does not support gradients or Hessians (rigid molecules).
    """

    motifs: View[State, Table[ParticleId, MotifData]] = field(static=True)
    motif_potential_out: View[State, Energy] = field(static=True)

    def __call__(
        self, state: State, patch: StatePatch | None = None
    ) -> WithPatch[PotentialOut[EmptyType, EmptyType], Patch[State]]:
        sys_idx = self.motifs(state).data.system
        if patch is not None:
            accept = Table(sys_idx.keys, jnp.ones(sys_idx.num_labels, dtype=jnp.bool))
            state = patch(state, accept)
        motifs = self.motifs(state)
        motif_energies = self.motif_potential_out(state)

        out_energies = motifs.data.system.sum_over(motif_energies[motifs.data.values])
        return WithPatch(PotentialOut(out_energies, EMPTY, EMPTY), IdPatch())
