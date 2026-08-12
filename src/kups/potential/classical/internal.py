# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Internal energy corrections for rigid molecular motifs.

This module provides potential energy corrections accounting for intramolecular
energies of rigid molecules. Useful when only intermolecular interactions should
be computed (e.g., rigid body Monte Carlo with pre-optimized geometries).
"""

from typing import Any, Literal, overload

import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.lens import View
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import (
    EMPTY,
    CompensatedPotentialResult,
    EmptyType,
    Energy,
    Potential,
    PotentialOut,
    PotentialResult,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field
from kups.core.utils.kahan import KahanSummand


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
class InternalEnergies[State, StatePatch: Patch[Any]](
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

    @overload
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
        *,
        include_compensate: Literal[False] = False,
    ) -> PotentialResult[State, EmptyType, EmptyType]: ...
    @overload
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
        *,
        include_compensate: Literal[True],
    ) -> CompensatedPotentialResult[State, EmptyType, EmptyType]: ...
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
        *,
        include_compensate: bool = False,
    ) -> (
        PotentialResult[State, EmptyType, EmptyType]
        | CompensatedPotentialResult[State, EmptyType, EmptyType]
    ):
        """Sum the precomputed motif energies per system.

        Args:
            state: Current simulation state
            patch: Optional state patch applied before summing
            include_compensate: Return the accumulator instead of its value. The
                energies are read from a table rather than accumulated, so the
                compensation is zero.

        Returns:
            Potential output with an identity patch
        """
        sys_idx = self.motifs(state).data.system
        if patch is not None:
            accept = Table(sys_idx.keys, jnp.ones(sys_idx.num_labels, dtype=jnp.bool))
            state = patch(state, accept)
        motifs = self.motifs(state)
        motif_energies = self.motif_potential_out(state)

        out_energies = motifs.data.system.sum_over(motif_energies[motifs.data.values])
        out = PotentialOut(out_energies, EMPTY, EMPTY)
        if include_compensate:
            return WithPatch(KahanSummand.init(out), IdPatch[Any]())
        return WithPatch(out, IdPatch[Any]())
