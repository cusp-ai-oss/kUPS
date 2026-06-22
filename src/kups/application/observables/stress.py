# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for virial stress observables.

These adapters extract particles, groups, and systems from a concrete
simulation state and delegate to the state-agnostic virial-theorem functions in
[kups.observables.stress][].
"""

from __future__ import annotations

from typing import Protocol

from jax import Array

from kups.core.data import Table
from kups.core.typing import GroupId, HasSystemIndex, IsState, SystemId
from kups.observables.stress import (
    IsMolecularVirialParticles,
    IsVirialParticles,
    IsVirialSystems,
    molecular_stress_via_virial_theorem,
    stress_via_virial_theorem,
)


class IsMolecularVirialState(
    IsState[IsMolecularVirialParticles, IsVirialSystems], Protocol
):
    """State with groups for molecular virial stress."""

    @property
    def groups(self) -> Table[GroupId, HasSystemIndex]: ...


def virial_stress_from_state(
    key: Array, state: IsState[IsVirialParticles, IsVirialSystems]
) -> Table[SystemId, Array]:
    """Compute atomic virial stress from a state."""
    del key
    return stress_via_virial_theorem(state.particles, state.systems)


def molecular_virial_stress_from_state(
    key: Array, state: IsMolecularVirialState
) -> Table[SystemId, Array]:
    """Compute molecular virial stress from a state."""
    del key
    return molecular_stress_via_virial_theorem(
        state.particles, state.groups, state.systems
    )
