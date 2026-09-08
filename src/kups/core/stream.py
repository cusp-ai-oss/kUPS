# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Particle-row reservations and assertion-based host refill requests."""

import jax.numpy as jnp
from jax import Array

from kups.core.assertion import Fix, runtime_assert
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import View
from kups.core.propagator import Propagator
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field


def reserve_slots(n_slots: int, capacity: int) -> Table[SystemId, Index[ParticleId]]:
    """Reserve ``capacity`` particle rows per system, including unused payload rows."""
    if n_slots < 1 or capacity < 1:
        raise ValueError("n_slots and capacity must be positive.")
    return Table.arange(
        Index.integer(
            jnp.arange(n_slots * capacity).reshape(n_slots, capacity),
            n=n_slots * capacity,
            label=ParticleId,
            max_count=1,
        ),
        label=SystemId,
    )


def slot_owners(
    slots: Table[SystemId, Index[ParticleId]],
) -> Table[ParticleId, Index[SystemId]]:
    """Map reserved particle rows to systems, independent of payload occupancy.

    Reservations must be disjoint, with shape ``(n_slots, capacity)``. They need
    not be contiguous; unreserved rows receive an invalid system index.
    """
    rows = slots.data
    owners = Index(
        slots.keys,
        jnp.full(len(rows.keys), len(slots), int),
        rows.indices.shape[1],
        _cls=SystemId,
    )
    return Table(rows.keys, owners).update(
        rows,
        Index(
            slots.keys,
            jnp.broadcast_to(slots.index.indices[:, None], rows.indices.shape),
            rows.indices.shape[1],
            _cls=SystemId,
        ),
    )


@dataclass
class RefillPropagator[State](Propagator[State]):
    """Request host service for selected systems before advancing a batch.

    Compose this BEFORE a ResetOnErrorPropagator wrapping the numerical step.
    A pending request prevents that step from committing; propagate_and_fix
    calls refill on the returned state, then retries. The fix must clear the
    serviced requests, including slots left empty at end of input.

    Payloads stay in state. Only an empty tuple travels through fix_args,
    because assertions in compiled loops reduce their arguments by maximum.
    Use one such composition per host cycle: refill is ordinary host work,
    not an unbounded stream inside propagate_and_fix's bounded retry loop.
    The numerical work may be a bounded LoopPropagator block. Set its repetition
    count to zero while requests are pending to avoid work on repair-only attempts.

    Fix callbacks run on the host, not transactionally: an exception aborts the
    cycle without undoing input consumption or output writes. Prepare and validate
    replacements before committing those effects. Snapshot completed results before
    replacing slots; retain job identities separately from reusable system keys.
    """

    requested: View[State, Table[SystemId, Array]] = field(static=True)
    refill: Fix[State, Table[SystemId, Array]] = field(static=True)

    def _fix(self, state: State, unused: tuple[()]) -> State:
        return self.refill(state, self.requested(state))

    def __call__(self, key: Array, state: State) -> State:
        del key
        runtime_assert(
            ~self.requested(state).data.any(),
            "Batch slots require refill.",
            fix_fn=self._fix,
            fix_args=(),
        )
        return state
