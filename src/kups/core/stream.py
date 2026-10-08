# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Fixed particle-row slots for streaming batches, and host refill requests.

A streaming application keeps a fixed number of *slots* on the device, one
system per slot, and replaces a slot's system from the host once it is
finished. Shapes are static under JIT, so every slot reserves a fixed set of
particle rows up front; a smaller system leaves the remaining rows as padding.
Slots are ordinary ``Table``/``Index`` relations: gather a slotted view with
``particles[slots.data]`` and write it back with
``particles.update(slots.data, ...)``.

- **[reserve_slots][kups.core.stream.reserve_slots]**: Reserves one contiguous
  block of ``capacity`` rows per slot.
- **[slot_owners][kups.core.stream.slot_owners]**: Maps each row back to the
  slot that reserves it. Unlike a particle's ``system`` index, which marks
  padding rows out of bounds, it does not depend on occupancy, so per-slot
  state such as an optimizer's per-system reductions and resets also covers
  rows a future system will occupy.
- **[RefillGate][kups.core.stream.RefillGate]**: Lets compiled code
  ask the host for new systems through a failing
  [runtime_assert][kups.core.assertion.runtime_assert]; its fix runs the
  caller's refill callback before ``propagate_and_fix`` retries the cycle.

Nothing here depends on relaxation; payloads and request flags live in the
caller's state.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.assertion import Fix, runtime_assert
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import View
from kups.core.propagator import Propagator
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, is_traced


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

    Raises:
        ValueError: If reservations are not two-dimensional or, when concrete,
            reserve a row more than once. Traced reservations are not checked
            for overlap.
    """
    rows = slots.data
    if rows.ndim != 2:
        raise ValueError("Reservations must have shape (n_slots, capacity).")
    if not is_traced(rows.indices):
        reserved = np.asarray(rows.indices)
        reserved = np.sort(reserved[reserved < len(rows.keys)])
        if (reserved[1:] == reserved[:-1]).any():
            raise ValueError("Slot reservations must be disjoint.")
    no_owner = len(slots)  # Index's out-of-bounds sentinel: see Index.valid_mask
    owners = Index(
        slots.keys,
        jnp.full(len(rows.keys), no_owner, int),
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
class RefillGate[State](Propagator[State]):
    """Ask the host to refill slots, through a failing runtime assertion.

    Does no numerical work. Each call asserts that no slot is flagged in
    ``requested``. While any slot is, the assertion fails and its fix runs
    ``refill(state, requested(state))`` on the host; ``propagate_and_fix`` then
    retries the cycle. ``refill`` must install new systems and clear every flag
    it serviced, including for slots it leaves idle at end of input; otherwise
    each retry fails again until ``propagate_and_fix`` runs out of tries.

    Compose it at cycle level, before the numerical step's error recovery::

        SequentialPropagator((RefillGate(...), ResetOnErrorPropagator(step)))

    A pending request then fails the cycle before ``step`` commits:
    ``ResetOnErrorPropagator`` rolls ``step`` back, the host refills, and the
    retry runs ``step`` on the new systems. For a bounded block per cycle, loop
    the wrapped step, ``LoopPropagator(ResetOnErrorPropagator(step), reps)``:
    each iteration then rolls back on its own, so iterations committed before
    a capacity failure survive its repair. Give the loop zero repetitions while
    a request is pending, since every iteration of a repair-only attempt is
    rolled back.

    Constraints, and why:

    - One refill per host cycle. ``propagate_and_fix`` bounds its retries, so
      refill is a single repair, not a loop. A request raised by the refilled
      step is serviced next cycle. For the same reason the gate belongs outside
      any ``LoopPropagator``: there, a request raised by a later iteration
      fails the retry again.
    - A request must survive to the end of the attempt. A driver that rolls
      back several cycles as one unit (a scanned block checked once at the
      end) discards the step that raised it, so the fix would find no flagged
      slot; it raises ``RuntimeError`` instead of retrying in vain. Run the
      gate one cycle per attempt.
    - The request mask is re-read from state rather than carried in
      ``fix_args`` (an empty tuple). State is the source of truth, and
      assertion payloads leaving a ``while`` loop such as ``LoopPropagator``
      are reduced by elementwise maximum (see ``assertion.while_handler``),
      which turns a boolean mask all ``True``.
    - ``refill`` runs on the host and is not transactional. If it raises, the
      cycle aborts without undoing input consumed or output written. Validate
      replacements before committing them, snapshot completed results before
      overwriting a slot, and keep job identities separate from the reusable
      slot keys.

    Attributes:
        requested: View returning one boolean per slot; ``True`` requests new
            work for that slot.
        refill: Host callback ``(state, requested) -> state`` that services the
            flagged slots and clears their flags.
    """

    requested: View[State, Table[SystemId, Array]] = field(static=True)
    refill: Fix[State, Table[SystemId, Array]] = field(static=True)

    def _fix(self, state: State, unused: tuple[()]) -> State:
        requested = self.requested(state)
        if not bool(requested.data.any()):
            raise RuntimeError(
                "Refill was requested, but no slot is flagged in the state passed "
                "to the fix: the cycle that raised the request was rolled back. "
                "Run the refill gate one cycle per propagate_and_fix attempt."
            )
        return self.refill(state, requested)

    def __call__(self, key: Array, state: State) -> State:
        del key
        runtime_assert(
            ~self.requested(state).data.any(),
            "Batch slots require refill.",
            fix_fn=self._fix,
            fix_args=(),
        )
        return state
