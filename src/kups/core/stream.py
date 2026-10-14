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
- **[slot_owners][kups.core.stream.slot_owners]**: Maps each reserved row to
  its slot, including padding rows, which a particle's ``system`` index marks
  out of bounds. Per-slot state built on it, such as an optimizer's per-system
  reductions and resets, also covers rows a future system will occupy.
- **[slot_rows_lens][kups.core.stream.slot_rows_lens]**: Reads each slot's
  rows with a slot-local system index
  ([slot_system_index][kups.core.stream.slot_system_index]) and writes them
  back to the slots.
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
from kups.core.lens import LambdaLens, Lens, View, bind
from kups.core.propagator import Propagator
from kups.core.typing import HasSystemIndex, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, is_traced


def reserve_slots(n_slots: int, capacity: int) -> Table[SystemId, Index[ParticleId]]:
    """Reserve particle rows ``[i * capacity, (i + 1) * capacity)`` for slot ``i``."""
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
    """Map each reserved particle row to its slot, occupied or not.

    Reservations have shape ``(n_slots, capacity)`` and must be disjoint. They
    need not be contiguous; out-of-bounds entries reserve no row. Unreserved
    rows map to an out-of-bounds slot index.

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


def slot_system_index(valid: Array) -> Index[SystemId]:
    """System index of slot payload rows, which do not name their slot.

    Valid rows reference the single key ``SystemId(0)``; the others are out of
    bounds.

    Args:
        valid: Which rows hold particles, any shape.

    Returns:
        An index of the same shape over the keys ``(SystemId(0),)``.
    """
    return Index.zeros(valid.shape, label=SystemId).apply_mask(valid)


def slot_rows_lens[P: HasSystemIndex](
    slots: Table[SystemId, Index[ParticleId]],
) -> Lens[Table[ParticleId, P], P]:
    """Lens onto each slot's reserved rows, with slot-local system indices.

    ``get`` gathers ``particles[slots.data]``, shape ``(n_slots, capacity, ...)``,
    and sets its system index to
    [slot_system_index][kups.core.stream.slot_system_index] of the rows'
    validity. ``set`` points each valid row at the slot that reserves it, as
    [slot_owners][kups.core.stream.slot_owners] does, marks padding rows out of
    bounds, and writes all of them with ``particles.update(slots.data, ...)``.
    Rows outside every reservation are left as they are.

    The particles' system index must be keyed by ``slots.keys``. ``set``
    increases its ``max_count`` to at least the capacity; an index built from
    ``slot_owners(slots).data`` already has that bound, so ``set`` keeps the
    table's pytree structure, as a loop carry needs.

    Raises:
        ValueError: When the lens is built, if the reservations are invalid
            (see ``slot_owners``); on ``get`` or ``set``, if the particles'
            system index is not keyed by ``slots.keys``.
    """
    owners = slot_owners(slots)[slots.data]

    def check(particles: Table[ParticleId, P]) -> None:
        if particles.data.system.keys != slots.keys:
            raise ValueError("The particles' system index must be keyed by the slots.")

    def get(particles: Table[ParticleId, P]) -> P:
        check(particles)
        rows = particles[slots.data]
        return (
            bind(rows)
            .focus(lambda p: p.system)
            .set(slot_system_index(rows.system.valid_mask))
        )

    def put(particles: Table[ParticleId, P], /, value: P) -> Table[ParticleId, P]:
        check(particles)
        rows = (
            bind(value)
            .focus(lambda p: p.system)
            .set(owners.apply_mask(value.system.valid_mask))
        )
        return particles.update(slots.data, rows)

    return LambdaLens(get, put)


@dataclass
class RefillGate[State](Propagator[State]):
    """Ask the host to refill slots through a failing runtime assertion.

    Leaves the state unchanged. Each call asserts that no slot is flagged in
    ``requested``. While any slot is, the assertion fails, its fix calls
    ``refill(state, requested(state))`` on the host, and ``propagate_and_fix``
    retries the cycle. ``refill`` must clear every flag, including those of
    slots it leaves idle when the input is exhausted; otherwise every retry
    fails again until ``propagate_and_fix`` runs out of tries.

    Compose it at cycle level, before the step's ``ResetOnErrorPropagator``::

        SequentialPropagator((RefillGate(...), ResetOnErrorPropagator(step)))

    A pending request then makes ``ResetOnErrorPropagator`` roll ``step`` back,
    and the retry runs ``step`` on the refilled slots. For a bounded block of
    steps per cycle, loop the wrapped step,
    ``LoopPropagator(ResetOnErrorPropagator(step), reps)``. Each iteration then
    rolls back on its own, so steps committed before a capacity failure survive
    its repair. Give the loop zero repetitions while a request is pending:
    every iteration of that attempt rolls back.

    Constraints:

    - One refill per host cycle, since ``propagate_and_fix`` bounds its
      retries. A request raised by ``step`` after a refill waits for the next
      cycle. Keep the gate outside any ``LoopPropagator``: inside one, a later
      iteration's request fails the retry again.
    - One cycle per attempt. A driver that rolls back several cycles as one
      unit, such as a scanned block checked once at its end, can discard the
      cycle that raised a request; the fix then finds no flagged slot and
      raises ``RuntimeError``.
    - ``refill`` is not transactional. If it raises, the cycle aborts without
      undoing consumed input or written output. Validate replacements before
      committing them, copy finished results out before overwriting a slot,
      and track jobs by their own identity, since slot keys are reused.

    Attributes:
        requested: View returning one boolean per slot; ``True`` requests a
            refill of that slot.
        refill: Host callback ``(state, requested) -> state`` that refills the
            flagged slots.
    """

    requested: View[State, Table[SystemId, Array]] = field(static=True)
    refill: Fix[State, Table[SystemId, Array]] = field(static=True)

    def _fix(self, state: State, unused: tuple[()]) -> State:
        # Read the mask from the state being repaired, not from fix_args: the
        # driver may have rolled that state back, and assertion.while_handler
        # max-reduces fix_args from a -inf sentinel, which sets every flag.
        requested = self.requested(state)
        if not bool(requested.data.any()):
            raise RuntimeError(
                "Refill was requested, but no slot is flagged in the state passed "
                "to the fix: the cycle that raised the request was rolled back. "
                "Run the refill gate one cycle per attempt."
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
