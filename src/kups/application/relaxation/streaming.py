# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Streaming relaxation: slot packing, slot replacement, the step and its runner.

[run_streaming_relax][kups.application.relaxation.streaming.run_streaming_relax]
relaxes a sequence of structures and logs every step. The parts below it take
host input and output through the caller's ``refill`` callback of
[RefillGate][kups.core.stream.RefillGate].
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from kups.application.relaxation.data import (
    RelaxParticles,
    RelaxRunConfig,
    RelaxSystems,
    relax_gradients,
    relax_index_prefix,
    relax_particles,
    relax_systems,
)
from kups.application.relaxation.logging import RelaxStepData
from kups.application.relaxation.simulation import (
    IsRelaxState,
    OptInit,
    make_relax_step,
)
from kups.application.utils.particles import Particles, default_exclusion
from kups.application.utils.propagate import make_cycle_function, run_simulation_cycles
from kups.core import units
from kups.core.cell import AnyPeriodicity, Cell, TriclinicFrame
from kups.core.data import Buffered, Index, Table
from kups.core.data.index import SupportsSorting
from kups.core.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndCellIndex,
)
from kups.core.lens import LambdaLens, Lens, bind, identity_lens
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.patch import IndexLensPatch, Patch
from kups.core.potential import EmptyType, Potential
from kups.core.propagator import (
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
)
from kups.core.storage import EveryNStep, HDF5StorageWriter, WriterGroupConfig
from kups.core.stream import (
    RefillGate,
    reserve_slots,
    slot_owners,
    slot_rows_lens,
    slot_system_index,
)
from kups.core.typing import Label, ParticleId, SystemId
from kups.core.units import catalog as u
from kups.core.utils.jax import dataclass, jit, tree_concat, tree_map
from kups.relaxation.convergence import converged_per_system
from kups.relaxation.optimizer import Resettable

type Structure = tuple[Table[ParticleId, Particles], Cell[AnyPeriodicity]]


class OptReset[OptState](Protocol):
    """Reset selected systems' optimizer state for newly installed particle/cell data."""

    def __call__(
        self,
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
        opt_state: OptState,
        mask: Table[SystemId, Array],
    ) -> OptState: ...


@dataclass
class RelaxPayload:
    """Slot contents, one row per slot or packed structure.

    Leaves of ``particles`` and ``position_gradients`` have shape
    ``(rows, capacity, ...)``; the other leaves have shape ``(rows, ...)``.

    Attributes:
        particles: Particle data of each row.
        position_gradients: Position gradients of each row's particles.
        systems: System data of each row.
        step: Number of optimizer updates of each row's job.
        ordinal: Job ordinal of each row, ``-1`` for an idle row.
    """

    particles: Particles
    position_gradients: units.Force
    systems: RelaxSystems
    step: Array
    ordinal: Array


class IsStreamingRelaxState[OptState](IsRelaxState[OptState], Protocol):
    """Relaxation state whose systems are slots, each running one job at a time.

    ``step`` has one entry per slot: the number of optimizer updates of the
    slot's current job.

    Attributes:
        ordinal: Job ordinal of each slot, ``-1`` while the slot is idle.
        finished: Per-slot refill request, set when a job has converged or used
            its update budget and cleared when the slot is replaced.
    """

    @property
    def ordinal(self) -> Array: ...
    @property
    def finished(self) -> Table[SystemId, Array]: ...


@dataclass
class StreamingRelaxState[OptState]:
    """Slotted relaxation batch.

    Built by
    [make_streaming_relax_state][kups.application.relaxation.streaming.make_streaming_relax_state]:
    ``particles`` holds every slot's reserved rows, padding included, and
    ``systems`` one row per slot.

    Attributes:
        particles: Reserved particle rows of every slot, padding included.
        systems: System data, one row per slot.
        neighborlist_params: Capacity hints for the neighbor list.
        opt_state: Optimizer state.
        step: Per-slot count of the current job's optimizer updates.
        ordinal: Job ordinal of each slot, ``-1`` while the slot is idle.
        finished: Per-slot refill request.
    """

    particles: Table[ParticleId, RelaxParticles]
    systems: Table[SystemId, RelaxSystems]
    neighborlist_params: UniversalNeighborlistParameters
    opt_state: OptState
    step: Array
    ordinal: Array
    finished: Table[SystemId, Array]


def slot_payload_lens[State: IsStreamingRelaxState[object]](
    slots: Table[SystemId, Index[ParticleId]],
) -> Lens[State, RelaxPayload]:
    """Lens onto the batch as one payload row per slot.

    Particle rows go through [slot_rows_lens][kups.core.stream.slot_rows_lens].
    Reservations must be disjoint and have the same system keys as the batch.
    """
    rows: Lens[Table[ParticleId, RelaxParticles], RelaxParticles] = slot_rows_lens(
        slots
    )

    def get(s: State) -> RelaxPayload:
        assert s.systems.keys == slots.keys, (
            "Reservations must match the batch systems."
        )
        p = rows.get(s.particles)
        return RelaxPayload(
            particles=Particles(
                positions=p.positions,
                masses=p.masses,
                atomic_numbers=p.atomic_numbers,
                charges=p.charges,
                labels=p.labels,
                system=p.system,
            ),
            position_gradients=p.position_gradients,
            systems=s.systems.data,
            step=s.step,
            ordinal=s.ordinal,
        )

    def put(s: State, /, value: RelaxPayload) -> State:
        assert s.systems.keys == slots.keys, (
            "Reservations must match the batch systems."
        )
        particles = relax_particles(
            value.particles,
            position_gradients=value.position_gradients,
            exclusion=s.particles[slots.data].exclusion,
        )
        return (
            bind(s)
            .focus(lambda x: (x.particles, x.systems.data, x.step, x.ordinal))
            .set(
                (
                    rows.set(s.particles, particles),
                    value.systems,
                    value.step,
                    value.ordinal,
                )
            )
        )

    return LambdaLens(get, put)


def replace_slots[State, OptState](
    state: State,
    payload: RelaxPayload,
    mask: Table[SystemId, Array],
    *,
    state_lens: Lens[State, IsStreamingRelaxState[OptState]],
    slots: Table[SystemId, Index[ParticleId]],
    reset: OptReset[OptState],
) -> State:
    """Replace the slots selected by ``mask`` with their ``payload`` rows.

    ``payload`` has one row per slot. Resets the optimizer state of the replaced
    slots and clears their refill requests; the other slots keep their rows and
    optimizer state. Pass
    [idle_payload][kups.application.relaxation.streaming.idle_payload] rows once
    input is exhausted.
    """
    payload_lens: Lens[IsStreamingRelaxState[OptState], RelaxPayload] = (
        slot_payload_lens(slots)
    )
    slot_lens = state_lens.nest(payload_lens)
    state = IndexLensPatch(payload, slots.index, slot_lens)(state, mask)
    sub = state_lens.get(state)
    opt_state = reset(sub.particles, sub.systems, sub.opt_state, mask)
    return state_lens.focus(lambda s: (s.opt_state, s.finished.data)).set(
        state,
        (opt_state, jnp.where(mask[sub.finished.index], False, sub.finished.data)),
    )


def payload_from_structures(
    structures: Sequence[Structure],
    capacity: int,
    species: Sequence[Label],
    ordinals: Sequence[int],
) -> RelaxPayload:
    """Pack host structures into payload rows (leading axis ``len(structures)``).

    Pads each structure to ``capacity`` rows, remaps its labels into the fixed
    ``species`` vocabulary and wraps its cell in the relaxation frame with
    ``cell_factor = n_atoms``. Cells must be unbatched and lower triangular, in
    the same coordinate frame as their particles. Each cell is rebuilt with
    [Cell.from_pbc][kups.core.cell.Cell.from_pbc] on a ``TriclinicFrame``, so
    cells that differ only in frame type or cell class can share a stream. All
    structures of a stream, its idle rows included, must share periodicity.

    Raises:
        ValueError: If ``structures`` is empty or not as long as ``ordinals``,
            ``species`` is not sorted and unique, or the structures do not
            share periodicity; or if a structure's cell is batched or not lower
            triangular, its particles are not one complete system, it has more
            atoms than ``capacity``, or it has a label outside ``species``.
    """
    if not structures:
        raise ValueError("Need at least one structure to pack.")
    vocabulary = tuple(species)
    if vocabulary != tuple(sorted(set(vocabulary))):
        raise ValueError(
            f"The species vocabulary {vocabulary} must be sorted and unique."
        )
    periodic = structures[0][1].periodic
    rows: list[RelaxPayload] = []
    for (particles, cell), ordinal in zip(structures, ordinals, strict=True):
        if cell.vectors.shape != (3, 3) or bool(
            jnp.any(jnp.triu(units.magnitudes(cell.vectors), 1) != 0)
        ):
            raise ValueError(f"Structure {ordinal} needs one lower-triangular cell.")
        if cell.periodic != periodic:
            raise ValueError(
                f"Structures must share periodicity: structure {ordinal} has "
                f"{cell.periodic}, the first {periodic}."
            )
        if particles.data.system.num_labels != 1 or not bool(
            jnp.all(particles.data.system.valid_mask)
        ):
            raise ValueError(f"Structure {ordinal} must be one complete system.")
        cell = Cell.from_pbc(TriclinicFrame.from_matrix(cell.vectors), cell.periodic)
        n = len(particles.keys)
        if n > capacity:
            raise ValueError(
                f"Structure {ordinal} has {n} atoms, more than the slot capacity "
                f"{capacity}."
            )
        try:
            relabeled = (
                bind(particles)
                .focus(lambda p: p.data.labels)
                .apply(lambda labels: labels.update_labels(vocabulary))
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Structure {ordinal} has labels outside the species vocabulary "
                f"{vocabulary}."
            ) from error
        padded = Buffered.pad(relabeled, capacity - n).data
        dtype = padded.positions.dtype

        def add_batch_axis[K: SupportsSorting](x: Array | Index[K]) -> Array | Index[K]:
            return x[None]

        rows.append(
            RelaxPayload(
                particles=jax.tree.map(
                    add_batch_axis,
                    bind(padded)
                    .focus(lambda p: p.system)
                    .set(slot_system_index(padded.system.valid_mask)),
                    is_leaf=lambda x: isinstance(x, Index),
                ),
                position_gradients=units.zeros(
                    (1, capacity, 3), u.eV / u.angstrom, dtype
                ),
                systems=relax_systems(cell, jnp.asarray([n], dtype)),
                step=jnp.zeros((1,), int),
                ordinal=jnp.asarray([ordinal], int),
            )
        )

    def strong_dtype(x: Array) -> Array:
        return x.astype(x.dtype)

    # Canonical (strong) dtypes: a weakly typed leaf would change the compiled
    # signature of the state once it is installed in a slot.
    return jax.tree.map(strong_dtype, tree_concat(*rows))


def idle_payload(
    capacity: int,
    species: Sequence[Label],
    periodic: AnyPeriodicity,
    cutoff: units.Length,
    dtype: DTypeLike = float,
) -> RelaxPayload:
    """An unoccupied slot (ordinal -1) in a finite cubic placeholder cell.

    The cell side is ``max(2 * cutoff, 1 Å)``, so a neighbor list sees one image
    per axis and a handful of cells; a degenerate cell would produce NaNs or huge
    candidate counts.
    """
    side = max(2.0 * float(cutoff.to(u.angstrom)), 1.0)
    cell = Cell.from_pbc(
        TriclinicFrame.from_matrix(jnp.eye(3, dtype=dtype) * side * u.angstrom),
        periodic,
    )
    particles: Table[ParticleId, Particles] = Table(
        (),
        Particles(
            positions=units.zeros((0, 3), u.angstrom, dtype),
            masses=units.zeros((0,), u.amu, dtype),
            atomic_numbers=jnp.zeros((0,), int),
            charges=units.zeros((0,), u.e, dtype),
            labels=Index(tuple(species), jnp.zeros((0,), int), _cls=Label),
            system=Index.integer(jnp.zeros((0,), int), n=1, label=SystemId),
        ),
        _cls=ParticleId,
    )
    return payload_from_structures([(particles, cell)], capacity, species, [-1])


def make_streaming_relax_state[OptState](
    slots: Table[SystemId, Index[ParticleId]],
    idle: RelaxPayload,
    opt_init: OptInit[OptState],
    neighborlist_params: UniversalNeighborlistParameters,
) -> StreamingRelaxState[OptState]:
    """Create idle slots requesting their initial refill."""
    n_rows = len(slots.data.keys)

    def repeat_first_row(x: Array) -> Array:
        return jnp.repeat(x[0, :1], n_rows, axis=0)

    particle_data = tree_map(repeat_first_row, idle.particles)
    particles = Table(
        slots.data.keys,
        relax_particles(
            bind(particle_data)
            .focus(lambda p: p.system)
            .set(slot_owners(slots).data.apply_mask(jnp.zeros(n_rows, bool))),
            position_gradients=units.zeros(
                particle_data.positions.shape,
                u.eV / u.angstrom,
                particle_data.positions.dtype,
            ),
            exclusion=default_exclusion(n_rows),
        ),
    )

    def repeat_per_slot(x: Array) -> Array:
        return jnp.repeat(x, len(slots), axis=0)

    systems = Table(slots.keys, tree_map(repeat_per_slot, idle.systems))
    return StreamingRelaxState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        opt_state=opt_init(particles, systems),
        step=jnp.zeros(len(slots), int),
        ordinal=jnp.full(len(slots), -1, int),
        finished=systems.set_data(jnp.ones(len(slots), bool)),
    )


def estimate_neighborlist_params(
    sample: Sequence[Structure],
    capacity: int,
    species: Sequence[Label],
    cutoff: Table[SystemId, units.Length],
    *,
    multiplier: float = 1.5,
) -> UniversalNeighborlistParameters:
    """Neighbor-list capacities for a slotted batch, sized on a representative sample.

    The per-particle averages of
    [UniversalNeighborlistParameters.estimate][kups.core.neighborlist.UniversalNeighborlistParameters.estimate]
    are taken over ``sample`` as if it were one batch and scaled by the sample's
    mean slot occupancy, since padding rows count towards the buffers but
    produce no candidates. Buffer size dominates the per-step cost, so this
    sizes for the typical batch. A batch that overflows the buffers grows them
    through the retry loop at the cost of a recompile; ``multiplier`` sets the
    headroom.

    Raises:
        ValueError: If ``sample`` is empty or
            [payload_from_structures][kups.application.relaxation.streaming.payload_from_structures]
            cannot pack it.
    """
    if not sample:
        raise ValueError("Need at least one structure to size the neighbor list.")
    payload = payload_from_structures(sample, capacity, species, range(len(sample)))
    systems = Table.arange(payload.systems, label=SystemId)
    n_atoms = jnp.asarray([len(p.keys) for p, _ in sample])
    occupancy = float(n_atoms.mean()) / capacity
    return UniversalNeighborlistParameters.estimate(
        Table.arange(n_atoms, label=SystemId),
        systems,
        Table.broadcast_to(cutoff, systems),
        multiplier=multiplier * occupancy,
    )


def make_streaming_relax_propagator[State, OptState, P: Patch[Any]](
    state_lens: Lens[State, IsStreamingRelaxState[OptState]],
    potential: Potential[State, PositionsAndCell[units.Force], EmptyType, P],
    optimizer: Resettable[PositionsAndCell[units.Length], OptState],
    gradient: Lens[Geometry, PositionsAndCell[units.Length]],
    slots: Table[SystemId, Index[ParticleId]],
    *,
    force_tolerance: units.Force,
    max_steps: int,
    include_cell: bool = True,
) -> tuple[Propagator[State], OptInit[OptState], OptReset[OptState]]:
    """Relax occupied slots and flag finished ones for refill.

    A finished slot keeps its last evaluated geometry (up to the gradient
    filter's round-off) with that geometry's energy and gradients. A job that
    reaches ``max_steps`` is evaluated at its final geometry before it
    finishes. Compose [RefillGate][kups.core.stream.RefillGate] before this
    step to service finished slots through
    [propagate_and_fix][kups.core.propagator.propagate_and_fix].

    Args:
        state_lens: Lens onto the streaming sub-state; pass the same lens to
            [replace_slots][kups.application.relaxation.streaming.replace_slots].
        potential: Potential reporting the DOF gradient, built with the same
            ``gradient`` filter.
        optimizer: [Resettable][kups.relaxation.optimizer.Resettable] optimizer
            whose ``reset_layout`` the returned ``reset`` applies.
        gradient: Relaxation filter selecting the optimizer DOFs.
        slots: Disjoint particle rows reserved for each slot, e.g. by
            [reserve_slots][kups.core.stream.reserve_slots]; pass the same table
            to ``replace_slots``,
            [slot_payload_lens][kups.application.relaxation.streaming.slot_payload_lens]
            and
            [make_streaming_relax_state][kups.application.relaxation.streaming.make_streaming_relax_state].
        force_tolerance: A job converges once its largest DOF gradient norm is
            below this; a non-finite gradient fails an unfixable assertion (see
            [converged_per_system][kups.relaxation.convergence.converged_per_system]).
        max_steps: Optimizer updates a job may take before it finishes
            unconverged.
        include_cell: Whether cell gradients count towards convergence.

    Returns:
        ``(propagator, init, reset)``: one optimizer step per running slot that
        flags finished slots and rolls back on a failed assertion, optimizer
        state for ``make_streaming_relax_state``, and the reset of refilled
        slots' optimizer state for ``replace_slots``.

    Raises:
        ValueError: If ``max_steps`` or ``force_tolerance`` is negative, or if
            ``slots`` is not two-dimensional or, when concrete, reserves a row
            twice; see [slot_owners][kups.core.stream.slot_owners].
    """
    if max_steps < 0 or units.magnitudes(force_tolerance) < 0:
        raise ValueError("max_steps and force_tolerance must be non-negative.")

    def finished(s: State) -> Table[SystemId, Array]:
        sub = state_lens.get(s)
        converged = converged_per_system(
            relax_gradients(sub),
            relax_index_prefix(sub.particles, sub.systems),
            force_tolerance,
            include_cell=include_cell,
        )
        return converged.set_data(
            (sub.ordinal >= 0) & (converged.data | (sub.step >= max_steps))
        )

    def accept(s: State) -> Table[SystemId, Array]:
        done = finished(s)
        return done.set_data((state_lens.get(s).ordinal >= 0) & ~done.data)

    owners = slot_owners(slots)
    step, init = make_relax_step(
        state_lens,
        potential,
        optimizer,
        gradient,
        accept=accept,
        index_prefix=lambda particles, systems: PositionsAndCellIndex(
            owners[particles.index], systems.index
        ),
    )

    def count_and_flag(key: Array, s: State) -> State:
        del key
        sub = state_lens.get(s)
        done = finished(s)
        advancing = (sub.ordinal >= 0) & ~done.data
        return state_lens.focus(lambda x: (x.step, x.finished)).set(
            s, (sub.step + advancing.astype(sub.step.dtype), done)
        )

    def reset(
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
        opt_state: OptState,
        mask: Table[SystemId, Array],
    ) -> OptState:
        return optimizer.reset_layout.reset(opt_state, init(particles, systems), mask)

    return (
        ResetOnErrorPropagator(SequentialPropagator((step, count_and_flag))),
        init,
        reset,
    )


@dataclass
class StreamingRelaxStepData(RelaxStepData):
    """Per-step snapshot of every slot for the HDF5 log.

    The inherited fields have one row per slot, and ``atoms`` holds every
    reserved particle row. A structure's last record is its final geometry
    with that geometry's energy and gradients.

    Attributes:
        ordinal: Structure held by each slot, as its position in the input;
            ``-1`` for an idle slot.
        step: Optimizer updates of each slot's structure so far.
    """

    ordinal: Array
    step: Array

    @staticmethod
    def from_slots(state: IsStreamingRelaxState[object]) -> StreamingRelaxStepData:
        """Extract per-step logging data from a streaming relaxation state."""
        data = RelaxStepData.from_state(state)
        return StreamingRelaxStepData(
            atoms=data.atoms,
            potential_energy=data.potential_energy,
            max_force=data.max_force,
            stress_tensor=data.stress_tensor,
            ordinal=state.ordinal,
            step=state.step,
        )


@dataclass
class StreamingRelaxLoggedData:
    """HDF5 writer configuration for streaming relaxation.

    Attributes:
        step: Writer for the per-step snapshot of every slot.
    """

    step: WriterGroupConfig[IsStreamingRelaxState[object], StreamingRelaxStepData] = (
        WriterGroupConfig(StreamingRelaxStepData.from_slots, EveryNStep(1))
    )


def run_streaming_relax[OptState](
    key: Array,
    structures: Sequence[Structure],
    potential: Potential[
        StreamingRelaxState[OptState],
        PositionsAndCell[units.Force],
        EmptyType,
        Patch[Any],
    ],
    cutoff: Table[SystemId, units.Length],
    optimizer: Resettable[PositionsAndCell[units.Length], OptState],
    gradient: Lens[Geometry, PositionsAndCell[units.Length]],
    config: RelaxRunConfig,
    *,
    neighborlist_multiplier: float = 1.5,
) -> StreamingRelaxState[OptState]:
    """Relax ``structures`` in ``config.batch_size`` slots, logging every step.

    A structure takes a free slot in input order and leaves it once it has
    converged to ``config.force_tolerance`` or taken ``config.max_steps``
    updates. Each structure relaxes as it would alone. Slots hold as many rows
    as the largest structure has atoms.

    Args:
        key: JAX PRNG key.
        structures: Structures to relax; see
            [payload_from_structures][kups.application.relaxation.streaming.payload_from_structures]
            for what they must share.
        potential: Potential reporting the DOF gradient, built on an identity
            lens with the same ``gradient`` filter.
        cutoff: The potential's cutoff, which sizes the neighbor list.
        optimizer: Optimizer whose state is reset when a slot changes structure.
        gradient: Relaxation filter selecting the optimizer DOFs.
        config: Run configuration.
        neighborlist_multiplier: Headroom of the neighbor-list buffers.

    Returns:
        The final state, with every slot idle.

    Raises:
        ValueError: If ``structures`` cannot be packed into one stream.
    """
    state_lens = identity_lens(StreamingRelaxState[OptState])
    capacity = max(len(particles) for particles, _ in structures)
    species = sorted(
        {k for particles, _ in structures for k in particles.data.labels.keys}
    )
    slots = reserve_slots(min(config.batch_size, len(structures)), capacity)
    step, opt_init, reset = make_streaming_relax_propagator(
        state_lens,
        potential,
        optimizer,
        gradient,
        slots,
        force_tolerance=config.force_tolerance,
        max_steps=config.max_steps,
        include_cell=config.optimize_cell,
    )
    idle = idle_payload(
        capacity,
        species,
        structures[0][1].periodic,
        units.max(cutoff.data),
        structures[0][0].data.positions.dtype,
    )
    # Packs every structure once, so an invalid one raises before the run.
    neighborlist_params = estimate_neighborlist_params(
        structures, capacity, species, cutoff, multiplier=neighborlist_multiplier
    )
    state = make_streaming_relax_state(slots, idle, opt_init, neighborlist_params)

    @jit
    def install(
        s: StreamingRelaxState[OptState],
        payload: RelaxPayload,
        mask: Table[SystemId, Array],
    ) -> StreamingRelaxState[OptState]:
        return replace_slots(
            s, payload, mask, state_lens=state_lens, slots=slots, reset=reset
        )

    n_started = 0

    def refill(
        s: StreamingRelaxState[OptState], requested: Table[SystemId, Array]
    ) -> StreamingRelaxState[OptState]:
        nonlocal n_started
        ordinal = n_started
        rows: list[RelaxPayload] = []
        for flagged in np.asarray(requested.data):
            if flagged and ordinal < len(structures):
                rows.append(
                    payload_from_structures(
                        [structures[ordinal]], capacity, species, [ordinal]
                    )
                )
                ordinal += 1
            else:
                rows.append(idle)
        s = install(s, tree_concat(*rows), requested)
        n_started = ordinal
        return s

    def n_running(s: StreamingRelaxState[OptState]) -> int:
        return int((np.asarray(s.ordinal) >= 0).sum())

    def progress(s: StreamingRelaxState[OptState]) -> dict[str, str]:
        return {"finished": f"{n_started - n_running(s)}/{len(structures)}"}

    cycle = make_cycle_function(
        SequentialPropagator((RefillGate(lambda s: s.finished, refill), step))
    )
    # Every cycle but the last evaluates at least one structure, and a structure
    # is evaluated once per update and once more to finish.
    n_cycles = len(structures) * (config.max_steps + 1) + 1
    logger = CompositeLogger(
        TqdmLogger(n_cycles, postfix=progress),
        HDF5StorageWriter(config.out_file, StreamingRelaxLoggedData(), state, n_cycles),
    )
    return run_simulation_cycles(
        key,
        cycle,
        state,
        n_cycles,
        logger,
        convergence_fn=lambda s: n_running(s) == 0,
    )
