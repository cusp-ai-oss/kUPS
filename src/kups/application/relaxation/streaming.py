# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pure slot packing, replacement, and batched relaxation.

Host input and output are supplied by the caller through RefillPropagator's
fix callback. No device queue, file formats, or background workers live here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from kups.application.relaxation.data import (
    RelaxParticles,
    RelaxSystems,
    relax_cell,
    relax_gradients,
    relax_index_prefix,
    relax_particles,
    relax_systems,
)
from kups.application.relaxation.simulation import (
    IsRelaxState,
    OptInit,
    make_relax_step,
)
from kups.application.utils.particles import Particles, default_exclusion
from kups.core.cell import AnyPeriodicity, Cell, TriclinicFrame
from kups.core.data import Buffered, Index, Table
from kups.core.lens import LambdaLens, Lens, bind
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.patch import IndexLensPatch, Patch
from kups.core.potential import EmptyType, Potential
from kups.core.propagator import (
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
)
from kups.core.stream import slot_owners
from kups.core.typing import Label, ParticleId, SystemId
from kups.core.utils.jax import dataclass, tree_concat, tree_map
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndCellIndex,
)
from kups.relaxation.convergence import converged_per_system
from kups.relaxation.optimizer import Optimizer, ResetLayout

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
    """Slot rows; particle leaves have shape (slots, capacity, ...)."""

    particles: Particles
    position_gradients: Array
    systems: RelaxSystems
    step: Array
    ordinal: Array


class IsStreamingRelaxState[OptState](IsRelaxState[OptState], Protocol):
    @property
    def slot_ordinal(self) -> Array: ...
    @property
    def finished(self) -> Table[SystemId, Array]: ...


@dataclass
class StreamingRelaxState[OptState]:
    particles: Table[ParticleId, RelaxParticles]
    systems: Table[SystemId, RelaxSystems]
    neighborlist_params: UniversalNeighborlistParameters
    opt_state: OptState
    step: Array
    slot_ordinal: Array
    finished: Table[SystemId, Array]


def _slotted_system(valid: Array) -> Index[SystemId]:
    """Payload rows use one local system key, independent of their destination."""
    return Index(
        (SystemId(0),),
        jnp.where(valid, 0, 1),
        _cls=SystemId,
    )


def slot_payload_lens[State: IsStreamingRelaxState[object]](
    slots: Table[SystemId, Index[ParticleId]],
) -> Lens[State, RelaxPayload]:
    """Gather/scatter particle rows for the batch's systems, preserving vocabularies.

    Reservations must be disjoint and have the same system keys as the batch.
    """

    owners = slot_owners(slots)[slots.data]

    def get(s: State) -> RelaxPayload:
        assert s.systems.keys == slots.keys, (
            "Reservations must match the batch systems."
        )
        p = s.particles[slots.data]
        return RelaxPayload(
            particles=Particles(
                positions=p.positions,
                masses=p.masses,
                atomic_numbers=p.atomic_numbers,
                charges=p.charges,
                labels=p.labels,
                system=_slotted_system(p.system.valid_mask),
            ),
            position_gradients=p.position_gradients,
            systems=s.systems.data,
            step=s.step,
            ordinal=s.slot_ordinal,
        )

    def set(s: State, /, value: RelaxPayload) -> State:
        assert s.systems.keys == slots.keys, (
            "Reservations must match the batch systems."
        )
        particles = relax_particles(
            bind(value.particles)
            .focus(lambda p: p.system)
            .set(owners.apply_mask(value.particles.system.valid_mask)),
            position_gradients=value.position_gradients,
            exclusion=s.particles[slots.data].exclusion,
        )
        return (
            bind(s)
            .focus(lambda x: (x.particles, x.systems.data, x.step, x.slot_ordinal))
            .set(
                (
                    s.particles.update(slots.data, particles),
                    value.systems,
                    value.step,
                    value.ordinal,
                )
            )
        )

    return LambdaLens(get, set)


def replace_slots[State, OptState](
    state: State,
    payload: RelaxPayload,
    mask: Table[SystemId, Array],
    *,
    state_lens: Lens[State, IsStreamingRelaxState[OptState]],
    slots: Table[SystemId, Index[ParticleId]],
    reset: OptReset[OptState],
) -> State:
    """Install selected slot rows, reset their optimizer, and acknowledge refill.

    Supply idle payload rows (ordinal -1) when input is exhausted. Unselected
    slots, including their optimizer history, are preserved.
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

    Each structure is padded to ``capacity`` rows, its labels are remapped into
    the fixed ``species`` vocabulary and its cell is wrapped in the relaxation
    frame with ``cell_factor = n_atoms``. Cells must be unbatched and lower
    triangular, in the same coordinate frame as their particles. Plain frames
    are promoted to TriclinicFrame so orthogonal and triclinic jobs can mix.

    Raises:
        ValueError: If a structure has more atoms than ``capacity`` or a label
            outside ``species``.
    """
    vocabulary = tuple(species)
    rows: list[RelaxPayload] = []
    for (particles, cell), ordinal in zip(structures, ordinals, strict=True):
        if cell.vectors.shape != (3, 3) or bool(
            jnp.any(jnp.triu(cell.vectors, 1) != 0)
        ):
            raise ValueError("Each structure requires one lower-triangular cell.")
        if particles.data.system.num_labels != 1 or not bool(
            jnp.all(particles.data.system.valid_mask)
        ):
            raise ValueError("Each structure must contain one complete system.")
        cell = (
            bind(cell)
            .focus(lambda c: c.frame)
            .set(TriclinicFrame.from_matrix(cell.vectors))
        )
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
        rows.append(
            RelaxPayload(
                particles=jax.tree.map(
                    lambda x: x[None],
                    bind(padded)
                    .focus(lambda p: p.system)
                    .set(_slotted_system(padded.system.valid_mask)),
                    is_leaf=lambda x: isinstance(x, Index),
                ),
                position_gradients=jnp.zeros((1, capacity, 3), dtype),
                systems=relax_systems(relax_cell(cell, jnp.asarray([n], dtype))),
                step=jnp.zeros((1,), int),
                ordinal=jnp.asarray([ordinal], int),
            )
        )
    # Canonical (strong) dtypes: a weakly typed leaf would change the compiled
    # signature of the state once it is installed in a slot.
    return jax.tree.map(lambda x: x.astype(x.dtype), tree_concat(*rows))


def idle_payload(
    capacity: int,
    species: Sequence[Label],
    periodic: AnyPeriodicity,
    cutoff: float,
    dtype: DTypeLike = float,
) -> RelaxPayload:
    """An unoccupied slot (ordinal -1) in a finite cubic placeholder cell.

    The cell side is ``2 * cutoff`` so a neighbor list sees one image per axis
    and a handful of cells; a degenerate cell would produce NaNs or huge
    candidate counts.
    """
    side = max(2.0 * float(cutoff), 1.0)
    cell = Cell.from_pbc(
        TriclinicFrame.from_matrix(jnp.eye(3, dtype=dtype) * side), periodic
    )
    particles: Table[ParticleId, Particles] = Table(
        (),
        Particles(
            positions=jnp.zeros((0, 3), dtype),
            masses=jnp.zeros((0,), dtype),
            atomic_numbers=jnp.zeros((0,), int),
            charges=jnp.zeros((0,), dtype),
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
    particle_data = tree_map(
        lambda x: jnp.repeat(x[0, :1], n_rows, axis=0), idle.particles
    )
    particles = Table(
        slots.data.keys,
        relax_particles(
            bind(particle_data)
            .focus(lambda p: p.system)
            .set(slot_owners(slots).data.apply_mask(jnp.zeros(n_rows, bool))),
            position_gradients=jnp.zeros_like(particle_data.positions),
            exclusion=default_exclusion(n_rows),
        ),
    )
    systems = Table(
        slots.keys, tree_map(lambda x: jnp.repeat(x, len(slots), axis=0), idle.systems)
    )
    return StreamingRelaxState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        opt_state=opt_init(particles, systems),
        step=jnp.zeros(len(slots), int),
        slot_ordinal=jnp.full(len(slots), -1, int),
        finished=systems.set_data(jnp.ones(len(slots), bool)),
    )


def estimate_stream_neighborlist(
    sample: Sequence[Structure],
    capacity: int,
    species: Sequence[Label],
    cutoff: Table[SystemId, Array],
    *,
    multiplier: float = 1.5,
) -> UniversalNeighborlistParameters:
    """Neighbor-list capacities for a slotted batch, sized on a representative sample.

    The per-particle averages of
    [UniversalNeighborlistParameters.estimate][kups.core.neighborlist.UniversalNeighborlistParameters.estimate]
    are taken over ``sample`` as if it were one batch and scaled by the sample's
    mean slot occupancy, since padding rows count towards the buffers but
    produce no candidates. Buffer size is the dominant per-step cost, so this
    sizes for the typical batch; an unusually demanding batch grows the
    capacities through the retry loop at the price of a recompile, which
    ``multiplier`` trades against.
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


def make_streaming_relax_propagator[
    State,
    OptState,
    ResetData,
    ResetIndices,
    P: Patch[Any],
](
    state_lens: Lens[State, IsStreamingRelaxState[OptState]],
    potential: Potential[State, PositionsAndCell, EmptyType, P],
    optimizer: Optimizer[PositionsAndCell, OptState],
    gradient: Lens[Geometry, PositionsAndCell],
    slots: Table[SystemId, Index[ParticleId]],
    *,
    reset_layout: ResetLayout[OptState, ResetData, ResetIndices],
    force_tolerance: float,
    max_steps: int,
    include_cell: bool = True,
) -> tuple[Propagator[State], OptInit[OptState], OptReset[OptState]]:
    """Relax occupied slots; completed slots retain their evaluated geometry.

    Compose RefillPropagator before this step to service finished slots through
    propagate_and_fix. A final evaluation after the last allowed update ensures
    that budget-exhausted results also carry consistent energy and gradients.
    """
    if max_steps < 0 or force_tolerance < 0:
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
            (sub.slot_ordinal >= 0) & (converged.data | (sub.step >= max_steps))
        )

    def accept(s: State) -> Table[SystemId, Array]:
        done = finished(s)
        return done.set_data((state_lens.get(s).slot_ordinal >= 0) & ~done.data)

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

    def count(key: Array, s: State) -> State:
        del key
        sub = state_lens.get(s)
        done = finished(s)
        advancing = (sub.slot_ordinal >= 0) & ~done.data
        return state_lens.focus(lambda x: (x.step, x.finished)).set(
            s, (sub.step + advancing.astype(sub.step.dtype), done)
        )

    def reset(
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
        opt_state: OptState,
        mask: Table[SystemId, Array],
    ) -> OptState:
        fresh = init(particles, systems)
        return IndexLensPatch(
            reset_layout.fields.get(fresh),
            reset_layout.system_index(opt_state),
            reset_layout.fields,
        )(opt_state, mask)

    return (
        ResetOnErrorPropagator(SequentialPropagator((step, count))),
        init,
        reset,
    )
