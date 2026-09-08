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
import optax
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
    OptReset,
    make_relax_step,
)
from kups.application.utils.particles import Particles, default_exclusion
from kups.core.cell import AnyPeriodicity, Cell, TriclinicFrame
from kups.core.data import Buffered, Index, Table
from kups.core.data.slots import SlotLayout
from kups.core.lens import LambdaLens, Lens, bind
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.patch import Patch
from kups.core.potential import EmptyType, Potential
from kups.core.propagator import (
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
)
from kups.core.typing import ExclusionId, Label, ParticleId, SystemId
from kups.core.utils.jax import (
    dataclass,
    tree_concat,
    tree_map,
    tree_where_broadcast_last,
)
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndCellIndex,
)
from kups.relaxation.convergence import converged_per_system
from kups.relaxation.optimizer import Optimizer

type Structure = tuple[Table[ParticleId, Particles], Cell[AnyPeriodicity]]


@dataclass
class RelaxPayload:
    """Slot rows; particle leaves have shape (slots, capacity, ...)."""

    particles: Particles
    position_gradients: Array
    systems: RelaxSystems
    step: Array
    ordinal: Array


class IsStreamingRelaxState(IsRelaxState, Protocol):
    @property
    def slot_ordinal(self) -> Array: ...
    @property
    def finished(self) -> Table[SystemId, Array]: ...


@dataclass
class StreamingRelaxState:
    particles: Table[ParticleId, RelaxParticles]
    systems: Table[SystemId, RelaxSystems]
    neighborlist_params: UniversalNeighborlistParameters
    opt_state: optax.OptState
    step: Array
    slot_ordinal: Array
    finished: Table[SystemId, Array]


def _slotted_system(layout: SlotLayout, valid: Array) -> Index[SystemId]:
    return Index(
        layout.keys,
        jnp.where(valid, 0, layout.n_slots),
        layout.capacity,
        _cls=SystemId,
    )


def _particles_from_payload(
    layout: SlotLayout,
    payload: RelaxPayload,
    exclusion: Index[ExclusionId],
) -> RelaxParticles:
    particles = layout.from_slots(payload.particles)
    return relax_particles(
        bind(particles)
        .focus(lambda p: p.system)
        .set(layout.system_index(payload.particles.system.valid_mask)),
        position_gradients=layout.from_slots(payload.position_gradients),
        exclusion=exclusion,
    )


def slot_payload_lens[State: IsStreamingRelaxState](
    layout: SlotLayout,
) -> Lens[State, RelaxPayload]:
    """Adapt a flat batch to slot rows, preserving its typed label vocabularies."""

    def get(s: State) -> RelaxPayload:
        p = layout.to_slots(s.particles.data)
        return RelaxPayload(
            Particles(
                p.positions,
                p.masses,
                p.atomic_numbers,
                p.charges,
                p.labels,
                _slotted_system(layout, p.system.valid_mask),
            ),
            p.position_gradients,
            s.systems.data,
            s.step,
            s.slot_ordinal,
        )

    def set(s: State, /, value: RelaxPayload) -> State:
        particles = _particles_from_payload(layout, value, s.particles.data.exclusion)
        return (
            bind(s)
            .focus(lambda x: (x.particles.data, x.systems.data, x.step, x.slot_ordinal))
            .set((particles, value.systems, value.step, value.ordinal))
        )

    return LambdaLens(get, set)


def replace_slots[State: IsStreamingRelaxState](
    state: State,
    payload: RelaxPayload,
    mask: Table[SystemId, Array],
    *,
    layout: SlotLayout,
    reset: OptReset,
) -> State:
    """Install selected slot rows, reset their optimizer, and acknowledge refill.

    Supply idle payload rows (ordinal -1) when input is exhausted. Unselected
    slots, including their optimizer history, are preserved.
    """
    slot_lens: Lens[State, RelaxPayload] = slot_payload_lens(layout)
    state = slot_lens.set(
        state, tree_where_broadcast_last(mask.data, payload, slot_lens.get(state))
    )
    opt_state = reset(state.particles, state.systems, state.opt_state, mask)
    return (
        bind(state)
        .focus(lambda s: (s.opt_state, s.finished.data))
        .set(
            (
                opt_state,
                jnp.where(mask.data, False, state.finished.data),
            )
        )
    )


def payload_from_structures(
    structures: Sequence[Structure],
    layout: SlotLayout,
    species: Sequence[Label],
    ordinals: Sequence[int],
) -> RelaxPayload:
    """Pack host structures into payload rows (leading axis ``len(structures)``).

    Each structure is padded to ``capacity`` rows, its labels are remapped into
    the fixed ``species`` vocabulary and its cell is wrapped in the relaxation
    frame with ``cell_factor = n_atoms``.

    Raises:
        ValueError: If a structure has more atoms than ``capacity`` or a label
            outside ``species``.
    """
    vocabulary = tuple(species)
    rows: list[RelaxPayload] = []
    for (particles, cell), ordinal in zip(structures, ordinals, strict=True):
        n = len(particles.keys)
        if n > layout.capacity:
            raise ValueError(
                f"Structure {ordinal} has {n} atoms, more than the slot capacity "
                f"{layout.capacity}."
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
        padded = Buffered.pad(relabeled, layout.capacity - n).data
        dtype = padded.positions.dtype
        rows.append(
            RelaxPayload(
                particles=jax.tree.map(
                    lambda x: x[None],
                    bind(padded)
                    .focus(lambda p: p.system)
                    .set(_slotted_system(layout, padded.system.valid_mask)),
                    is_leaf=lambda x: isinstance(x, Index),
                ),
                position_gradients=jnp.zeros((1, layout.capacity, 3), dtype),
                systems=relax_systems(relax_cell(cell, jnp.asarray([n], dtype))),
                step=jnp.zeros((1,), int),
                ordinal=jnp.asarray([ordinal], int),
            )
        )
    # Canonical (strong) dtypes: a weakly typed leaf would change the compiled
    # signature of the state once it is installed in a slot.
    return jax.tree.map(lambda x: x.astype(x.dtype), tree_concat(*rows))


def idle_payload(
    layout: SlotLayout,
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
    return payload_from_structures([(particles, cell)], layout, species, [-1])


def make_streaming_relax_state(
    layout: SlotLayout,
    idle: RelaxPayload,
    opt_init: OptInit,
    neighborlist_params: UniversalNeighborlistParameters,
) -> StreamingRelaxState:
    """Create idle slots requesting their initial refill."""
    slotted = tree_map(lambda x: jnp.repeat(x, layout.n_slots, axis=0), idle)
    particles = Table.arange(
        _particles_from_payload(layout, slotted, default_exclusion(layout.n_rows)),
        label=ParticleId,
    )
    systems = Table.arange(slotted.systems, label=SystemId)
    return StreamingRelaxState(
        particles,
        systems,
        neighborlist_params,
        opt_init(particles, systems),
        jnp.zeros(layout.n_slots, int),
        jnp.full(layout.n_slots, -1, int),
        systems.set_data(jnp.ones(layout.n_slots, bool)),
    )


def estimate_stream_neighborlist(
    sample: Sequence[Structure],
    layout: SlotLayout,
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
    single = SlotLayout(n_slots=len(sample), capacity=layout.capacity)
    payload = payload_from_structures(sample, single, species, range(len(sample)))
    systems = Table.arange(payload.systems, label=SystemId)
    n_atoms = jnp.asarray([len(p.keys) for p, _ in sample])
    occupancy = float(n_atoms.mean()) / layout.capacity
    return UniversalNeighborlistParameters.estimate(
        Table.arange(n_atoms, label=SystemId),
        systems,
        Table.broadcast_to(cutoff, systems),
        multiplier=multiplier * occupancy,
    )


def make_streaming_relax_propagator[
    State,
    Substate: IsStreamingRelaxState,
    OptState,
    P: Patch[Any],
](
    state_lens: Lens[State, Substate],
    potential: Potential[State, PositionsAndCell, EmptyType, P],
    optimizer: Optimizer[PositionsAndCell, OptState],
    gradient: Lens[Geometry, PositionsAndCell],
    layout: SlotLayout,
    *,
    force_tolerance: float,
    max_steps: int,
    include_cell: bool = True,
) -> tuple[Propagator[State], OptInit, OptReset]:
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

    step, init, reset = make_relax_step(
        state_lens,
        potential,
        optimizer,
        gradient,
        accept=accept,
        index_prefix=lambda particles, systems: PositionsAndCellIndex(
            layout.slot_of_row, systems.index
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

    def initialize(
        particles: Table[ParticleId, RelaxParticles],
        systems: Table[SystemId, RelaxSystems],
    ) -> optax.OptState:
        opt_state = init(particles, systems)
        # Validate reset support before any structures are consumed.
        return reset(
            particles,
            systems,
            opt_state,
            systems.set_data(jnp.zeros(len(systems), bool)),
        )

    return (
        ResetOnErrorPropagator(SequentialPropagator((step, count))),
        initialize,
        reset,
    )
