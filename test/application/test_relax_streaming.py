# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Streaming relaxation through assertion fixes and pure slot replacement."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_type

import ase.build
import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest
from jax import Array

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_ONLY
from kups.application.relaxation.simulation import OptInit
from kups.application.relaxation.streaming import (
    OptReset,
    RelaxPayload,
    StreamingRelaxState,
    Structure,
    estimate_stream_neighborlist,
    idle_payload,
    make_streaming_relax_propagator,
    make_streaming_relax_state,
    payload_from_structures,
    replace_slots,
    slot_payload_lens,
)
from kups.application.utils.particles import particles_from_ase
from kups.application.utils.propagate import make_cycle_function
from kups.core.cell import OrthogonalFrame
from kups.core.data import Index, Table
from kups.core.data.index import SupportsSorting
from kups.core.lens import Lens, bind, const_lens, identity_lens, lens
from kups.core.patch import Patch
from kups.core.potential import EmptyType, Potential
from kups.core.propagator import (
    LoopPropagator,
    Propagator,
    SequentialPropagator,
    propagate_and_fix,
)
from kups.core.result import as_result_function
from kups.core.stream import RefillPropagator, reserve_slots
from kups.core.typing import Label, ParticleId, SystemId
from kups.core.utils.jax import dataclass as state_dataclass
from kups.core.utils.jax import jit, tree_concat, tree_map
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.potential.common.geometry import PositionsAndCell, PositionsAndCellIndex
from kups.relaxation.optimizer import ChainOptimizer, ChainOptState, ResetLayout, chain
from kups.relaxation.transforms import MaxStepSize, ScaleByAseLbfgs, ScaleByFire
from kups.relaxation.transforms.fire import (
    FireReset,
    ScaleByFireState,
    fire_reset_layout,
)
from kups.relaxation.transforms.lbfgs import LbfgsResetIndices, lbfgs_reset_layout

SPECIES = (Label("Ar"),)
CAPACITY = 8
type Batch = StreamingRelaxState[ChainOptState]
type ResetIndices = (
    tuple[()]
    | FireReset[PositionsAndCellIndex, Index[SupportsSorting]]
    | LbfgsResetIndices
)
LJ = LennardJonesParameters.from_dict(
    cutoff=5.0, parameters={"Ar": (3.405, 0.010326)}, mixing_rule="lorentz_berthelot"
)


def _structures() -> list[Structure]:
    structures: list[Structure] = []
    for i, mult in enumerate(((1, 1, 1), (2, 1, 1), (1, 2, 1))):
        atoms = ase.build.bulk("Ar", "fcc", a=5.3, cubic=True) * mult
        atoms.rattle(0.1, seed=i)
        p, cell, _ = particles_from_ase(atoms)
        structures.append((p, cell))
    return structures


@dataclass
class Setup:
    slots: Table[SystemId, Index[ParticleId]]
    idle: RelaxPayload
    step: Propagator[Batch]
    state: Batch
    reset: OptReset[ChainOptState]
    potential: Potential[Batch, PositionsAndCell, EmptyType, Patch[Batch]]


def _setup(
    structures: Sequence[Structure],
    slots: int,
    max_steps: int,
    tolerance: float,
    optimizer: str,
    optimize_cell: bool = False,
    reservations: Table[SystemId, Index[ParticleId]] | None = None,
) -> Setup:
    if reservations is None:
        reservations = reserve_slots(slots, CAPACITY)
    state_lens = identity_lens(StreamingRelaxState[ChainOptState])
    gradient = FRECHET_FILTER if optimize_cell else POSITIONS_ONLY
    parameters = (
        bind(LJ)
        .focus(lambda p: p.cutoff)
        .set(Table(reservations.keys, jnp.full(slots, 5.0)))
    )
    potential = make_lennard_jones_from_state(
        state_lens, parameters=parameters, gradient=gradient
    )
    idle = idle_payload(CAPACITY, SPECIES, (True, True, True), 5.0)
    opt: ChainOptimizer[PositionsAndCell] = chain(optax.scale(-0.1))
    reset_layout: ResetLayout[ChainOptState, object, ResetIndices] = ResetLayout(
        fields=const_lens(()), system_index=lambda s: ()
    )
    if optimizer == "lbfgs":
        opt = chain(
            ScaleByAseLbfgs[PositionsAndCell](memory_size=3),
            MaxStepSize[PositionsAndCell](0.2),
            optax.scale(-1.0),
        )
        lbfgs = lbfgs_reset_layout()
        reset_layout = ResetLayout(
            fields=lens(lambda s: s[0]).nest(lbfgs.fields),
            system_index=lambda s: lbfgs.system_index(s[0]),
        )
    elif optimizer == "fire":
        opt = chain(optax.scale(-1.0), ScaleByFire[PositionsAndCell](dt_start=0.1))
        fire = fire_reset_layout()
        reset_layout = ResetLayout(
            fields=lens(lambda s: s[1]).nest(fire.fields),
            system_index=lambda s: fire.system_index(s[1]),
        )
    step, init, reset = make_streaming_relax_propagator(
        state_lens,
        potential,
        opt,
        gradient,
        reservations,
        reset_layout=reset_layout,
        force_tolerance=tolerance,
        max_steps=max_steps,
        include_cell=optimize_cell,
    )
    nlp = estimate_stream_neighborlist(
        structures, CAPACITY, SPECIES, LJ.cutoff, multiplier=2.0
    )
    return Setup(
        reservations,
        idle,
        step,
        make_streaming_relax_state(reservations, idle, init, nlp),
        reset,
        potential,
    )


def _run(
    structures: Sequence[Structure],
    *,
    slots: int = 2,
    max_steps: int = 3,
    tolerance: float = 0.0,
    optimizer: str = "sgd",
    block_size: int = 1,
    optimize_cell: bool = False,
    reservations: Table[SystemId, Index[ParticleId]] | None = None,
) -> tuple[list[RelaxPayload], Setup]:
    setup = _setup(
        structures, slots, max_steps, tolerance, optimizer, optimize_cell, reservations
    )
    next_ordinal = 0
    completed: list[RelaxPayload] = []
    payload_lens: Lens[Batch, RelaxPayload] = slot_payload_lens(setup.slots)

    @jit
    def install(s: Batch, payload: RelaxPayload, mask: Table[SystemId, Array]) -> Batch:
        return replace_slots(
            s,
            payload,
            mask,
            state_lens=identity_lens(StreamingRelaxState[ChainOptState]),
            slots=setup.slots,
            reset=setup.reset,
        )

    snapshot = jit(payload_lens.get)

    def refill(s: Batch, mask: Table[SystemId, Array]) -> Batch:
        nonlocal next_ordinal
        ordinal = next_ordinal
        requested, old = jax.device_get((mask.data, snapshot(s)))
        rows: list[RelaxPayload] = []
        emitted: list[RelaxPayload] = []
        for i in range(slots):
            if requested[i]:
                if int(old.ordinal[i]) >= 0:
                    emitted.append(tree_map(lambda x: x[i : i + 1].copy(), old))
                if ordinal < len(structures):
                    rows.append(
                        payload_from_structures(
                            [structures[ordinal]], CAPACITY, SPECIES, [ordinal]
                        )
                    )
                    ordinal += 1
                else:
                    rows.append(setup.idle)
            else:
                rows.append(setup.idle)  # replace_slots preserves unselected rows.
        result = jax.block_until_ready(install(s, tree_concat(*rows), mask))
        next_ordinal = ordinal
        completed.extend(emitted)
        return result

    gate: RefillPropagator[Batch] = RefillPropagator(lambda s: s.finished, refill)

    def repetitions(s: Batch) -> Array:
        run = ~s.finished.data.any() & (s.slot_ordinal >= 0).any()
        return jnp.where(run, block_size, 0)

    cycle = make_cycle_function(
        SequentialPropagator((gate, LoopPropagator(setup.step, repetitions)))
    )
    for i in range(40):
        setup.state = propagate_and_fix(cycle, jax.random.key(i), setup.state)
        if bool(jnp.all(setup.state.slot_ordinal < 0)):
            break
    else:
        pytest.fail("Stream did not exhaust.")
    return completed, setup


@pytest.mark.parametrize("max_steps,tolerance", [(0, 0.0), (3, 0.0), (3, 100.0)])
@pytest.mark.parametrize("block_size", [1, 8])
def test_refill_exhaustion_and_evaluated_results(
    max_steps: int, tolerance: float, block_size: int
) -> None:
    structures = _structures()
    rows, setup = _run(
        structures, max_steps=max_steps, tolerance=tolerance, block_size=block_size
    )
    assert sorted(int(row.ordinal[0]) for row in rows) == [0, 1, 2]
    assert not bool(setup.state.finished.data.any())
    assert not bool(setup.state.particles.data.system.valid_mask.any())
    payload_lens: Lens[Batch, RelaxPayload] = slot_payload_lens(setup.slots)
    evaluate = jit(as_result_function(lambda s: setup.potential(s).data))
    for row in rows:
        assert int(row.step[0]) == (0 if tolerance else max_steps)
        state = payload_lens.set(
            setup.state, jax.device_put(tree_concat(row, setup.idle))
        )
        result = evaluate(state)
        result.raise_assertion()
        npt.assert_allclose(
            row.systems.potential_energy[0],
            result.value.total_energies.data[0],
            atol=1e-12,
        )
        npt.assert_allclose(
            row.position_gradients[0],
            result.value.gradients.positions.data[:CAPACITY],
            atol=1e-12,
        )
        if max_steps == 0 or tolerance:
            original = structures[int(row.ordinal[0])][0].data.positions
            npt.assert_array_equal(
                row.particles.positions[0, : len(original)], original
            )


@pytest.mark.parametrize("optimizer", ["sgd", "fire", "lbfgs"])
@pytest.mark.parametrize("optimize_cell", [False, True])
def test_streamed_updates_match_fresh_single_structure_runs(
    optimizer: str, optimize_cell: bool
) -> None:
    structures = _structures()
    together, _ = _run(
        structures, optimizer=optimizer, block_size=8, optimize_cell=optimize_cell
    )
    for row in together:
        alone, _ = _run(
            [structures[int(row.ordinal[0])]],
            slots=1,
            optimizer=optimizer,
            optimize_cell=optimize_cell,
        )
        npt.assert_allclose(
            row.particles.positions, alone[0].particles.positions, atol=1e-12
        )
        npt.assert_allclose(
            row.position_gradients, alone[0].position_gradients, atol=1e-12
        )
        npt.assert_allclose(
            row.systems.cell.vectors, alone[0].systems.cell.vectors, atol=1e-12
        )
        npt.assert_allclose(
            row.systems.potential_energy, alone[0].systems.potential_energy, atol=1e-12
        )


def test_pack_rejects_unknown_species_and_oversized_structures() -> None:
    structure = _structures()[0]
    with pytest.raises(ValueError, match="species vocabulary"):
        payload_from_structures([structure], 8, (Label("Ne"),), [0])
    with pytest.raises(ValueError, match="slot capacity"):
        payload_from_structures([structure], 2, SPECIES, [0])


@state_dataclass
class WrappedState:
    batch: Batch
    unrelated: Array


def test_factory_composes_with_a_nested_state_lens() -> None:
    structures = _structures()
    setup = _setup(structures, 2, 3, 0.0, "sgd")
    state_lens = identity_lens(WrappedState).focus(lambda s: s.batch)
    potential = make_lennard_jones_from_state(
        state_lens, parameters=LJ, gradient=POSITIONS_ONLY
    )
    step, _, reset = make_streaming_relax_propagator(
        state_lens,
        potential,
        chain(optax.scale(-0.1)),
        POSITIONS_ONLY,
        setup.slots,
        reset_layout=ResetLayout(fields=const_lens(()), system_index=lambda s: ()),
        force_tolerance=0.0,
        max_steps=3,
        include_cell=False,
    )
    payload = tree_concat(
        payload_from_structures(structures[:1], CAPACITY, SPECIES, [0]),
        setup.idle,
    )
    state = replace_slots(
        WrappedState(setup.state, jnp.array(17)),
        payload,
        setup.state.finished,
        state_lens=state_lens,
        slots=setup.slots,
        reset=reset,
    )
    assert_type(state, WrappedState)
    final = propagate_and_fix(make_cycle_function(step), jax.random.key(0), state)
    assert int(final.unrelated) == 17
    npt.assert_array_equal(final.batch.step, [1, 0])


def test_orthogonal_and_triclinic_jobs_share_the_idle_signature() -> None:
    structures = _structures()
    particles, cell = structures[0]
    orthogonal = (
        bind(cell).focus(lambda c: c.frame).set(OrthogonalFrame(jnp.diag(cell.vectors)))
    )
    structures[0] = particles, orthogonal
    rows, _ = _run(structures, max_steps=0)
    assert sorted(int(row.ordinal[0]) for row in rows) == [0, 1, 2]
    for row in rows:
        original = structures[int(row.ordinal[0])]
        npt.assert_array_equal(row.systems.cell.vectors[0], original[1].vectors)


def test_factory_preserves_native_optimizer_state_type() -> None:
    setup = _setup(_structures(), 2, 3, 0.0, "sgd")
    state_lens = identity_lens(StreamingRelaxState[ScaleByFireState])
    potential = make_lennard_jones_from_state(
        state_lens, parameters=LJ, gradient=POSITIONS_ONLY
    )
    _, init, reset = make_streaming_relax_propagator(
        state_lens,
        potential,
        ScaleByFire[PositionsAndCell](),
        POSITIONS_ONLY,
        setup.slots,
        reset_layout=fire_reset_layout(),
        force_tolerance=0.0,
        max_steps=3,
        include_cell=False,
    )
    assert_type(init, OptInit[ScaleByFireState])
    assert_type(reset, OptReset[ScaleByFireState])
    state = make_streaming_relax_state(
        setup.slots, setup.idle, init, setup.state.neighborlist_params
    )
    assert_type(state, StreamingRelaxState[ScaleByFireState])
    assert isinstance(state.opt_state, ScaleByFireState)
    initialized = init(particles=state.particles, systems=state.systems)
    assert_type(initialized, ScaleByFireState)
    restored = reset(
        particles=state.particles,
        systems=state.systems,
        opt_state=initialized,
        mask=state.finished,
    )
    assert_type(restored, ScaleByFireState)
    npt.assert_array_equal(restored.dt.data, initialized.dt.data)


def test_noncontiguous_reservations_and_nondefault_keys() -> None:
    slots = Table(
        (SystemId(11), SystemId(29)),
        Index(
            tuple(ParticleId(100 + 2 * i) for i in range(2 * CAPACITY)),
            jnp.arange(2 * CAPACITY).reshape(CAPACITY, 2).T,
            max_count=1,
            _cls=ParticleId,
        ),
    )
    structures = _structures()
    expected, _ = _run(structures, optimizer="fire")
    actual, setup = _run(structures, optimizer="fire", reservations=slots)
    assert setup.state.particles.keys == slots.data.keys
    assert setup.state.systems.keys == slots.keys
    for before, after in zip(expected, actual, strict=True):
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
            npt.assert_allclose(a, b, atol=1e-12)
