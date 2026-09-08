# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Streaming relaxation through assertion fixes and pure slot replacement."""

from collections.abc import Sequence
from dataclasses import dataclass

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
from kups.application.potential.filter import POSITIONS_ONLY
from kups.application.relaxation.simulation import OptReset
from kups.application.relaxation.streaming import (
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
from kups.core.data import Table
from kups.core.data.slots import SlotLayout
from kups.core.lens import Lens, identity_lens
from kups.core.patch import Patch
from kups.core.potential import EmptyType, Potential
from kups.core.propagator import (
    LoopPropagator,
    Propagator,
    SequentialPropagator,
    propagate_and_fix,
)
from kups.core.result import as_result_function
from kups.core.stream import RefillPropagator
from kups.core.typing import Label, SystemId
from kups.core.utils.jax import dataclass as state_dataclass
from kups.core.utils.jax import tree_concat, tree_map
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.potential.common.geometry import PositionsAndCell
from kups.relaxation.optimizer import ChainOptimizer, chain
from kups.relaxation.transforms import MaxStepSize, ScaleByAseLbfgs, ScaleByFire

SPECIES = (Label("Ar"),)
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
    layout: SlotLayout
    idle: RelaxPayload
    step: Propagator[StreamingRelaxState]
    state: StreamingRelaxState
    reset: OptReset
    potential: Potential[
        StreamingRelaxState, PositionsAndCell, EmptyType, Patch[StreamingRelaxState]
    ]


def _setup(
    structures: Sequence[Structure],
    slots: int,
    max_steps: int,
    tolerance: float,
    optimizer: str,
) -> Setup:
    layout = SlotLayout(slots, 8)
    state_lens = identity_lens(StreamingRelaxState)
    potential = make_lennard_jones_from_state(
        state_lens, parameters=LJ, gradient=POSITIONS_ONLY
    )
    idle = idle_payload(layout, SPECIES, (True, True, True), 5.0)
    opt: ChainOptimizer[PositionsAndCell] = chain(optax.scale(-0.1))
    if optimizer == "lbfgs":
        opt = chain(
            ScaleByAseLbfgs[PositionsAndCell](memory_size=3),
            MaxStepSize[PositionsAndCell](0.2),
            optax.scale(-1.0),
        )
    elif optimizer == "fire":
        opt = chain(optax.scale(-1.0), ScaleByFire[PositionsAndCell](dt_start=0.1))
    step, init, reset = make_streaming_relax_propagator(
        state_lens,
        potential,
        opt,
        POSITIONS_ONLY,
        layout,
        force_tolerance=tolerance,
        max_steps=max_steps,
        include_cell=False,
    )
    nlp = estimate_stream_neighborlist(
        structures, layout, SPECIES, LJ.cutoff, multiplier=2.0
    )
    return Setup(
        layout,
        idle,
        step,
        make_streaming_relax_state(layout, idle, init, nlp),
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
) -> tuple[list[RelaxPayload], Setup]:
    setup = _setup(structures, slots, max_steps, tolerance, optimizer)
    pending = iter(enumerate(structures))
    completed: list[RelaxPayload] = []
    payload_lens: Lens[StreamingRelaxState, RelaxPayload] = slot_payload_lens(
        setup.layout
    )

    @jax.jit
    def install(
        s: StreamingRelaxState, payload: RelaxPayload, mask: Table[SystemId, Array]
    ) -> StreamingRelaxState:
        return replace_slots(s, payload, mask, layout=setup.layout, reset=setup.reset)

    snapshot = jax.jit(payload_lens.get)

    def refill(
        s: StreamingRelaxState, mask: Table[SystemId, Array]
    ) -> StreamingRelaxState:
        requested, old = jax.device_get((mask.data, snapshot(s)))
        rows: list[RelaxPayload] = []
        for i in range(slots):
            if requested[i]:
                if int(old.ordinal[i]) >= 0:
                    completed.append(tree_map(lambda x: x[i : i + 1].copy(), old))
                item = next(pending, None)
                rows.append(
                    setup.idle
                    if item is None
                    else payload_from_structures(
                        [item[1]], setup.layout, SPECIES, [item[0]]
                    )
                )
            else:
                rows.append(setup.idle)  # replace_slots preserves unselected rows.
        return install(s, tree_concat(*rows), mask)

    gate: RefillPropagator[StreamingRelaxState] = RefillPropagator(
        lambda s: s.finished, refill
    )

    def repetitions(s: StreamingRelaxState) -> Array:
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
    payload_lens: Lens[StreamingRelaxState, RelaxPayload] = slot_payload_lens(
        setup.layout
    )
    evaluate = jax.jit(as_result_function(lambda s: setup.potential(s).data))
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
            result.value.gradients.positions.data[: setup.layout.capacity],
            atol=1e-12,
        )
        if max_steps == 0 or tolerance:
            original = structures[int(row.ordinal[0])][0].data.positions
            npt.assert_array_equal(
                row.particles.positions[0, : len(original)], original
            )


@pytest.mark.parametrize("optimizer", ["sgd", "fire", "lbfgs"])
def test_streamed_updates_match_fresh_single_structure_runs(optimizer: str) -> None:
    structures = _structures()
    together, _ = _run(structures, optimizer=optimizer, block_size=8)
    for row in together:
        alone, _ = _run([structures[int(row.ordinal[0])]], slots=1, optimizer=optimizer)
        npt.assert_allclose(
            row.particles.positions, alone[0].particles.positions, atol=1e-12
        )
        npt.assert_allclose(
            row.position_gradients, alone[0].position_gradients, atol=1e-12
        )


def test_pack_rejects_unknown_species_and_oversized_structures() -> None:
    structure = _structures()[0]
    with pytest.raises(ValueError, match="species vocabulary"):
        payload_from_structures([structure], SlotLayout(1, 8), (Label("Ne"),), [0])
    with pytest.raises(ValueError, match="slot capacity"):
        payload_from_structures([structure], SlotLayout(1, 2), SPECIES, [0])


@state_dataclass
class WrappedState:
    batch: StreamingRelaxState
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
        setup.layout,
        force_tolerance=0.0,
        max_steps=3,
        include_cell=False,
    )
    payload = tree_concat(
        payload_from_structures(structures[:1], setup.layout, SPECIES, [0]),
        setup.idle,
    )
    batch = replace_slots(
        setup.state, payload, setup.state.finished, layout=setup.layout, reset=reset
    )
    final = propagate_and_fix(
        make_cycle_function(step), jax.random.key(0), WrappedState(batch, jnp.array(17))
    )
    assert int(final.unrelated) == 17
    npt.assert_array_equal(final.batch.step, [1, 0])
