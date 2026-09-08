# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Fixed batches versus streaming on identical heterogeneous LJ relaxations.

Run from the repository root, for example:
    CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false \
        .venv/bin/python benchmarks/bench_relax_streaming.py --slots 128

JSONL output separates warmup from synchronized timed trials. Inputs are packed
once on CPU; both policies receive the same shuffled stream of 32 template
structures. Timings include refill, device transfers, and result snapshots, but
exclude initial packing and the final NumPy comparison. No file I/O is timed.
The heterogeneous mix deliberately includes already-converged structures;
--homogeneous is the equal-work control. All jobs must return finite results
and either meet the common force tolerance or reach the reported step budget.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from jax import Array

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import POSITIONS_ONLY
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
from kups.core.lens import bind, identity_lens
from kups.core.propagator import LoopPropagator, SequentialPropagator, propagate_and_fix
from kups.core.stream import RefillPropagator
from kups.core.typing import Label, SystemId
from kups.core.utils.jax import dataclass as state_dataclass
from kups.core.utils.jax import tree_concat
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.relaxation.optimizer import chain
from kups.relaxation.transforms import MaxStepSize, ScaleByFire


@state_dataclass
class State:
    batch: StreamingRelaxState
    evaluations: Array


@dataclass
class Measurement:
    steps: npt.NDArray[np.int64]
    energy: npt.NDArray[np.float32 | np.float64]
    positions: npt.NDArray[np.float32 | np.float64]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--mult", type=int, default=3)
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--modes", default="fixed:8,stream:1,stream:8")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--minimum-rattle", type=float, default=0.002)
    parser.add_argument("--homogeneous", action="store_true")
    args = parser.parse_args()
    if args.waves < 1 or args.trials < 0:
        parser.error("waves must be positive and trials non-negative")
    x64 = bool(jax.config.read("jax_enable_x64"))
    slots, count = args.slots, args.slots * args.waves
    capacity = 4 * args.mult**3
    layout = SlotLayout(slots, capacity)
    species = (Label("Ar"),)
    rng = np.random.default_rng(20260908)
    # The same shuffled job order in every policy. Pool reuse bounds preparation.
    choices = rng.permutation(np.arange(count) % 32)
    t0 = time.perf_counter()
    structures: list[Structure] = []
    with jax.default_device(jax.devices("cpu")[0]):
        for i in range(32):
            amplitude = (
                0.15
                if args.homogeneous
                else (0.002 if i < 16 else 0.03 if i < 24 else 0.15 if i < 30 else 0.35)
            )
            amplitude = max(amplitude, args.minimum_rattle)
            atoms = ase.build.bulk("Ar", "fcc", a=5.3, cubic=True) * args.mult
            atoms.rattle(amplitude, seed=0 if args.homogeneous else i)
            p, cell, _ = particles_from_ase(atoms)
            structures.append((p, cell))
        idle = idle_payload(layout, species, (True, True, True), 5.0)
        pool = jax.device_get(
            tree_concat(
                payload_from_structures(structures, layout, species, range(32)),
                idle,
            )
        )
        lj = LennardJonesParameters.from_dict(
            cutoff=5.0,
            parameters={"Ar": (3.405, 0.010326)},
            mixing_rule="lorentz_berthelot",
        )
        nlp = estimate_stream_neighborlist(
            structures, layout, species, lj.cutoff, multiplier=2.0
        )
    lj = jax.device_put(lj)
    state_lens = identity_lens(State).focus(lambda s: s.batch)
    potential = make_lennard_jones_from_state(
        state_lens, parameters=lj, gradient=POSITIONS_ONLY
    )
    step, init, reset = make_streaming_relax_propagator(
        state_lens,
        potential,
        chain(optax.scale(-1.0), ScaleByFire(dt_start=0.1), MaxStepSize(0.2)),
        POSITIONS_ONLY,
        layout,
        force_tolerance=args.tolerance,
        max_steps=args.max_steps,
        include_cell=False,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        template = jax.device_get(
            State(
                make_streaming_relax_state(layout, idle, init, nlp),
                jnp.array(0),
            )
        )
    print(
        json.dumps(
            dict(
                kind="setup",
                slots=slots,
                atoms=capacity,
                total_atoms=slots * capacity,
                jobs=count,
                x64=x64,
                device=str(jax.devices()[0]),
                homogeneous=args.homogeneous,
                minimum_rattle=args.minimum_rattle,
                neighborlist_capacities=dict(
                    candidates=nlp.avg_candidates,
                    image_candidates=nlp.avg_image_candidates,
                    edges=nlp.avg_edges,
                ),
                packing_seconds=time.perf_counter() - t0,
            )
        ),
        flush=True,
    )
    payload_lens = slot_payload_lens(layout)

    @jax.jit
    def snapshot(s: State) -> RelaxPayload:
        return payload_lens.get(s.batch)

    @jax.jit
    def install(s: State, payload: RelaxPayload, mask: Table[SystemId, Array]) -> State:
        return state_lens.set(
            s, replace_slots(s.batch, payload, mask, layout=layout, reset=reset)
        )

    def counted(key: Array, s: State) -> State:
        return bind(step(key, s)).focus(lambda x: x.evaluations).apply(lambda n: n + 1)

    reference: Measurement | None = None
    for mode in args.modes.split(","):
        policy, block_text = mode.split(":")
        block = int(block_text)
        if policy not in {"fixed", "stream"} or block < 1:
            parser.error("modes must be fixed:N or stream:N with positive N")
        next_job = 0
        outputs: list[RelaxPayload] = []
        callback_seconds = 0.0

        def requested(s: State) -> Table[SystemId, Array]:
            done = s.batch.finished
            if policy == "fixed":
                ready = jnp.all(done.data | (s.batch.slot_ordinal < 0))
                return done.map_data(lambda m: m & ready)
            return done

        def refill(s: State, mask: Table[SystemId, Array]) -> State:
            nonlocal next_job, callback_seconds
            start = time.perf_counter()
            mask_host, old = jax.device_get((mask.data, snapshot(s)))
            output_rows = np.flatnonzero(mask_host & (old.ordinal >= 0))
            if len(output_rows):
                outputs.append(jax.tree.map(lambda x: x[output_rows].copy(), old))
            slots_to_fill = np.flatnonzero(mask_host)
            amount = min(len(slots_to_fill), count - next_job)
            selection = np.full(slots, 32, dtype=np.int64)
            ordinal = np.full(slots, -1, dtype=np.asarray(pool.ordinal).dtype)
            selection[slots_to_fill[:amount]] = choices[next_job : next_job + amount]
            ordinal[slots_to_fill[:amount]] = np.arange(next_job, next_job + amount)
            payload = (
                bind(jax.tree.map(lambda x: x[selection], pool))
                .focus(lambda p: p.ordinal)
                .set(ordinal)
            )
            next_job += amount
            s = install(s, payload, mask)
            jax.block_until_ready(s)
            callback_seconds += time.perf_counter() - start
            return s

        def repetitions(s: State) -> Array:
            # A repair-only attempt does not evaluate the potential; EOS does not either.
            run = ~requested(s).data.any() & (s.batch.slot_ordinal >= 0).any()
            return jnp.where(run, block, 0)

        cycle = make_cycle_function(
            SequentialPropagator(
                (
                    RefillPropagator(requested, refill),
                    LoopPropagator(counted, repetitions),
                )
            )
        )
        finished = jax.jit(lambda s: jnp.all(s.batch.slot_ordinal < 0))
        for trial in range(args.trials + 1):
            next_job, callback_seconds = 0, 0.0
            outputs = []
            s = jax.device_put(jax.tree.map(np.copy, template))
            jax.block_until_ready(s)
            start = time.perf_counter()
            cycles = 0
            while True:
                s = propagate_and_fix(cycle, jax.random.key(42), s)
                cycles += 1
                if bool(finished(s)):
                    break
                if cycles > count * (args.max_steps + 2):
                    raise RuntimeError("Stream did not drain")
            jax.block_until_ready(s)
            elapsed = time.perf_counter() - start
            # numpy-only collection after synchronization; no GPU operation per result row.
            ordinals = np.concatenate([np.asarray(p.ordinal) for p in outputs])
            order = np.argsort(ordinals)
            np.testing.assert_array_equal(ordinals[order], np.arange(count))
            steps = np.concatenate([np.asarray(p.step) for p in outputs])[order].astype(
                np.int64
            )
            energy = np.concatenate(
                [np.asarray(p.systems.potential_energy) for p in outputs]
            )[order]
            forces = np.concatenate(
                [np.asarray(p.position_gradients) for p in outputs]
            )[order]
            positions = np.concatenate(
                [np.asarray(p.particles.positions) for p in outputs]
            )[order]
            finite = (
                np.isfinite(positions).all(axis=(1, 2))
                & np.isfinite(energy)
                & np.isfinite(forces).all(axis=(1, 2))
            )
            if not finite.all():
                bad = np.flatnonzero(~finite)
                print(
                    json.dumps(
                        dict(
                            kind="invalid_results",
                            mode=mode,
                            ordinals=bad.tolist(),
                            templates=choices[bad].tolist(),
                            steps=steps[bad].tolist(),
                        )
                    ),
                    flush=True,
                )
                raise AssertionError("Non-finite relaxation results")
            fmax = np.linalg.norm(forces, axis=-1).max(axis=-1)
            assert np.all((fmax < args.tolerance) | (steps == args.max_steps))
            result = Measurement(steps, energy, positions)
            if reference is None:
                reference = result
            position_error = float(np.max(np.abs(positions - reference.positions)))
            energy_error = float(np.max(np.abs(energy - reference.energy)))
            # Tolerate floating point reduction-order changes, not dropped or unfinished jobs.
            if x64:
                np.testing.assert_allclose(
                    energy, reference.energy, rtol=1e-8, atol=1e-8
                )
                np.testing.assert_allclose(
                    positions, reference.positions, rtol=1e-7, atol=1e-7
                )
            else:
                np.testing.assert_allclose(
                    energy, reference.energy, rtol=1e-4, atol=1e-4
                )
                np.testing.assert_allclose(
                    positions, reference.positions, rtol=1e-3, atol=1e-3
                )
            print(
                json.dumps(
                    dict(
                        kind="warmup" if trial == 0 else "measurement",
                        mode=mode,
                        trial=trial,
                        seconds=elapsed,
                        jobs_per_second=count / elapsed,
                        callback_seconds=callback_seconds,
                        host_cycles=cycles,
                        evaluations=int(s.evaluations),
                        useful_fraction=float(
                            (steps + 1).sum() / (int(s.evaluations) * slots)
                        ),
                        step_quantiles=np.quantile(
                            steps, [0, 0.5, 0.9, 0.99, 1]
                        ).tolist(),
                        capped=int((steps == args.max_steps).sum()),
                        max_position_difference=position_error,
                        max_energy_difference=energy_error,
                        max_step_difference=int(
                            np.max(np.abs(steps - reference.steps))
                        ),
                        memory=jax.devices()[0].memory_stats(),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
