"""kUPS Lennard-Jones MD step time: kUPS's neighbor lists vs nvalchemi's cell list.

Builds B replicas of an fcc-argon supercell (R x R x R unit cells, 4 R^3 atoms each)
exactly as ``kups.application.simulations.md`` does, swapping only the neighbor-list
factory, and prints one JSON line per configuration.

    md_bench.py --nl kups|kups-cell|nvalchemi-stock|nvalchemi-ptx --reps R --batch B
                [--block K] [--steps S] [--f32] [--check]

``--nl kups`` is main's default (AdaptiveNeighborList: dense below 10k particles per
system, cell list above); ``kups-cell`` forces kUPS's cell list. ``--block K`` fuses K
steps per dispatch (LoopPropagator); per-step times are wall time / steps.
``--check`` compares the edge set, energies and forces against ``kups``.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("--nl", default="nvalchemi-ptx",
                    choices=["kups", "kups-cell", "nvalchemi-stock", "nvalchemi-ptx",
                             "nvalchemi-lj-stock", "nvalchemi-lj-ptx"])
parser.add_argument("--reps", type=int, default=8)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--block", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--cutoff", type=float, default=8.5)
parser.add_argument("--f32", action="store_true")
parser.add_argument("--check", action="store_true")
parser.add_argument("--profile", type=int, default=0, help="print the top N kernels")
args = parser.parse_args()

import kups_nvalchemi  # noqa: E402  (before nvalchemiops.jax is imported)

if args.nl.startswith("nvalchemi"):
    kups_nvalchemi.use_backend("jax-ptx" if args.nl.endswith("-ptx") else "stock")

import ase.build  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", not args.f32)
jax.config.update("jax_default_matmul_precision", "highest")

from kups.application.md.data import MdParameters, MdState, md_state_from_ase  # noqa: E402
from kups.application.md.simulation import make_md_propagator  # noqa: E402
from kups.application.potential.filter import POSITIONS_AND_CELL  # noqa: E402
from kups.application.simulations.potentials import LjPotentialConfig  # noqa: E402
from kups.application.utils.propagate import (  # noqa: E402
    make_cycle_function,
    propagate_and_fix,
)
from kups.core.data import Table  # noqa: E402
from kups.core.lens import identity_lens  # noqa: E402
from kups.core.neighborlist import (  # noqa: E402
    AdaptiveNeighborList,
    CellListNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.propagator import LoopPropagator  # noqa: E402
from kups.core.utils.jax import key_chain  # noqa: E402

A_FCC = 5.26  # argon lattice constant (Å)
MD = MdParameters(temperature=100.0, time_step=2.0, friction_coefficient=1.0,
                  thermostat_time_constant=100.0, target_pressure=1.0,
                  pressure_coupling_time=1.0e10, compressibility=4.5e-5,
                  minimum_scale_factor=1.0, integrator="verlet", initialize_momenta=True)


def nvalchemi_cells(reps: int, batch: int, cutoff: float) -> int:
    """nvalchemi's grid for these boxes: floor(L / cutoff) cells per axis, at least 4."""
    return batch * max(4, math.floor(A_FCC * reps / cutoff)) ** 3


SIGMA, EPSILON = 3.405, 0.010326  # argon (Å, eV)


def factory(nl: str):
    if nl == "kups-cell":
        return CellListNeighborList.from_state
    if nl in ("nvalchemi-stock", "nvalchemi-ptx"):
        return functools.partial(kups_nvalchemi.NvalchemiNeighborList.from_state,
                                 cutoff=args.cutoff,
                                 max_total_cells=nvalchemi_cells(args.reps, args.batch,
                                                                 args.cutoff))
    return AdaptiveNeighborList.from_state


def build(nl: str, seed: int = 0):
    lj = LjPotentialConfig(cutoff=args.cutoff, parameters={"Ar": (SIGMA, EPSILON)},
                           mixing_rule="lorentz_berthelot")
    state_lens = identity_lens(MdState)
    potential, cutoff = lj.build(state_lens, POSITIONS_AND_CELL, factory(nl))
    if nl.startswith("nvalchemi-lj"):
        potential = kups_nvalchemi.NvalchemiLennardJones(
            EPSILON, SIGMA, args.cutoff, nvalchemi_cells(args.reps, args.batch, args.cutoff))
    propagator = make_md_propagator(state_lens, MD.integrator, potential)
    chain = key_chain(jax.random.key(seed))
    parts, systems = [], []
    for b in range(args.batch):
        atoms = ase.build.bulk("Ar", "fcc", a=A_FCC, cubic=True).repeat((args.reps,) * 3)
        atoms.rattle(0.05, seed=b)
        p, s = md_state_from_ase(atoms, MD, key=next(chain))
        parts.append(p)
        systems.append(s)
    particles, systems = Table.union(parts, systems)
    params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, cutoff, base=2)
    state = MdState(particles=particles, systems=systems, neighborlist_params=params,
                    step=jnp.array([0]))
    return potential, propagator, state


def check() -> None:
    """Edges, energies and forces of the nvalchemi neighbor list against kUPS main's."""
    out = {}
    for nl in ("kups", args.nl):
        _, propagator, state = build(nl)
        cycle = make_cycle_function(propagator)
        for k in range(3):  # deterministic Verlet steps, same keys
            state = propagate_and_fix(cycle, jax.random.key(k), state)
        out[nl] = (np.asarray(state.systems.data.potential_energy),
                   np.asarray(state.particles.data.position_gradients),
                   np.asarray(state.systems.data.cell_gradients.vectors))
    (e0, g0, c0), (e1, g1, c1) = out["kups"], out[args.nl]
    print(json.dumps({"check": args.nl, "energy_max_rel": float(np.max(np.abs(e1 - e0) /
                                                                     np.abs(e0))),
                      "force_max_abs": float(np.max(np.abs(g1 - g0))),
                      "force_scale": float(np.max(np.abs(g0))),
                      "cell_grad_max_abs": float(np.max(np.abs(c1 - c0))),
                      "cell_grad_scale": float(np.max(np.abs(c0)))}))


def bench() -> None:
    _, propagator, state = build(args.nl)
    if args.block > 1:
        propagator = LoopPropagator(propagator, args.block)
    cycle = make_cycle_function(propagator)
    chain = key_chain(jax.random.key(1))
    t = time.perf_counter()
    state = propagate_and_fix(cycle, next(chain), state)
    jax.block_until_ready(state)
    compile_s = time.perf_counter() - t
    for _ in range(3):
        state = propagate_and_fix(cycle, next(chain), state)
    jax.block_until_ready(state)
    cycles = max(3, args.steps // args.block)
    keys = [next(chain) for _ in range(cycles)]
    t = time.perf_counter()
    for k in keys:
        state = propagate_and_fix(cycle, k, state)
    jax.block_until_ready(state)
    dt = (time.perf_counter() - t) / (cycles * args.block)
    n = state.particles.size
    if args.profile:
        from devtime import device_kernels
        from kups.core.result import as_result_function

        key = next(chain)
        step = jax.jit(as_result_function(propagator))  # the cycle donates its state
        ks = device_kernels(lambda s: step(key, s).value, state, reps=5)
        total = sum(ks.values())
        print(f"# device {total / 1e3 / args.block:.3f} ms/step of {1e3 * dt:.3f} wall")
        for name, us in list(ks.items())[:args.profile]:
            print(f"#  {us / args.block:9.1f} us  {name[:100]}")
    print(json.dumps({"nl": args.nl, "reps": args.reps, "batch": args.batch,
                      "atoms": n, "atoms_per_system": n // args.batch, "block": args.block,
                      "dtype": "f32" if args.f32 else "f64", "ms_per_step": 1e3 * dt,
                      "atom_steps_per_s": n / dt, "compile_s": compile_s,
                      "avg_edges": state.neighborlist_params.avg_edges,
                      "device": jax.devices()[0].device_kind}))


if args.check:
    check()
else:
    bench()
