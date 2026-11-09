"""kUPS Lennard-Jones MD step time on the stack: kUPS's own path vs nvalchemi ops.

Builds B replicas of an fcc-argon supercell (R x R x R unit cells, 4 R^3 atoms each)
as ``kups.application.simulations.md`` does: a Verlet skin list (``--skin``, the app's
default 1 Å) read by LJ over neighbor rows. Prints one JSON line per configuration.

    md_bench.py --nl IMPL --reps R --batch B [--block K] [--steps S] [--f32] [--check]

IMPL:
  kups                 the stack's MD path: skin list built by AdaptiveNeighborList,
                       LJ over its neighbor rows
  nvalchemi-ptx|stock  the same, with nvalchemi's cell list building the skin list
  nvalchemi-lj-ptx|stock  nvalchemi's cell list and fused LJ kernel, every step
                       (kUPS's skin list unused)
  nvalchemi-rows-ptx|stock  nvalchemi's fused LJ kernel on kUPS's skin list rows
                       (pair kernel against pair kernel)

``--f32`` is the app's fast setting: float32 state, float32 LJ kernel (Pallas) and int32
indices; otherwise float64 throughout with int64 indices (the app's defaults).
``--block K`` fuses K steps per dispatch. ``--check`` compares energies, forces and cell
gradients after three Verlet steps against ``kups``.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("--nl", default="kups",
                    choices=["kups", "nvalchemi-ptx", "nvalchemi-stock",
                             "nvalchemi-lj-ptx", "nvalchemi-lj-stock",
                             "nvalchemi-rows-ptx", "nvalchemi-rows-stock"])
parser.add_argument("--reps", type=int, default=8)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--block", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--cutoff", type=float, default=8.5)
parser.add_argument("--skin", type=float, default=1.0)
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

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from kups.application.md.data import MdParameters, MdState, md_state_from_ase  # noqa: E402
from kups.application.md.simulation import make_md_propagator  # noqa: E402
from kups.application.potential.filter import POSITIONS_AND_CELL  # noqa: E402
from kups.application.simulations.potentials import LjPotentialConfig  # noqa: E402
from kups.application.utils.propagate import make_cycle_function  # noqa: E402
from kups.config import set_index_dtype  # noqa: E402
from kups.core.data import Table  # noqa: E402
from kups.core.lens import identity_lens  # noqa: E402
from kups.core.neighborlist import UniversalNeighborlistParameters  # noqa: E402
from kups.core.neighborlist.verlet import SkinList  # noqa: E402
from kups.core.propagator import LoopPropagator, propagate_and_fix  # noqa: E402
from kups.core.typing import SystemId  # noqa: E402
from kups.core.utils.jax import key_chain  # noqa: E402

A_FCC = 5.26  # argon lattice constant (Å)
SIGMA, EPSILON = 3.405, 0.010326  # argon (Å, eV)
MD = MdParameters(temperature=100.0, time_step=2.0, friction_coefficient=1.0,
                  thermostat_time_constant=100.0, target_pressure=1.0,
                  pressure_coupling_time=1.0e10, compressibility=4.5e-5,
                  minimum_scale_factor=1.0, integrator="verlet", initialize_momenta=True,
                  skin=args.skin)


def nvalchemi_cells(cutoff: float) -> int:
    """nvalchemi's grid for these boxes: floor(L / cutoff) cells per axis, at least 4."""
    return args.batch * max(4, math.floor(A_FCC * args.reps / cutoff)) ** 3


def build(nl: str, seed: int = 0):
    set_index_dtype("int32" if args.f32 else "int64")
    float_dtype = jnp.float32 if args.f32 else jnp.float64
    chain = key_chain(jax.random.key(seed))
    parts, systems = [], []
    for b in range(args.batch):
        atoms = ase.build.bulk("Ar", "fcc", a=A_FCC, cubic=True).repeat((args.reps,) * 3)
        atoms.rattle(0.05, seed=b)
        p, s = md_state_from_ase(atoms, MD, key=next(chain), float_dtype=float_dtype)
        parts.append(p)
        systems.append(s)
    particles, systems = Table.union(parts, systems)
    lj = LjPotentialConfig(cutoff=args.cutoff, parameters={"Ar": (SIGMA, EPSILON)},
                           mixing_rule="lorentz_berthelot",
                           kernel_dtype="float32" if args.f32 else "float64")
    state_lens = identity_lens(MdState)
    potential, cutoff = lj.build(state_lens, POSITIONS_AND_CELL, particles, systems)
    if nl.startswith("nvalchemi-lj"):
        potential = kups_nvalchemi.NvalchemiLennardJones(
            EPSILON, SIGMA, args.cutoff, nvalchemi_cells(args.cutoff))
    elif nl.startswith("nvalchemi-rows"):
        potential = kups_nvalchemi.NvalchemiRowsLennardJones(EPSILON, SIGMA, args.cutoff)
    evaluate_forces, propagator = make_md_propagator(state_lens, MD.integrator, potential)
    params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, cutoff, base=2)
    skins = Table.arange(jnp.array([args.skin]), label=SystemId)
    builder = {}
    if nl in ("nvalchemi-ptx", "nvalchemi-stock"):
        builder["builder"] = functools.partial(
            kups_nvalchemi.NvalchemiNeighborList.new, cutoff=args.cutoff + args.skin,
            max_total_cells=nvalchemi_cells(args.cutoff + args.skin))
    skin_list = SkinList.create(particles, systems, cutoff, skins, **builder)
    state = MdState(particles, systems, params, skin_list, jnp.array([0]))
    state = propagate_and_fix(make_cycle_function(evaluate_forces), jax.random.key(seed),
                              state)
    return propagator, state


def check() -> None:
    """Energies, forces and cell gradients against the stack's own MD path."""
    out = {}
    for nl in ("kups", args.nl):
        propagator, state = build(nl)
        cycle = make_cycle_function(propagator)
        for k in range(3):  # deterministic Verlet steps, same keys
            state = propagate_and_fix(cycle, jax.random.key(k), state)
        out[nl] = (np.asarray(state.systems.data.potential_energy, np.float64),
                   np.asarray(state.particles.data.position_gradients, np.float64),
                   np.asarray(state.systems.data.cell_gradients.vectors, np.float64))
    (e0, g0, c0), (e1, g1, c1) = out["kups"], out[args.nl]
    print(json.dumps({"check": args.nl, "dtype": "f32" if args.f32 else "f64",
                      "energy_max_rel": float(np.max(np.abs(e1 - e0) / np.abs(e0))),
                      "force_max_abs": float(np.max(np.abs(g1 - g0))),
                      "force_scale": float(np.max(np.abs(g0))),
                      "cell_grad_max_abs": float(np.max(np.abs(c1 - c0))),
                      "cell_grad_scale": float(np.max(np.abs(c0)))}))


def bench() -> None:
    propagator, state = build(args.nl)
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
    print(json.dumps({"nl": args.nl, "reps": args.reps, "batch": args.batch,
                      "atoms": n, "atoms_per_system": n // args.batch, "block": args.block,
                      "dtype": "f32" if args.f32 else "f64", "skin": args.skin,
                      "ms_per_step": 1e3 * dt, "atom_steps_per_s": n / dt,
                      "compile_s": compile_s,
                      "avg_edges": state.neighborlist_params.avg_edges,
                      "device": jax.devices()[0].device_kind}))
    if args.profile:
        from devtime import device_kernels
        from kups.core.result import as_result_function

        key = next(chain)
        step = jax.jit(as_result_function(propagator))  # the cycle donates its state
        ks = device_kernels(lambda s: step(key, s).value, state, reps=5)
        total = sum(ks.values())
        print(json.dumps({"nl": args.nl, "atoms": n, "dtype": "f32" if args.f32 else "f64",
                          "device_ms_per_step": total / 1e3 / args.block,
                          "wall_ms_per_step": 1e3 * dt,
                          "top_kernels": [[name[:80], round(us / args.block, 1)]
                                          for name, us in list(ks.items())[:args.profile]]}))


if args.check:
    check()
else:
    bench()
