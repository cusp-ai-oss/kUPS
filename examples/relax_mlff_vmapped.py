# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Batched MLFF relaxation with per-system vmapped LBFGS optimizer states.

This is a standalone proof-of-concept demonstrating the fix for the
shared-LBFGS convergence issue in batched relaxation. It uses kUPS's
model loading and optimizer primitives but bypasses the Table/Lens
propagator machinery to implement the split-vmap-scatter pattern
directly.

Architecture:
  1. Batched model forward pass: one call over all concatenated atoms
     (kUPS-native, no padding of model inputs)
  2. Split per-atom gradients by system using static index arrays
  3. Pad to (B, max_atoms, 3) for vmapped optimizer update
  4. jax.vmap(optimizer.update) — independent LBFGS state per molecule
  5. Unpad and scatter updates back to concatenated positions

The entire step (gradient + split + pad + vmap LBFGS + unpad + scatter)
is compiled into a single JIT'd JAX kernel.

Usage:
    python relax_mlff_vmapped.py config.yaml

YAML config format (same as relax_mlff, plus optional per_system_optimizer):
    inp_files:
      - structure1.cif
      - structure2.xyz
    model_path: /path/to/model.zip  # or hf://...
    relax:
      optimizer:
        - transform: scale_by_ase_lbfgs
          memory_size: 20
          alpha: 70
        - transform: max_step_size
          max_step_size: 0.2
        - transform: scale
          step_size: -1
      optimize_cell: false
    run:
      out_file: relax_output.json
      max_steps: 200
      force_tolerance: 0.01
      seed: 42
"""

from __future__ import annotations

import json
import logging
import time
import zipfile
from pathlib import Path

import ase.io
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import optax
import torch
import yaml
from jax import export as jax_export

jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Conditionally import kUPS components (graceful fallback for standalone use)
try:
    from kups.relaxation.optax import make_optimizer
except ImportError:
    make_optimizer = None

try:
    from fairchem.core.datasets.atomic_data import AtomicData

    AtomicData.validate = lambda self: None
except ImportError:
    AtomicData = None


# ─────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────
def load_jaxified_model(model_path: str | Path):
    """Load a jaxified MLFF model from a kUPS-format zip archive."""
    model_path = Path(model_path)
    if str(model_path).startswith("hf://"):
        from huggingface_hub import hf_hub_download

        parts = str(model_path)[5:].split("/", 2)
        model_path = Path(
            hf_hub_download(repo_id=f"{parts[0]}/{parts[1]}", filename=parts[2])
        )

    with zipfile.ZipFile(model_path) as zf:
        exp = jax_export.deserialize(bytearray(zf.read("model.jax")))
        raw = msgpack.unpackb(zf.read("params.msgpack"), raw=False)
        params = [
            jnp.array(
                np.frombuffer(e["data"], dtype=np.dtype(e["dtype"]))
                .reshape(e["shape"])
                .copy()
            )
            for e in raw
        ]
        metadata = json.loads(zf.read("metadata.json"))
    return exp, params, metadata


# ─────────────────────────────────────────────────────────────────────────
# Structure loading and graph construction
# ─────────────────────────────────────────────────────────────────────────
def load_molecules(inp_files, radius=6.0, max_neigh=300):
    """Load structures and build per-molecule graph data."""
    torch.set_default_device("cpu")
    mol_data = []
    for f in inp_files:
        atoms = ase.io.read(str(f))
        if not any(atoms.pbc):
            atoms.center(vacuum=5.0)
            atoms.pbc = True
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin", 1)
        data = AtomicData.from_ase(
            atoms,
            task_name="omol",
            r_edges=True,
            r_data_keys=["spin", "charge"],
            max_neigh=max_neigh,
            radius=radius,
            target_dtype=torch.float32,
        )
        mol_data.append(
            {
                "name": Path(f).stem,
                "n_atoms": len(atoms),
                "pos": data.pos.numpy(),
                "atomic_numbers": data.atomic_numbers.numpy(),
                "cell": data.cell.numpy(),
                "pbc": data.pbc.numpy(),
                "edge_index": data.edge_index.numpy(),
                "cell_offsets": data.cell_offsets.numpy(),
                "batch": data.batch.numpy(),
                "charge": data.charge.numpy(),
                "spin": data.spin.numpy(),
            }
        )
    return mol_data


def build_batched_graph(mol_data):
    """Concatenate per-molecule data into a single batched graph."""
    B = len(mol_data)
    system_counts = np.array([m["n_atoms"] for m in mol_data])
    N_total = int(system_counts.sum())
    max_atoms = int(system_counts.max())

    all_pos = np.concatenate([m["pos"] for m in mol_data])
    all_an = np.concatenate([m["atomic_numbers"] for m in mol_data])
    all_cell = np.concatenate([m["cell"] for m in mol_data])
    all_pbc = np.concatenate([m["pbc"] for m in mol_data])
    all_charge = np.concatenate([m["charge"] for m in mol_data])
    all_spin = np.concatenate([m["spin"] for m in mol_data])
    batch_idx = np.concatenate(
        [np.full(m["n_atoms"], i) for i, m in enumerate(mol_data)]
    )

    atom_offset = 0
    ei_list, co_list = [], []
    for m in mol_data:
        ei_list.append(m["edge_index"] + atom_offset)
        co_list.append(m["cell_offsets"])
        atom_offset += m["n_atoms"]

    static_graph = {
        "atomic_numbers": jnp.array(all_an),
        "cell": jnp.array(all_cell, dtype=jnp.float32),
        "pbc": jnp.array(all_pbc),
        "edge_index": jnp.array(np.concatenate(ei_list, axis=1)),
        "cell_offsets": jnp.array(np.concatenate(co_list), dtype=jnp.float32),
        "batch": jnp.array(batch_idx),
        "charge": jnp.array(all_charge),
        "spin": jnp.array(all_spin),
    }
    positions = jnp.array(all_pos, dtype=jnp.float32)

    # Static index arrays for JIT-friendly split/pad/scatter
    system_offsets = np.concatenate([[0], np.cumsum(system_counts[:-1])])
    gather_indices = np.full((B, max_atoms), N_total, dtype=np.int32)
    for i in range(B):
        gather_indices[i, : system_counts[i]] = np.arange(
            system_offsets[i], system_offsets[i] + system_counts[i]
        )
    grad_mask = np.zeros((B, max_atoms, 1))
    for i in range(B):
        grad_mask[i, : system_counts[i]] = 1.0
    fmax_mask = np.zeros((B, max_atoms), dtype=bool)
    for i in range(B):
        fmax_mask[i, : system_counts[i]] = True

    return (
        static_graph,
        positions,
        B,
        N_total,
        max_atoms,
        system_counts,
        jnp.array(gather_indices),
        jnp.array(grad_mask),
        jnp.array(fmax_mask),
    )


# ─────────────────────────────────────────────────────────────────────────
# JIT'd optimization step
# ─────────────────────────────────────────────────────────────────────────
def make_full_step(energy_fn, static_graph, optimizer, gather_idx, grad_mask,
                   fmax_mask, N_total):
    """Build a fully JIT'd optimization step function.

    The returned function compiles gradient computation, per-system
    splitting, vmapped LBFGS update, and scatter-back into one kernel.
    """
    sg = static_graph
    vmapped_update = jax.vmap(lambda g, s, p: optimizer.update(g, s, p))

    @jax.jit
    def step(positions, opt_states, freeze_mask):
        def energy_sum(pos):
            return jnp.sum(energy_fn({**sg, "pos": pos}))

        _, grad = jax.value_and_grad(energy_sum)(positions)
        per_sys_e = energy_fn({**sg, "pos": positions})

        grad_ext = jnp.concatenate([grad, jnp.zeros((1, 3), dtype=grad.dtype)])
        padded_grad = grad_ext[gather_idx] * grad_mask * freeze_mask

        pos_ext = jnp.concatenate(
            [positions, jnp.zeros((1, 3), dtype=positions.dtype)]
        )
        padded_pos = pos_ext[gather_idx]

        force_norms = jnp.linalg.norm(-padded_grad, axis=-1)
        force_norms = jnp.where(fmax_mask, force_norms, 0.0)
        per_mol_fmax = jnp.max(force_norms, axis=-1)

        padded_updates, new_opt_states = vmapped_update(
            padded_grad, opt_states, padded_pos
        )
        padded_updates = padded_updates * freeze_mask

        B_local, max_a, _ = padded_updates.shape
        flat_updates = padded_updates.reshape(B_local * max_a, 3)
        flat_indices = gather_idx.reshape(B_local * max_a)
        real_mask = flat_indices < N_total
        safe_idx = jnp.where(real_mask, flat_indices, 0)
        new_positions = positions.at[safe_idx].add(flat_updates * real_mask[:, None])

        return new_positions.astype(jnp.float32), new_opt_states, per_sys_e, per_mol_fmax

    return step


# ─────────────────────────────────────────────────────────────────────────
# Main relaxation loop
# ─────────────────────────────────────────────────────────────────────────
def run_relaxation(config_path: str | Path):
    """Run batched relaxation with per-molecule vmapped LBFGS."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    max_steps = config["run"]["max_steps"]
    fmax_tol = config["run"]["force_tolerance"]
    out_file = config["run"].get("out_file", "relax_results.json")

    # Load model
    logging.info("Loading model...")
    exp, params, metadata = load_jaxified_model(config["model_path"])
    cutoff = metadata.get("cutoff", 6.0)

    def energy_fn(data):
        return exp.call(params, data)

    # Load structures and build batched graph
    logging.info("Loading structures...")
    mol_data = load_molecules(config["inp_files"], radius=cutoff)
    (sg, positions, B, N_total, max_atoms, system_counts,
     gather_idx, grad_mask, fmax_mask) = build_batched_graph(mol_data)

    logging.info(
        f"{B} molecules, {N_total} total atoms, max_atoms={max_atoms}"
    )

    # Build optimizer
    if make_optimizer is not None:
        optimizer = make_optimizer(config["relax"]["optimizer"])
    else:
        optimizer = optax.chain(
            optax.scale_by_lbfgs(memory_size=20),
            optax.scale(-1.0),
        )

    # Initialize per-molecule optimizer states
    pos_ext = jnp.concatenate(
        [positions, jnp.zeros((1, 3), dtype=positions.dtype)]
    )
    padded_pos_init = pos_ext[gather_idx]
    all_opt_states = jax.vmap(optimizer.init)(padded_pos_init)

    # Build JIT'd step function
    full_step = make_full_step(
        energy_fn, sg, optimizer, gather_idx, grad_mask, fmax_mask, N_total
    )

    # Compile
    logging.info("Compiling JIT'd step...")
    t0 = time.time()
    freeze_mask = jnp.ones((B, 1, 1))
    new_pos, new_ost, pse, pmf = full_step(positions, all_opt_states, freeze_mask)
    jax.block_until_ready(pmf)
    compile_time = time.time() - t0
    logging.info(f"Compiled in {compile_time:.1f}s")

    # Speed check
    times = []
    pos_tmp, ost_tmp = positions, all_opt_states
    for _ in range(10):
        t0 = time.perf_counter()
        pos_tmp, ost_tmp, _, _ = full_step(pos_tmp, ost_tmp, freeze_mask)
        jax.block_until_ready(pos_tmp)
        times.append(time.perf_counter() - t0)
    ms_per_step = sorted(times)[5] * 1000
    logging.info(f"Per-step: {ms_per_step:.1f}ms for {B} molecules")

    # Reset state
    positions = jnp.array(
        np.concatenate([m["pos"] for m in mol_data]), dtype=jnp.float32
    )
    pos_ext = jnp.concatenate(
        [positions, jnp.zeros((1, 3), dtype=positions.dtype)]
    )
    all_opt_states = jax.vmap(optimizer.init)(pos_ext[gather_idx])

    # Main loop
    converged = [False] * B
    converged_step = [-1] * B
    final_energy = np.zeros(B)
    final_fmax = np.full(B, 999.0)
    freeze_arr = np.ones(B)

    logging.info(
        f"\nOptimizing {B} molecules, fmax<{fmax_tol}, max {max_steps} steps"
    )
    t_start = time.time()

    for step in range(max_steps):
        freeze_mask = jnp.array(freeze_arr)[:, None, None]
        positions, all_opt_states, per_sys_e, per_mol_fmax = full_step(
            positions, all_opt_states, freeze_mask
        )

        fmax_np = np.asarray(jax.block_until_ready(per_mol_fmax))
        energy_np = np.asarray(per_sys_e)

        all_done = True
        for i in range(B):
            if converged[i]:
                continue
            final_energy[i] = energy_np[i]
            final_fmax[i] = fmax_np[i]
            if fmax_np[i] < fmax_tol:
                converged[i] = True
                converged_step[i] = step
                freeze_arr[i] = 0.0
            else:
                all_done = False

        if all_done:
            break

        if step % 25 == 0:
            nc = sum(converged)
            afmax = [final_fmax[i] for i in range(B) if not converged[i]]
            if afmax:
                logging.info(
                    f"  Step {step:>3} ({time.time()-t_start:>5.1f}s): "
                    f"{nc}/{B} converged, "
                    f"fmax=[{min(afmax):.4f}, {max(afmax):.4f}]"
                )

    t_total = time.time() - t_start
    nc = sum(converged)

    # Results
    logging.info(f"\n{'='*80}")
    logging.info(f"  {nc}/{B} converged in {t_total:.1f}s ({ms_per_step:.1f}ms/step)")
    logging.info(f"{'='*80}")
    logging.info(
        f"{'Molecule':<28} {'Atoms':>5} {'Steps':>6} {'fmax':>10} "
        f"{'Energy (eV)':>14} {'Conv':>5}"
    )
    logging.info("-" * 80)
    for i, m in enumerate(mol_data):
        st = converged_step[i] if converged[i] else max_steps
        c = "Y" if converged[i] else "N"
        logging.info(
            f"{m['name']:<28} {m['n_atoms']:>5} {st:>6} "
            f"{final_fmax[i]:>10.6f} {final_energy[i]:>14.4f} {c:>5}"
        )

    results = [
        {
            "name": mol_data[i]["name"],
            "n_atoms": mol_data[i]["n_atoms"],
            "steps": converged_step[i] if converged[i] else max_steps,
            "fmax": float(final_fmax[i]),
            "energy": float(final_energy[i]),
            "converged": converged[i],
        }
        for i in range(B)
    ]
    output = {
        "wall_time_s": t_total,
        "compile_time_s": compile_time,
        "per_step_ms": ms_per_step,
        "converged": nc,
        "total": B,
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"\nResults saved to {out_file}")
    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} config.yaml")
        sys.exit(1)
    run_relaxation(sys.argv[1])
