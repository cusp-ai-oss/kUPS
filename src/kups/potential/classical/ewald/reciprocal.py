# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Reciprocal structure-factor kernels independent of Ewald caches and potentials.

``_structure_factor_full`` selects factored phase grids on CPU, direct sums on
GPU, or bounded explicit tiles. The tiled path also bounds derivative workspace
when ``tiled=True``. Each kernel accepts signed charges for incremental deltas.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.data import Index
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, is_traced
from kups.core.utils.kahan import KahanSummand
from kups.core.utils.segment import segment_sum

from .parameters import (
    _DEFAULT_RECIPROCAL_K_CHUNK,
    _DEFAULT_RECIPROCAL_PARTICLE_CHUNK,
    RECIPROCAL_GRID_BACKENDS,
    ReciprocalGridBound,
)

# CPU grids retain a complete grid per particle.
_CPU_GRID_PARTICLE_CHUNK = 256


def _frequency_response(
    positions: Array,
    charges: Array,
    kvecs: Array,
    batch_mask: Index[SystemId],
) -> Array:
    """Per-particle ``q * [cos(k·r), sin(k·r)]``, shaped (particles, k-vectors, 2)."""
    exponent = einops.einsum(
        kvecs[batch_mask.indices],
        positions,
        "particles shifts dim, particles dim->particles shifts",
    )
    return charges[:, None, None] * jnp.stack(
        [jnp.cos(exponent), jnp.sin(exponent)], axis=-1
    )


def _phase_factors_grid(
    positions: Array,
    inverse_vectors: Array,
    batch_mask: Index[SystemId],
    bound: ReciprocalGridBound,
) -> tuple[Array, Array]:
    """Factor ``exp(i k·r)`` into xy phases and z rotations.

    Trigonometric work grows with the sum of the axis lengths. Keeping the
    final contraction separate lets a single system sum particles inside it.
    """
    sys_idx = batch_mask.indices
    phi = (
        2
        * jnp.pi
        * einops.einsum(
            positions,
            inverse_vectors[sys_idx],
            "particles d, particles d a -> particles a",
        )
    )
    angles = [
        phi[:, axis, None] * jnp.arange(0 if axis == 0 else -b, b + 1, dtype=phi.dtype)
        for axis, b in enumerate(bound)
    ]
    cx, cy, cz = map(jnp.cos, angles)
    sx, sy, sz = map(jnp.sin, angles)

    # Small dots materialize phase tables; elementwise products repeat the trig.
    def rotation(c: Array, sn: Array) -> Array:
        return jnp.stack(
            [jnp.stack([c, sn], axis=-1), jnp.stack([-sn, c], axis=-1)], axis=-2
        )

    xy = jnp.einsum("pxk,pykc->pxyc", jnp.stack([cx, sx], axis=-1), rotation(cy, sy))
    return xy, rotation(cz, sz)


def _frequency_response_grid(
    positions: Array,
    charges: Array,
    inverse_vectors: Array,
    batch_mask: Index[SystemId],
    bound: ReciprocalGridBound,
) -> Array:
    """Per-particle responses in ``reciprocal_grid_shifts`` order, real/imag last."""
    xy, z = _phase_factors_grid(positions, inverse_vectors, batch_mask, bound)
    xyz = jnp.einsum("pxyk,pzkc->pxyzc", xy, z)
    n = positions.shape[0]
    return charges[:, None, None] * xyz.reshape(n, -1, 2)


def _use_grid_response(bound: ReciprocalGridBound) -> bool:
    return (
        any(b > 0 for b in bound) and jax.default_backend() in RECIPROCAL_GRID_BACKENDS
    )


def _structure_factor_grid(
    positions: Array,
    charges: Array,
    inverse_vectors: Array,
    batch_mask: Index[SystemId],
    bound: ReciprocalGridBound,
    particle_chunk_size: int,
    k_chunk_size: int,
) -> Array:
    """Sum complete reciprocal grids in particle chunks within the workspace budget."""
    n_particles, n_systems = len(positions), batch_mask.num_labels
    bx, by, bz = bound
    n_kvecs = (bx + 1) * (2 * by + 1) * (2 * bz + 1)
    capacity = min(
        n_particles,
        _CPU_GRID_PARTICLE_CHUNK,
        particle_chunk_size,
        max(1, particle_chunk_size * k_chunk_size // n_kvecs),
    )
    num_chunks = (n_particles + capacity - 1) // capacity
    chunk = (n_particles + num_chunks - 1) // num_chunks
    pad = -n_particles % chunk
    position_tiles = jnp.pad(positions, ((0, pad), (0, 0))).reshape(-1, chunk, 3)
    charge_tiles = jnp.pad(charges, (0, pad)).reshape(-1, chunk)
    system_tiles = jnp.pad(
        batch_mask.indices, (0, pad), constant_values=n_systems
    ).reshape(-1, chunk)
    dtype = jnp.result_type(positions, charges, inverse_vectors)

    def add_particles(
        acc: Array, tile: tuple[Array, Array, Array]
    ) -> tuple[Array, None]:
        positions, charges, system_ids = tile
        system = Index(batch_mask.keys, jnp.minimum(system_ids, n_systems - 1))
        if n_systems == 1:
            xy, z = _phase_factors_grid(positions, inverse_vectors, system, bound)
            # Contract the particle dimension before materializing the grid.
            # This avoids both a particle-by-grid response and a scatter.
            qz = jnp.where(system_ids == 0, charges, 0)[:, None, None, None] * z
            sf = jnp.einsum("pxyk,pzkc->xyzc", xy, qz).reshape(1, n_kvecs, 2)
            return acc + sf, None
        response = _frequency_response_grid(
            positions,
            charges,
            inverse_vectors,
            system,
            bound,
        )
        return acc + segment_sum(response, system_ids, n_systems, mode="drop"), None

    sf, _ = jax.lax.scan(
        jax.checkpoint(add_particles, prevent_cse=False),
        jnp.zeros((n_systems, n_kvecs, 2), dtype),
        (position_tiles, charge_tiles, system_tiles),
    )
    return sf


def _concrete_grid_index(
    grid_index: Array | np.ndarray | None,
) -> np.ndarray | None:
    """Return trace-time grid positions, or ``None`` when they are traced."""
    if isinstance(grid_index, np.ndarray):
        return grid_index
    if grid_index is None or is_traced(grid_index):
        return None
    return np.asarray(grid_index)


def _cropped_grid_layout(
    grid_index: np.ndarray, bound: ReciprocalGridBound
) -> tuple[np.ndarray, slice, np.ndarray]:
    """Occupied xy rows and z range covering one system's k-vectors.

    Returns flat xy row indices, the z slice and each k-vector's position in
    the cropped ``(rows, z)`` grid.
    """
    nz = 2 * bound[2] + 1
    rows, row_of_k = np.unique(grid_index // nz, return_inverse=True)
    z = grid_index % nz
    z0, z1 = int(z.min()), int(z.max()) + 1
    position = row_of_k.reshape(-1) * (z1 - z0) + (z - z0)
    return rows, slice(z0, z1), position


def _integer_phase_powers(cos: Array, sin: Array, n: np.ndarray) -> Array:
    """``(cos nφ, sin nφ)`` for static non-negative ``n`` from ``(cos φ, sin φ)``.

    Args:
        cos: Cosines of the unit phases, shaped ``(p,)``.
        sin: Sines of the unit phases, shaped ``(p,)``.
        n: Non-negative integer multiples, shaped ``(m,)``.

    Returns:
        Cosines and sines of every multiple, shaped ``(p, m, 2)``.
    """
    n = np.asarray(n, dtype=np.int64)
    assert (n >= 0).all(), "phase multiples must be non-negative"
    out_c = jnp.ones((len(cos), len(n)), cos.dtype)
    out_s = jnp.zeros((len(cos), len(n)), cos.dtype)
    pc, ps = cos[:, None], sin[:, None]
    for b in range(int(n.max(initial=0)).bit_length()):
        bit = ((n >> b) & 1).astype(bool)
        if bit.any():
            out_c, out_s = (
                jnp.where(bit, out_c * pc - out_s * ps, out_c),
                jnp.where(bit, out_c * ps + out_s * pc, out_s),
            )
        pc, ps = pc * pc - ps * ps, 2 * pc * ps
    return jnp.stack((out_c, out_s), axis=-1)


def _structure_factor_cropped(
    positions: Array,
    charges: Array,
    inverse_vectors: Array,
    batch_mask: Index[SystemId],
    bound: ReciprocalGridBound,
    grid_index: np.ndarray,
) -> Array:
    """One system's structure factor on only the occupied part of the phase grid.

    One contraction over particles yields the occupied xy rows times the
    occupied z range, from which the stored k-vectors are gathered. Intended
    for few particles, e.g. incremental moves.
    """
    rows, z_range, position = _cropped_grid_layout(grid_index.reshape(-1), bound)
    bx, by, bz = bound
    phi = 2 * jnp.pi * positions @ inverse_vectors[0]
    q = jnp.where(batch_mask.indices == 0, charges, 0)

    # Integer multiples of the unit phases via binary powers.
    cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)

    def phases(axis: int, n: np.ndarray) -> Array:
        return _integer_phase_powers(cos_phi[:, axis], sin_phi[:, axis], n)

    def signed_phases(axis: int, n: np.ndarray) -> Array:
        """Phases of signed ``n`` from those of ``|n|`` (odd sine)."""
        sign = np.stack((np.ones(len(n)), np.sign(n)), axis=-1)
        return phases(axis, np.abs(n)) * sign.astype(phi.dtype)

    x = phases(0, np.arange(bx + 1))
    y = signed_phases(1, np.arange(-by, by + 1))
    z = q[:, None, None] * signed_phases(2, np.arange(-bz, bz + 1)[z_range])
    x, y, z = jax.lax.optimization_barrier((x, y, z))
    xr, xi = x[:, :, None, 0], x[:, :, None, 1]
    yr, yi = y[:, None, :, 0], y[:, None, :, 1]
    xy = jnp.stack((xr * yr - xi * yi, xr * yi + xi * yr), axis=-1)
    xy = xy.reshape(len(positions), -1, 2)[:, rows]
    zr, zi = z[..., 0], z[..., 1]
    qz = jnp.stack(
        (jnp.stack((zr, zi), axis=-1), jnp.stack((-zi, zr), axis=-1)), axis=-2
    )
    grid = jnp.einsum("prk,pzkc->rzc", xy, qz).reshape(-1, 2)
    return grid[position][None]


def _structure_factor_full(
    positions: Array,
    charges: Array,
    kvecs: Array,
    inverse_vectors: Array | None = None,
    *,
    batch_mask: Index[SystemId],
    bound: ReciprocalGridBound = (0, 0, 0),
    particle_chunk_size: int = _DEFAULT_RECIPROCAL_PARTICLE_CHUNK,
    k_chunk_size: int = _DEFAULT_RECIPROCAL_K_CHUNK,
    tiled: bool = False,
    grid_index: Array | np.ndarray | None = None,
) -> Array:
    """Select CPU phase grids, fused GPU values, or tiled explicit sums.

    Signed charges allow the same reduction to evaluate incremental deltas.
    ``grid_index`` locates compact k-vectors inside the phase grid, shaped
    ``(n_systems, n_kvecs)``; ``None`` means the k-vectors enumerate the grid.
    """
    if particle_chunk_size < 1 or k_chunk_size < 1:
        raise ValueError("Reciprocal chunk sizes must be positive")
    n, ns, nk = len(positions), batch_mask.num_labels, kvecs.shape[1]
    dtype = jnp.result_type(positions, charges, kvecs)
    if n == 0 or ns == 0 or nk == 0:
        return jnp.zeros((ns, nk, 2), dtype)
    # Inactive slots can contain NaNs. Mask before trig and differentiation.
    valid = (batch_mask.indices >= 0) & (batch_mask.indices < ns)
    positions = jnp.where(valid[:, None], positions, 0)
    charges = jnp.where(valid, charges, 0)
    batch_mask = Index(batch_mask.keys, jnp.where(valid, batch_mask.indices, ns))
    if not tiled and _use_grid_response(bound) and inverse_vectors is not None:
        static_index = _concrete_grid_index(grid_index)
        if static_index is not None and ns == 1 and n <= _CPU_GRID_PARTICLE_CHUNK:
            return _structure_factor_cropped(
                positions, charges, inverse_vectors, batch_mask, bound, static_index
            ).astype(dtype)
        sf = _structure_factor_grid(
            positions,
            charges,
            inverse_vectors,
            batch_mask,
            bound,
            particle_chunk_size,
            k_chunk_size,
        ).astype(dtype)
        if grid_index is not None:
            sf = jnp.take_along_axis(sf, grid_index[..., None], axis=1)
        return sf
    if not tiled and jax.default_backend() in {"gpu", "cuda", "rocm"}:
        return segment_sum(
            _frequency_response(positions, charges, kvecs, batch_mask),
            batch_mask.indices,
            ns,
            mode="drop",
        )
    return _structure_factor_tiled(
        positions,
        charges,
        kvecs,
        batch_mask,
        particle_chunk_size=particle_chunk_size,
        k_chunk_size=k_chunk_size,
    )


@dataclass
class _ParticleTiles:
    """Sorted particle rows and the routing for each tile's partial system sums."""

    positions: Array  # (tiles, particles, 3)
    charges: Array  # (tiles, particles)
    system_ids: Array  # (tiles, particles): global system IDs for k-vector lookup
    local_ids: Array  # (tiles, particles): consecutive system IDs within each tile
    destinations: Array  # (tiles, systems_per_tile): global IDs for accumulation


def _sorted_particle_tiles(
    positions: Array,
    charges: Array,
    system_ids: Array,
    n_systems: int,
    chunk: int,
) -> _ParticleTiles:
    """Group particles by system and map each tile's local runs to global systems."""
    order = jnp.argsort(system_ids, stable=True)
    pad = -len(positions) % chunk
    positions = jnp.pad(positions[order], ((0, pad), (0, 0))).reshape(-1, chunk, 3)
    charges = jnp.pad(charges[order], (0, pad)).reshape(-1, chunk)
    system_ids = jnp.pad(
        system_ids[order], (0, pad), constant_values=n_systems
    ).reshape(-1, chunk)
    starts = jnp.concatenate(
        (
            jnp.ones_like(system_ids[:, :1], dtype=bool),
            system_ids[:, 1:] != system_ids[:, :-1],
        ),
        axis=1,
    )
    local_ids = jnp.cumsum(starts, axis=1, dtype=jnp.int32) - 1
    # A tile touches at most min(systems, particles) valid systems.
    destinations = jax.vmap(
        lambda ids, groups: (
            jnp.full(min(n_systems, chunk), n_systems, dtype=jnp.int32)
            .at[groups]
            .set(ids, mode="drop")
        )
    )(system_ids, local_ids)
    return _ParticleTiles(positions, charges, system_ids, local_ids, destinations)


def _accumulate_structure_factor(
    acc: KahanSummand[Array], partial: Array, destinations: Array
) -> KahanSummand[Array]:
    """Add a tile's partial sums to its distinct, valid system destinations."""
    old = jax.tree.map(lambda a: a.at[destinations].get(mode="fill", fill_value=0), acc)
    updated = old + partial
    # Compensation has zero tangent; stopping it avoids retaining every accumulator.
    updated = KahanSummand(updated.value, jax.lax.stop_gradient(updated.compensate))
    return jax.tree.map(
        lambda a, u: a.at[destinations].set(u, mode="drop", unique_indices=True),
        acc,
        updated,
    )


def _structure_factor_tiled(
    positions: Array,
    charges: Array,
    kvecs: Array,
    batch_mask: Index[SystemId],
    *,
    particle_chunk_size: int,
    k_chunk_size: int,
) -> Array:
    """Sum explicit k-vectors in bounded tiles, checkpointed for differentiation."""
    n, n_systems, n_k = len(positions), batch_mask.num_labels, kvecs.shape[1]
    dtype = jnp.result_type(positions, charges, kvecs)
    p_chunk, k_chunk = min(n, particle_chunk_size), min(n_k, k_chunk_size)
    sys_ids = batch_mask.indices.astype(jnp.int32)
    kvecs = jnp.pad(kvecs, ((0, 0), (0, -n_k % k_chunk), (0, 0)))
    pad = -n % p_chunk
    particle_tiles = (
        _sorted_particle_tiles(positions, charges, sys_ids, n_systems, p_chunk)
        if n > p_chunk and n_systems > p_chunk
        else (
            jnp.pad(positions, ((0, pad), (0, 0))).reshape(-1, p_chunk, 3),
            jnp.pad(charges, (0, pad)).reshape(-1, p_chunk),
            jnp.pad(sys_ids, (0, pad), constant_values=n_systems).reshape(-1, p_chunk),
        )
        if n > p_chunk
        else None
    )

    def sum_k_tile(k_start: Array) -> Array:
        kv = jax.lax.dynamic_slice_in_dim(kvecs, k_start, k_chunk, axis=1)

        def response(pos: Array, q: Array, ids: Array) -> Array:
            safe_ids = Index(batch_mask.keys, jnp.minimum(ids, n_systems - 1))
            return _frequency_response(pos, q, kv, safe_ids).astype(dtype)

        if particle_tiles is None:
            return segment_sum(
                response(positions, charges, sys_ids), sys_ids, n_systems, mode="drop"
            )

        def add_particles(
            acc: KahanSummand[Array], tile: _ParticleTiles | tuple[Array, Array, Array]
        ) -> tuple[KahanSummand[Array], None]:
            if isinstance(tile, _ParticleTiles):
                partial = segment_sum(
                    response(tile.positions, tile.charges, tile.system_ids),
                    tile.local_ids,
                    len(tile.destinations),
                    mode="drop",
                )
                return _accumulate_structure_factor(
                    acc, partial, tile.destinations
                ), None
            # These partial sums already have one row per system.
            positions, charges, ids = tile
            partial = segment_sum(
                response(positions, charges, ids), ids, n_systems, mode="drop"
            )
            updated = acc + partial
            # Compensation has zero tangent; do not retain each accumulator.
            return KahanSummand(
                updated.value, jax.lax.stop_gradient(updated.compensate)
            ), None

        acc, _ = jax.lax.scan(
            jax.checkpoint(add_particles, prevent_cse=False),
            KahanSummand.init(jnp.zeros((n_systems, k_chunk, 2), dtype=dtype)),
            particle_tiles,
        )
        return acc.total

    if n_k <= k_chunk_size:
        return sum_k_tile(jnp.int32(0))
    tiles = jax.lax.map(
        jax.checkpoint(sum_k_tile, prevent_cse=False),
        jnp.arange(0, n_k, k_chunk, dtype=jnp.int32),
    )
    return tiles.transpose(1, 0, 2, 3).reshape(n_systems, -1, 2)[:, :n_k]
