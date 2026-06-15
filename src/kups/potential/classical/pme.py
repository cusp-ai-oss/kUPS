# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Smooth Particle-Mesh Ewald (PME) reciprocal-space potential — pure JAX.

PME replaces the direct-Ewald reciprocal term (``ewald_long_range_energy``,
``O(N * N_k)`` with a dense ``(N, N_k, 2)`` structure-factor tensor that OOMs at
large ``N``) with an ``O(N log N)`` FFT-based evaluation that scales to >=100k
ions. It is a drop-in at the long-range seam: same ``EwaldLongRangeInput``
signature, same ``EwaldLongRangeComposer``, same ``PositionAndCell`` gradient
lens — so forces (``dE/dr``) and the NPT stress (``dE/dcell``) come from
``jax.vjp`` for free, exactly as for direct Ewald (no custom JVP/VJP).

Convention matches ``ewald.py`` so PME converges to the same number::

    P(k) = (2*pi/V) * exp(-k^2 / (4*alpha^2)) / k^2 ,   k = 2*pi * inverse_vectors^T @ n
    E_recip = sum_{k != 0} P(k) * B(m) * |FFT(Q)(m)|^2 * TO_STANDARD_UNITS

where ``Q`` is the order-``p`` cardinal B-spline charge mesh and ``B(m)`` is the
Euler exponential-spline aliasing correction (Essmann et al. 1995). ``alpha`` and
the real-space cutoff are reused verbatim from ``EwaldParameters`` so the
short-range / self / exclusion terms are unchanged and the splitting is
consistent. The FFT mesh dimensions are static (chosen once at construction);
under NPT the box fluctuates but the mesh stays fixed, as in standard PME.

Validated to rtol ~1e-5 (fp64) vs ``ewald_long_range_energy`` on the NaCl
Madelung system; see ``test/potential/test_pme.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from kups.core.cell import Periodic3D
from kups.core.data import Table, WithIndices
from kups.core.lens import Lens, NestedLens, SimpleLens, View
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import EMPTY_LENS, Energy, Potential, PotentialOut
from kups.core.typing import HasCell, ParticleId, SystemId
from kups.potential.classical.ewald import (
    TO_STANDARD_UNITS,
    EwaldCache,
    EwaldLongRangeComposer,
    EwaldLongRangeInput,
    EwaldParameters,
    IsEwaldPointData,
)
from kups.potential.common.energy import PotentialFromEnergy
from kups.potential.common.graph import PointCloud

DEFAULT_SPLINE_ORDER = 6
"""Default cardinal B-spline order. p=6 reaches ~1e-5 at ~1 Angstrom spacing."""


def pme_mesh_for_cell(
    cell_vectors: Array, spacing: float = 1.0, multiple_of: int = 2
) -> tuple[int, int, int]:
    """Pick static PME mesh dimensions for a reference cell (~``spacing`` Angstrom).

    Mesh dims are rounded up to a multiple of ``multiple_of`` (cheaper FFTs).
    Use the *initial* cell; the mesh is held fixed across an NPT trajectory.
    """
    import numpy as np

    lengths = np.linalg.norm(np.asarray(cell_vectors), axis=-1)
    dims = []
    for length in lengths:
        m = int(np.ceil(float(length) / spacing))
        if multiple_of > 1:
            m = ((m + multiple_of - 1) // multiple_of) * multiple_of
        dims.append(max(m, multiple_of))
    return (dims[0], dims[1], dims[2])


def _bspline_weights(frac: Array, order: int) -> Array:
    """Order-``p`` cardinal B-spline weights for the ``p`` nearest grid points.

    ``frac`` is the fractional part (``[0, 1)``) of the scaled grid coordinate.
    Returns ``frac.shape + (order,)`` weights for offsets ``j = 0..order-1`` (grid
    point ``floor(u) - j``). Weights are smooth in ``frac`` (so gradients flow) and
    form a partition of unity (sum to 1).
    """
    j = jnp.arange(order)
    x = frac[..., None] + j  # M_n evaluated at f, f+1, ..., f+(p-1)
    w = jnp.where((x >= 0) & (x < 1), 1.0, 0.0)  # M_1
    for k in range(2, order + 1):
        # M_k(x) = x/(k-1) M_{k-1}(x) + (k-x)/(k-1) M_{k-1}(x-1); shift gives M_{k-1}(x-1)
        w_shift = jnp.concatenate([jnp.zeros_like(w[..., :1]), w[..., :-1]], axis=-1)
        w = (x * w + (k - x) * w_shift) / (k - 1)
    return w


def _euler_modulus_sq(m_size: int, order: int, dtype) -> Array:
    """``|b(m)|^2`` Euler exponential-spline aliasing factor for one axis."""
    mvals = _bspline_weights(jnp.zeros((), dtype=dtype), order)  # [M_n(0..p-1)]
    m = jnp.arange(m_size)
    k = jnp.arange(order - 1)
    phase = jnp.exp(2j * jnp.pi * m[:, None] * k[None, :] / m_size)
    denom = jnp.sum(mvals[1:order][None, :] * phase, axis=-1)
    denom2 = jnp.abs(denom) ** 2
    return jnp.where(denom2 < 1e-10, 1.0, 1.0 / jnp.maximum(denom2, 1e-12))


def _pme_reciprocal_energy_batched(
    positions: Array,  # (N, 3) Cartesian, Angstrom
    charges: Array,  # (N,)
    sys_idx: Array,  # (N,) per-particle system index
    n_sys: int,  # number of systems (static)
    inverse_vectors: Array,  # (n_sys, 3, 3) (cell.inverse_vectors; differentiable in cell.vectors)
    volume: Array,  # (n_sys,)
    alpha: Array,  # (n_sys,)
    mesh: tuple[int, int, int],
    order: int,
    fp32_reciprocal: bool = False,
) -> Array:
    """Per-system smooth-PME reciprocal energy (atomic units; xTO_STANDARD_UNITS later).

    ``fp32_reciprocal`` runs the entire reciprocal pipeline (spread, FFT, influence, gather)
    in single precision and casts the energy back to the input dtype at the end — a large
    speedup where fp64 FFT/scatter/exp are slow. The reciprocal term is small and
    well-conditioned (|F|^2 differs ~6e-8 fp32-vs-fp64), so total energy/force accuracy is
    unaffected in practice; forces and the NPT cell-virial from this term are then
    fp32-computed, fp64-typed (autodiff flows back through the entry casts).
    """
    out_dtype = positions.dtype
    if fp32_reciprocal:
        positions = positions.astype(jnp.float32)
        charges = charges.astype(jnp.float32)
        inverse_vectors = inverse_vectors.astype(jnp.float32)
        volume = volume.astype(jnp.float32)
        alpha = alpha.astype(jnp.float32)
    dtype = positions.dtype
    Mx, My, Mz = mesh
    Mvec = jnp.asarray(mesh, dtype=dtype)

    # Fractional coords using each particle's own system cell: s = inv @ r.
    inv_p = inverse_vectors[sys_idx]  # (N, 3, 3)
    frac = jnp.einsum("nab,nb->na", inv_p, positions) % 1.0  # (N, 3) in [0, 1)
    u = frac * Mvec
    base = jnp.floor(u).astype(jnp.int32)
    f = u - base

    wx = _bspline_weights(f[:, 0], order)  # (N, p)
    wy = _bspline_weights(f[:, 1], order)
    wz = _bspline_weights(f[:, 2], order)
    j = jnp.arange(order)
    gx = (base[:, 0:1] - j) % Mx  # (N, p)
    gy = (base[:, 1:2] - j) % My
    gz = (base[:, 2:3] - j) % Mz

    # Scatter charges onto a per-system mesh Q of shape (n_sys, Mx, My, Mz).
    w3 = (
        charges[:, None, None, None]
        * wx[:, :, None, None]
        * wy[:, None, :, None]
        * wz[:, None, None, :]
    )
    flat = (
        sys_idx[:, None, None, None] * (Mx * My * Mz)
        + gx[:, :, None, None] * (My * Mz)
        + gy[:, None, :, None] * Mz
        + gz[:, None, None, :]
    )  # (N, p, p, p)
    Q = (
        jnp.zeros((n_sys * Mx * My * Mz,), dtype=dtype)
        .at[flat.reshape(-1)]
        .add(w3.reshape(-1))
    )
    Q = Q.reshape(n_sys, Mx, My, Mz)
    Fq = jnp.fft.fftn(Q, axes=(1, 2, 3))  # (n_sys, Mx, My, Mz) complex

    # Per-system k-vectors: k = 2 pi * inverse_vectors^T @ n  (matches ewald.py kvecs).
    nx = jnp.fft.fftfreq(Mx) * Mx
    ny = jnp.fft.fftfreq(My) * My
    nz = jnp.fft.fftfreq(Mz) * Mz
    nX, nY, nZ = jnp.meshgrid(nx, ny, nz, indexing="ij")
    n_idx = jnp.stack([nX, nY, nZ], axis=-1).astype(dtype)  # (Mx,My,Mz,3)
    kvec = 2.0 * jnp.pi * jnp.einsum("sba,xyzb->sxyza", inverse_vectors, n_idx)
    k2 = jnp.sum(kvec**2, axis=-1)  # (n_sys, Mx, My, Mz)

    bsq = (
        _euler_modulus_sq(Mx, order, dtype)[:, None, None]
        * _euler_modulus_sq(My, order, dtype)[None, :, None]
        * _euler_modulus_sq(Mz, order, dtype)[None, None, :]
    )  # (Mx,My,Mz)

    nonzero = k2 > 0
    k2_safe = jnp.where(nonzero, k2, 1.0)
    pref = (
        (2.0 * jnp.pi)
        / volume[:, None, None, None]
        * jnp.exp(-k2_safe / (4.0 * alpha[:, None, None, None] ** 2))
        / k2_safe
    )
    influence = jnp.where(nonzero, pref * bsq[None], 0.0)
    energy = jnp.sum(influence * (jnp.abs(Fq) ** 2), axis=(1, 2, 3))  # (n_sys,)
    return energy.astype(out_dtype)


def make_pme_long_range_energy(
    mesh: tuple[int, int, int], order: int, fp32_reciprocal: bool = False
):
    """Build the PME long-range energy fn for a fixed static ``mesh`` and ``order``.

    ``fp32_reciprocal=True`` evaluates the reciprocal pipeline in single precision (see
    ``_pme_reciprocal_energy_batched``).
    """

    def pme_long_range_energy[State](
        inp: EwaldLongRangeInput[State],
    ) -> WithPatch[Table[SystemId, Energy], Patch[State]]:
        """Reciprocal-space (long-range) PME energy. Drop-in for ``ewald_long_range_energy``."""
        pc = inp.point_cloud
        cell = pc.systems.data.cell
        n_sys = pc.batch_size
        energy = _pme_reciprocal_energy_batched(
            positions=pc.particles.data.positions,
            charges=pc.particles.data.charges,
            sys_idx=pc.particles.data.system.indices,
            n_sys=n_sys,
            inverse_vectors=cell.inverse_vectors,
            volume=cell.volume,
            alpha=inp.parameters.alpha.data,
            mesh=mesh,
            order=order,
            fp32_reciprocal=fp32_reciprocal,
        )
        assert energy.shape == (n_sys,), (
            f"Expected energy shape {(n_sys,)} but got {energy.shape}."
        )
        energy = energy * TO_STANDARD_UNITS
        return WithPatch(Table.arange(energy, label=SystemId), IdPatch())

    return pme_long_range_energy


def make_pme_long_range_potential[State, Ptch: Patch, Gradients, Hessians](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasCell[Periodic3D]]],
    parameter_lens: Lens[State, EwaldParameters],
    cache_lens: Lens[State, EwaldCache] | None,
    probe: Probe[State, Ptch, WithIndices[ParticleId, IsEwaldPointData]] | None = None,
    gradient_lens: Lens[
        PointCloud[IsEwaldPointData, HasCell[Periodic3D]], Gradients
    ] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[State, Hessians] = EMPTY_LENS,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    *,
    mesh: tuple[int, int, int],
    order: int = DEFAULT_SPLINE_ORDER,
    fp32_reciprocal: bool = False,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create the PME reciprocal-space (long-range) potential.

    Mirrors ``make_ewald_long_range_potential`` exactly (same composer, same
    point-cloud gradient lens) so forces and NPT stress come from autodiff.
    ``cache_lens`` is ignored (PME for MD needs no structure-factor cache;
    ``PotentialFromEnergy`` emits an ``IdPatch``). ``mesh`` is the static FFT grid.
    ``fp32_reciprocal`` runs the reciprocal pipeline in single precision.
    """
    return PotentialFromEnergy(
        energy_fn=make_pme_long_range_energy(mesh, order, fp32_reciprocal),
        composer=EwaldLongRangeComposer(
            particles=particles_view,
            systems=systems_view,
            probe=probe,
            parameters=parameter_lens,
            cache=None,
        ),
        gradient_lens=NestedLens(
            SimpleLens[EwaldLongRangeInput, PointCloud](
                lambda state: state.point_cloud
            ),
            gradient_lens,
        ),
        hessian_lens=hessian_lens,
        cache_lens=None,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
