# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Smooth Particle-Mesh Ewald (PME) reciprocal-space potential.

An ``O(N log N)`` FFT-based replacement for the direct-Ewald reciprocal term
``ewald_long_range_energy``, whose dense ``(N, N_k, 2)`` structure-factor tensor
runs out of memory at large ``N``. It is a drop-in at the long-range seam, so
forces and the NPT stress still come from autodiff.

The influence function ``P(k)`` (``reciprocal_prefactor``), the neutralizing
background (``reciprocal_total_energy``), and ``alpha`` / the real-space cutoff
are shared with ``ewald.py``, so the short-range / self / exclusion terms are
unchanged, the splitting is consistent, and PME converges to the same number::

    E_recip = sum_{k != 0} P(k) * B(m) * |FFT(Q)(m)|^2 * TO_STANDARD_UNITS

where ``Q`` is the order-``p`` cardinal B-spline charge mesh and ``B(m)`` is the
Euler exponential-spline aliasing correction (Essmann et al. 1995).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from kups.core.cell import Cell, Periodic3D
from kups.core.data import Table, WithIndices
from kups.core.lens import Lens, View
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import EMPTY_LENS, Energy, Potential, PotentialOut
from kups.core.typing import HasCell, ParticleId, SystemId
from kups.core.utils.math import triangular_3x3_matmul
from kups.potential.classical.ewald import (
    EwaldCache,
    EwaldLongRangeInput,
    EwaldParameters,
    IsEwaldPointData,
    make_ewald_long_range_potential,
    reciprocal_prefactor,
    reciprocal_total_energy,
)
from kups.potential.common.graph import PointCloud

DEFAULT_SPLINE_ORDER = 6
"""Default cardinal B-spline order."""


@dataclasses.dataclass(frozen=True)
class PMESettings:
    """Select FFT-based PME for the reciprocal term instead of direct Ewald.

    Passed to `make_ewald_potential` / `make_ewald_from_state` as ``pme=``; leaving
    those ``None`` keeps the direct-Ewald reciprocal sum.

    Attributes:
        mesh: FFT grid dimensions, shared by every system in a batch and held
            fixed across a trajectory (size with `pme_mesh_for_cell`). Settings
            rather than traced state because FFT shapes must be static.
        order: Cardinal B-spline order.
    """

    mesh: tuple[int, int, int]
    order: int = DEFAULT_SPLINE_ORDER


def pme_mesh_for_cell(
    cell: Cell[Periodic3D], spacing: float = 1.0, multiple_of: int = 2
) -> tuple[int, int, int]:
    """Pick static PME mesh dimensions for a reference cell (~``spacing`` Angstrom).

    Resolution is measured across opposing faces (``cell.perpendicular_lengths``)
    so the spacing holds for sheared cells; a batched cell is reduced with a
    per-axis max. Dims are rounded up to a multiple of ``multiple_of`` (cheaper
    FFTs). Host-side by construction — FFT shapes must be static — so call it at
    setup time with the initial cell.

    The heuristic is purely geometric: the resolution PME needs also grows with
    the splitting parameter ``alpha``, so verify a chosen mesh against direct
    Ewald rather than trusting ``spacing`` alone.
    """
    lengths = np.asarray(cell.perpendicular_lengths).reshape(-1, 3).max(axis=0)
    dims: list[int] = []
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
    w = jnp.where((x >= 0) & (x < 1), 1.0, 0.0)  # M_1 = indicator of [0, 1)
    for k in range(2, order + 1):
        # Cox-de Boor: M_k(x) = [x M_{k-1}(x) + (k-x) M_{k-1}(x-1)] / (k-1); the
        # shift gives M_{k-1}(x-1). Each pass widens the support by one point, so
        # only j = 0 is nonzero above but all p offsets carry weight at the end.
        w_shift = jnp.concatenate([jnp.zeros_like(w[..., :1]), w[..., :-1]], axis=-1)
        w = (x * w + (k - x) * w_shift) / (k - 1)
    return w


def _euler_modulus_sq(m_size: int, order: int, dtype: DTypeLike) -> Array:
    """``|b(m)|^2`` Euler exponential-spline aliasing factor for one axis.

    For odd ``order`` the denominator vanishes on the Nyquist plane, where the true
    factor diverges; those modes are damped to 1 rather than left as inf. Even
    orders (the default) have no such plane.
    """
    mvals = _bspline_weights(jnp.zeros((), dtype=dtype), order)  # [M_n(0..p-1)]
    m = jnp.arange(m_size)
    k = jnp.arange(order - 1)
    phase = jnp.exp(2j * jnp.pi * m[:, None] * k[None, :] / m_size)
    denom = jnp.sum(mvals[1:order][None, :] * phase, axis=-1)
    denom2 = jnp.abs(denom) ** 2
    return jnp.where(denom2 < 1e-10, 1.0, 1.0 / denom2)


def _pme_reciprocal_energy(
    inp: EwaldLongRangeInput[Any], settings: PMESettings
) -> Array:
    """Per-system smooth-PME reciprocal energy, shape ``(n_sys,)``, atomic units."""
    pc = inp.point_cloud
    positions = pc.particles.data.positions
    charges = pc.particles.data.charges
    sys_idx = pc.particles.data.system.indices
    n_sys = pc.batch_size
    cell = pc.systems.data.cell
    alpha = inp.parameters.alpha[pc.systems.index]
    order = settings.order
    dtype = positions.dtype
    Mx, My, Mz = settings.mesh

    # Fractional coords in each particle's own cell, folded into [0, 1).
    cell_p = cell[sys_idx]
    frac, _ = cell_p.fold(cell_p.frame.to_fractional(positions))  # (N, 3)
    u = frac * jnp.asarray(settings.mesh, dtype=dtype)
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
    )  # (N, p, p, p)
    Q = (
        jnp.zeros((n_sys, Mx, My, Mz), dtype=dtype)
        .at[
            sys_idx[:, None, None, None],
            gx[:, :, None, None],
            gy[:, None, :, None],
            gz[:, None, None, :],
        ]
        .add(w3)
    )
    Fq = jnp.fft.fftn(Q, axes=(1, 2, 3))  # (n_sys, Mx, My, Mz) complex

    # Per-system k-vectors on the FFT frequency grid, as in EwaldLongRangeInput.kvecs.
    n_axes = [jnp.fft.fftfreq(m) * m for m in settings.mesh]  # integer mode numbers
    n_idx = jnp.stack(jnp.meshgrid(*n_axes, indexing="ij"), axis=-1).astype(dtype)
    kvec = triangular_3x3_matmul(
        cell.inverse_vectors.mT[:, None] * 2 * jnp.pi, n_idx.reshape(-1, 3), lower=False
    )
    k2 = jnp.sum(kvec**2, axis=-1).reshape(n_sys, Mx, My, Mz)

    bsq = (
        _euler_modulus_sq(Mx, order, dtype)[:, None, None]
        * _euler_modulus_sq(My, order, dtype)[None, :, None]
        * _euler_modulus_sq(Mz, order, dtype)[None, None, :]
    )  # (Mx,My,Mz)
    pref = reciprocal_prefactor(
        k2, inp.volume[:, None, None, None], alpha[:, None, None, None]
    )
    return jnp.sum(pref * bsq * jnp.abs(Fq) ** 2, axis=(1, 2, 3))  # (n_sys,)


def make_pme_long_range_energy(settings: PMESettings):
    """Build the PME long-range energy fn for fixed static ``settings``."""

    def pme_long_range_energy[State](
        inp: EwaldLongRangeInput[State],
    ) -> WithPatch[Table[SystemId, Energy], Patch[State]]:
        """Reciprocal-space (long-range) PME energy. Drop-in for ``ewald_long_range_energy``."""
        total = reciprocal_total_energy(inp, _pme_reciprocal_energy(inp, settings))
        return WithPatch(total, IdPatch())

    return pme_long_range_energy


def make_pme_long_range_potential[
    State,
    Ptch: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasCell[Periodic3D]]],
    parameter_lens: Lens[State, EwaldParameters],
    cache_lens: Lens[State, EwaldCache[Gradients, Hessians]] | None,
    probe: Probe[State, Ptch, WithIndices[ParticleId, IsEwaldPointData]] | None = None,
    gradient_lens: Lens[
        PointCloud[IsEwaldPointData, HasCell[Periodic3D]], Gradients
    ] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[State, Hessians] = EMPTY_LENS,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    *,
    settings: PMESettings,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create the PME reciprocal-space (long-range) potential.

    ``make_ewald_long_range_potential`` with the PME energy fn, so forces and NPT
    stress come from autodiff through the same composer. ``cache_lens`` is
    accepted for signature parity but must be ``None``: one moved particle still
    changes the whole FFT, so there is no incremental structure-factor cache.
    """
    assert cache_lens is None, "PME has no structure-factor cache to update."
    return make_ewald_long_range_potential(
        particles_view=particles_view,
        systems_view=systems_view,
        parameter_lens=parameter_lens,
        cache_lens=None,
        probe=probe,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
        energy_fn=make_pme_long_range_energy(settings),
    )
