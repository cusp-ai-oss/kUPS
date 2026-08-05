# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Smooth Particle-Mesh Ewald (PME) reciprocal-space potential.

An ``O(N log N)`` FFT-based replacement for the direct-Ewald reciprocal term
``ewald_long_range_energy``, whose dense ``(N, N_k, 2)`` structure-factor tensor
runs out of memory at large ``N``. It is a drop-in at the long-range seam, so
forces and the NPT stress still come from autodiff.

Convention matches ``ewald.py`` so PME converges to the same number::

    P(k) = (2*pi/V) * exp(-k^2 / (4*alpha^2)) / k^2 ,   k = 2*pi * inverse_vectors @ n
    E_recip = sum_{k != 0} P(k) * B(m) * |FFT(Q)(m)|^2 * TO_STANDARD_UNITS

where ``Q`` is the order-``p`` cardinal B-spline charge mesh and ``B(m)`` is the
Euler exponential-spline aliasing correction (Essmann et al. 1995). ``alpha`` and
the real-space cutoff are reused verbatim from ``EwaldParameters`` so the
short-range / self / exclusion terms are unchanged and the splitting is
consistent. The FFT mesh dimensions are static (chosen once at construction);
under NPT the box fluctuates but the mesh stays fixed, as in standard PME.

"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from kups.core.cell import Periodic3D
from kups.core.data import Table, WithIndices
from kups.core.lens import Lens, View, lens
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
    ewald_net_charge_energy,
)
from kups.potential.common.energy import PotentialFromEnergy
from kups.potential.common.graph import PointCloud

DEFAULT_SPLINE_ORDER = 6
"""Default cardinal B-spline order."""


@dataclasses.dataclass(frozen=True)
class PMESettings:
    """Select FFT-based PME for the reciprocal term instead of direct Ewald.

    Passed to `make_ewald_potential` / `make_ewald_from_state` as ``pme=``; leaving
    those ``None`` keeps the direct-Ewald reciprocal sum.

    Attributes:
        mesh: Static FFT grid dimensions, held fixed across a trajectory (see
            `pme_mesh_for_cell`).
        order: Cardinal B-spline order.
    """

    mesh: tuple[int, int, int]
    order: int = DEFAULT_SPLINE_ORDER


def pme_mesh_for_cell(
    cell_vectors: Array, spacing: float = 1.0, multiple_of: int = 2
) -> tuple[int, int, int]:
    """Pick static PME mesh dimensions for a reference cell (~``spacing`` Angstrom).

    Resolution is measured across opposing faces, not along the lattice vectors, so
    the spacing holds for sheared cells too. Dims are rounded up to a multiple of
    ``multiple_of`` (cheaper FFTs). Use the *initial* cell; the mesh is held fixed
    across an NPT trajectory.

    Note the heuristic is purely geometric: the resolution PME actually needs also
    grows with the Ewald splitting parameter ``alpha``, so verify the accuracy of a
    chosen mesh against direct Ewald rather than trusting ``spacing`` alone.
    """
    v = np.asarray(cell_vectors)
    volume = abs(float(np.linalg.det(v)))
    lengths = [
        volume / float(np.linalg.norm(np.cross(v[b], v[c])))
        for b, c in ((1, 2), (0, 2), (0, 1))
    ]
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
    w = jnp.where((x >= 0) & (x < 1), 1.0, 0.0)  # M_1
    for k in range(2, order + 1):
        # M_k(x) = x/(k-1) M_{k-1}(x) + (k-x)/(k-1) M_{k-1}(x-1); shift gives M_{k-1}(x-1)
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


def _pme_reciprocal_energy_batched(
    positions: Array,  # (N, 3) Cartesian, Angstrom
    charges: Array,  # (N,)
    sys_idx: Array,  # (N,) per-particle system index
    n_sys: int,  # number of systems (static)
    inverse_vectors: Array,  # (n_sys, 3, 3)
    volume: Array,  # (n_sys,)
    alpha: Array,  # (n_sys,)
    mesh: tuple[int, int, int],
    order: int,
) -> Array:
    """Per-system smooth-PME reciprocal energy, in atomic units (the caller scales
    by ``TO_STANDARD_UNITS``)."""
    dtype = positions.dtype
    Mx, My, Mz = mesh
    Mvec = jnp.asarray(mesh, dtype=dtype)

    # Fractional coords in each particle's own cell. The contraction must match
    # Frame.to_fractional (inverse_vectors contracted on its *first* axis); the
    # transpose cancels against kvec below, so getting it wrong is invisible for
    # orthorhombic cells but shears the effective lattice.
    inv_p = inverse_vectors[sys_idx]  # (N, 3, 3)
    frac = jnp.einsum("nba,nb->na", inv_p, positions) % 1.0  # (N, 3) in [0, 1)
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

    # Per-system k-vectors, matching EwaldLongRangeInput.kvecs.
    nx = jnp.fft.fftfreq(Mx) * Mx
    ny = jnp.fft.fftfreq(My) * My
    nz = jnp.fft.fftfreq(Mz) * Mz
    nX, nY, nZ = jnp.meshgrid(nx, ny, nz, indexing="ij")
    n_idx = jnp.stack([nX, nY, nZ], axis=-1).astype(dtype)  # (Mx,My,Mz,3)
    kvec = 2.0 * jnp.pi * jnp.einsum("sab,xyzb->sxyza", inverse_vectors, n_idx)
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
    return jnp.sum(influence * (jnp.abs(Fq) ** 2), axis=(1, 2, 3))  # (n_sys,)


def make_pme_long_range_energy(mesh: tuple[int, int, int], order: int):
    """Build the PME long-range energy fn for a fixed static ``mesh`` and ``order``."""

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
            # Key-based lookup (as in `prefactor`), not raw .data: the parameter and
            # system tables need not be keyed in the same order.
            alpha=inp.parameters.alpha[pc.systems.index],
            mesh=mesh,
            order=order,
        )
        assert energy.shape == (n_sys,), (
            f"Expected energy shape {(n_sys,)} but got {energy.shape}."
        )
        energy = energy * TO_STANDARD_UNITS
        # The mesh sum omits k = 0 just as the direct-Ewald sum does, so it needs the
        # same neutralizing-background term to stay correct for a net-charged system.
        total = ewald_net_charge_energy(inp).map_data(lambda e_net: e_net + energy)
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

    Mirrors ``make_ewald_long_range_potential`` exactly (same composer, same
    point-cloud gradient lens) so forces and NPT stress come from autodiff.

    ``cache_lens`` is accepted for signature parity with
    ``make_ewald_long_range_potential`` but must be ``None``: PME needs no
    structure-factor cache, and ``PotentialFromEnergy`` emits an ``IdPatch``.
    """
    assert cache_lens is None, "PME has no structure-factor cache to update."
    return PotentialFromEnergy(
        energy_fn=make_pme_long_range_energy(settings.mesh, settings.order),
        composer=EwaldLongRangeComposer(
            particles=particles_view,
            systems=systems_view,
            probe=probe,
            parameters=parameter_lens,
            cache=None,
        ),
        gradient_lens=lens(lambda x: x.point_cloud).nest(gradient_lens),
        hessian_lens=hessian_lens,
        cache_lens=None,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
