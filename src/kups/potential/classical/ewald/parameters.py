# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Ewald parameter estimation and reciprocal lattice preparation before tracing."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jax import Array
from scipy.special import erfc

from kups.core.cell import AnyPeriodicity, Cell, Periodic3D
from kups.core.data import Table
from kups.core.typing import HasCell, HasCharges, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, no_jax_tracing
from kups.core.utils.math import triangular_3x3_matmul
from kups.potential.common.graph import IsRadiusGraphPoints

type ReciprocalGridBound = tuple[int, int, int]

# Default reciprocal workspace limits and backends that use phase grids.
_DEFAULT_RECIPROCAL_PARTICLE_CHUNK = 1024
_GPU_RECIPROCAL_PARTICLE_CHUNK = 8192
_DEFAULT_RECIPROCAL_K_CHUNK = 512
RECIPROCAL_GRID_BACKENDS: frozenset[str] = frozenset({"cpu"})


def _default_reciprocal_particle_chunk() -> int:
    # Larger tiles reduce the number of GPU scan iterations.
    if jax.default_backend() in {"gpu", "cuda", "rocm"}:
        return _GPU_RECIPROCAL_PARTICLE_CHUNK
    return _DEFAULT_RECIPROCAL_PARTICLE_CHUNK


class IsEwaldPointData(HasCharges, IsRadiusGraphPoints, Protocol):
    """Particle data required by Ewald: charges, positions, system/inclusion/exclusion indices."""


@dataclass
class EwaldParameters:
    """Per-system cutoffs and reciprocal shifts prepared before evaluation.

    ``reciprocal_shift_bound`` always stores three axis bounds. ``(0, 0, 0)``
    selects explicit, sphere-filtered shifts padded for batching; other bounds
    describe a complete grid. Rebuild parameters and the structure-factor cache
    when changing the cell or cutoff.
    """

    alpha: Table[SystemId, Array]  # (n_graphs,)
    cutoff: Table[SystemId, Array]  # (n_graphs,)
    reciprocal_lattice_shifts: Table[SystemId, Array]  # (n_graphs, n_kvecs, 3)
    k_max: Table[SystemId, Array]
    reciprocal_shift_bound: ReciprocalGridBound = field(static=True, default=(0, 0, 0))
    reciprocal_particle_chunk_size: int = field(
        static=True, default_factory=_default_reciprocal_particle_chunk
    )
    reciprocal_k_chunk_size: int = field(
        static=True, default=_DEFAULT_RECIPROCAL_K_CHUNK
    )

    @classmethod
    @no_jax_tracing
    def make(
        cls,
        charges: Table[ParticleId, IsEwaldPointData],
        cell: Table[SystemId, HasCell[Periodic3D]],
        epsilon_total: float = 1e-8,
        real_cutoff: float | None = None,
        *,
        reciprocal_particle_chunk_size: int | None = None,
        reciprocal_k_chunk_size: int = _DEFAULT_RECIPROCAL_K_CHUNK,
    ) -> EwaldParameters:
        """Estimate each system's parameters and prepare its reciprocal shifts.

        Args:
            charges: Particle table with charges and system assignments.
            cell: System table with periodic cells.
            epsilon_total: Target combined real-space and reciprocal-space error.
            real_cutoff: Optional fixed real-space cutoff in Å.
            reciprocal_particle_chunk_size: Maximum particles per response tile;
                defaults to 8192 on GPU and 1024 elsewhere.
            reciprocal_k_chunk_size: Maximum explicit k-vectors per tile; also sets the grid workspace budget.

        Returns:
            Ewald parameters with stored integer shifts and reciprocal cutoffs.
        """
        n_systems = len(cell.keys)
        sys_idx = charges.data.system.indices
        cell_list = [cell.data.cell[i] for i in range(n_systems)]
        estimates = [
            estimate_ewald_parameters(
                charges.data.charges[sys_idx == i],
                u,
                real_cutoff=real_cutoff,
                epsilon_total=epsilon_total,
            )
            for i, u in enumerate(cell_list)
        ]
        return cls.from_cutoffs(
            cell,
            alpha=cell.set_data(jnp.asarray([est.alpha for est in estimates])),
            cutoff=cell.set_data(jnp.asarray([est.real_cutoff for est in estimates])),
            k_max=cell.set_data(jnp.asarray([est.k_max for est in estimates])),
            reciprocal_particle_chunk_size=reciprocal_particle_chunk_size,
            reciprocal_k_chunk_size=reciprocal_k_chunk_size,
        )

    @classmethod
    @no_jax_tracing
    def from_cutoffs(
        cls,
        cell: Table[SystemId, HasCell[Periodic3D]],
        alpha: Table[SystemId, Array],
        cutoff: Table[SystemId, Array],
        k_max: Table[SystemId, Array],
        *,
        compact: bool | None = None,
        reciprocal_particle_chunk_size: int | None = None,
        reciprocal_k_chunk_size: int = _DEFAULT_RECIPROCAL_K_CHUNK,
    ) -> EwaldParameters:
        """Prepare reciprocal shifts once from supplied physical cutoffs.

        Args:
            cell: System table with periodic cells.
            alpha: Screening parameters in 1/Å.
            cutoff: Real-space cutoffs in Å.
            k_max: Reciprocal cutoffs in 1/Å.
            compact: Filter the grid; defaults to False on CPU and True elsewhere.
            reciprocal_particle_chunk_size: Maximum particles per response tile;
                defaults to 8192 on GPU and 1024 elsewhere.
            reciprocal_k_chunk_size: Maximum k-vectors per tile.

        Returns:
            Parameters with stored integer shifts and reciprocal cutoffs.
        """
        if compact is None:
            compact = jax.default_backend() not in RECIPROCAL_GRID_BACKENDS
        if reciprocal_particle_chunk_size is None:
            reciprocal_particle_chunk_size = _default_reciprocal_particle_chunk()
        shifts, bound = prepare_reciprocal_shifts(
            cell.data.cell, k_max[cell.index], compact=compact
        )
        return cls(
            alpha=alpha,
            cutoff=cutoff,
            reciprocal_lattice_shifts=cell.set_data(shifts),
            k_max=k_max,
            reciprocal_shift_bound=bound,
            reciprocal_particle_chunk_size=reciprocal_particle_chunk_size,
            reciprocal_k_chunk_size=reciprocal_k_chunk_size,
        )


@dataclass
class EwaldParameterEstimates:
    """Estimated optimal Ewald parameters for given accuracy."""

    alpha: float
    real_cutoff: float
    k_max: float
    error_real: float
    error_recip: float


def _minimize_on_grid(
    fn: Callable[[float], float], bounds: tuple[float, float], samples: int = 200
) -> float:
    candidates = np.linspace(*bounds, samples)
    return candidates[np.argmin([fn(x) for x in candidates])]


def _cutoff_for_error(
    error: Callable[[float], float], bounds: tuple[float, float], target: float
) -> float:
    cutoff = _minimize_on_grid(lambda x: abs(error(x) - target), bounds)
    return np.inf if error(cutoff) > 2 * target else cutoff


@no_jax_tracing
def estimate_ewald_parameters(
    charges: Array,
    cell: Cell[AnyPeriodicity],
    /,
    real_cutoff: float | None = None,
    alpha: float | None = None,
    epsilon_total: float = 1e-8,
) -> EwaldParameterEstimates:
    """Estimate cutoffs and screening for one system before JIT compilation.

    Split ``epsilon_total`` equally between real and reciprocal error targets.

    Args:
        charges: Charges in e for one system, shaped ``(n_particles,)``.
        cell: Single simulation cell.
        real_cutoff: Optional fixed real-space cutoff in Å; otherwise optimized.
        alpha: Optional screening parameter in 1/Å; ignored with a fixed real cutoff.
        epsilon_total: Target combined real-space and reciprocal-space error.

    Returns:
        Screening, cutoffs and estimated errors.
    """
    charges_np = np.asarray(charges)
    net_charge = float(charges_np.astype(np.float64).sum())
    if abs(net_charge) > 1e-6:
        warnings.warn(
            f"System is not charge neutral (net charge {net_charge:.4g} e); "
            "the neutralizing-background correction will be applied.",
            stacklevel=2,
        )
    volume = np.asarray(cell.volume)
    squared_charge = np.vdot(charges_np, charges_np)
    n_particles = charges.size
    max_radius = float(np.min(cell.perpendicular_lengths, axis=0)) / 2
    target = epsilon_total / 2
    lattice_length = volume ** (1 / 3)

    def real_error(cutoff: float, alpha: float):
        if cutoff <= 0 or alpha <= 0:
            return np.inf
        return (squared_charge / np.sqrt(n_particles)) * (erfc(alpha * cutoff) / cutoff)

    def reciprocal_error(cutoff: float, alpha: float):
        if cutoff <= 0 or alpha <= 0:
            return np.inf
        exponent = -(cutoff**2) / (4 * alpha**2)
        if exponent < -700:
            return 0.0
        return (squared_charge * alpha / (np.sqrt(n_particles) * np.pi)) * np.exp(
            exponent
        )

    def cutoffs(alpha: float) -> tuple[float, float]:
        return (
            _cutoff_for_error(
                lambda r: real_error(r, alpha), (0.01, min(20.0, max_radius)), target
            ),
            _cutoff_for_error(lambda k: reciprocal_error(k, alpha), (0.1, 5.0), target),
        )

    def total_cost(alpha: float) -> float:
        if alpha <= 0:
            return np.inf
        rc, kc = cutoffs(alpha)
        if rc == np.inf or kc == np.inf:
            return np.inf
        n_kvecs = (2 * kc * lattice_length / 2 / np.pi + 1) ** 3
        real_cost = rc**3 * n_particles / volume * 4 / 3 * np.pi
        return float(real_cost + n_kvecs)

    if real_cutoff is not None:
        # erfc(alpha * cutoff) = cutoff * target fixes alpha for this cutoff.
        result = scipy.optimize.minimize_scalar(
            lambda z: abs(erfc(z) - real_cutoff * target)
        )
        if not result.success:
            raise ValueError("Failed to find alpha for given real space cutoff.")
        alpha_opt = result.x / real_cutoff
        rc_opt = real_cutoff
        kc_opt = 2 * alpha_opt * np.sqrt(-np.log(target))
    else:
        alpha_opt = (
            _minimize_on_grid(total_cost, (0.001, 2), samples=400)
            if alpha is None
            else alpha
        )
        rc_opt, kc_opt = cutoffs(alpha_opt)
        if rc_opt <= 0 or kc_opt <= 0 or rc_opt == np.inf or kc_opt == np.inf:
            raise ValueError("Invalid cutoff values computed")

    return EwaldParameterEstimates(
        alpha=alpha_opt,
        real_cutoff=rc_opt,
        k_max=kc_opt,
        error_real=real_error(rc_opt, alpha_opt),
        error_recip=reciprocal_error(kc_opt, alpha_opt),
    )


@no_jax_tracing
def reciprocal_grid_bound(cell: Cell[AnyPeriodicity], kmax: float) -> Array:
    """Uniform coefficient bound enclosing the reciprocal kmax sphere.

    Args:
        cell: Cell defining the reciprocal lattice.
        kmax: Maximum reciprocal-vector magnitude in 1/Å.

    Returns:
        Integer scalar bound enclosing the reciprocal cutoff sphere.
    """
    rvecs = cell.inverse_vectors.mT * 2 * jnp.pi
    min_length = jnp.min(jnp.linalg.svd(rvecs)[1])
    return jnp.ceil(kmax / min_length).astype(int)


def kvecs_from_kmax(cell: Cell[AnyPeriodicity], kmax: float) -> Array:
    """Integer half-space lattice coefficients within the reciprocal kmax sphere.

    Args:
        cell: Cell defining the reciprocal lattice.
        kmax: Maximum reciprocal-vector magnitude in 1/Å.

    Returns:
        Integer half-space lattice coefficients shaped ``(n_kvecs, 3)``.
    """
    rvecs = cell.inverse_vectors.mT * 2 * jnp.pi
    vecs = reciprocal_grid_shifts(int(reciprocal_grid_bound(cell, kmax)))
    kvecs = triangular_3x3_matmul(rvecs, vecs, lower=False)
    return vecs[jnp.sum(kvecs**2, axis=-1) <= kmax**2]


def reciprocal_grid_shifts(bound: int | ReciprocalGridBound) -> Array:
    """Integer half-space grid: nonnegative x, signed y/z, x-major.

    Args:
        bound: Nonnegative maximum absolute lattice coefficients, or one bound
            for all three axes.

    Returns:
        Integer coefficients shaped ``((bx + 1) * (2*by + 1) * (2*bz + 1), 3)``.
    """
    bx, by, bz = (bound, bound, bound) if isinstance(bound, int) else bound
    axes = (
        jnp.arange(0, bx + 1),
        jnp.arange(-by, by + 1),
        jnp.arange(-bz, bz + 1),
    )
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


@no_jax_tracing
def prepare_reciprocal_shifts(
    cells: Cell[Periodic3D], k_max: Array, *, compact: bool
) -> tuple[Array, ReciprocalGridBound]:
    """Batch sphere-filtered shifts, padding them or enclosing them in one grid.

    Compact shifts use ``(0, 0, 0)`` as their bound. Grid shifts use the tightest
    common axis bounds for all systems, including batches with different cells.
    """
    cutoffs = np.asarray(k_max)
    if np.any(~np.isfinite(cutoffs)) or np.any(cutoffs < 0):
        raise ValueError("Reciprocal cutoffs must be finite and nonnegative")
    shifts = [kvecs_from_kmax(cells[i], k) for i, k in enumerate(cutoffs)]
    if compact:
        if not shifts:
            return jnp.zeros((0, 0, 3), dtype=int), (0, 0, 0)
        capacity = max(map(len, shifts))
        padded = jnp.stack(
            [jnp.pad(s, ((0, capacity - len(s)), (0, 0))) for s in shifts]
        )
        return padded, (0, 0, 0)

    maxima = [np.max(np.abs(s), axis=0) for s in shifts]
    bx, by, bz = np.max(maxima or [(0, 0, 0)], axis=0)
    bound = (int(bx), int(by), int(bz))
    grid = reciprocal_grid_shifts(bound)
    return jnp.broadcast_to(grid, (len(cutoffs), *grid.shape)), bound
