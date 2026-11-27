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

from kups.core import units
from kups.core.cell import AnyPeriodicity, Cell, Periodic3D
from kups.core.data import Table
from kups.core.neighborlist import NeighborListPoints
from kups.core.typing import DTypeLike, HasCell, HasCharges, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, is_traced, no_jax_tracing

type ReciprocalGridBound = tuple[int, int, int]

# Dimension of the Ewald truncation-error estimates: an energy over ``k_e``.
type SquaredChargePerLength = units.Quantity[-1, 0, 2, 2, 0]

# The parameter search brackets are tuned on atomistic systems, so they scale
# with an atomistic length.
_SEARCH_LENGTH = 1.0 * units.catalog.angstrom  # noqa: KUP012

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


class IsEwaldPointData(HasCharges, NeighborListPoints, Protocol):
    """Particle data required by Ewald: charges, positions, system/inclusion/exclusion indices."""


class IsEwaldScreening(Protocol):
    """Per-system split of the Coulomb sum into real and reciprocal space."""

    @property
    def alpha(self) -> Table[SystemId, units.Wavenumber]: ...

    @property
    def cutoff(self) -> Table[SystemId, units.Length]: ...


@dataclass
class EwaldParameters:
    """Per-system cutoffs and reciprocal shifts prepared before evaluation.

    Rebuild parameters and the structure-factor cache when changing the cell
    or cutoff.

    Attributes:
        alpha: Screening parameter in 1/Å per system.
        cutoff: Real-space cutoff in Å per system.
        reciprocal_lattice_shifts: Sphere-filtered integer reciprocal shifts per
            system, zero-padded to a common count, shape ``(n_kvecs, 3)``.
        k_max: Reciprocal cutoff in 1/Å per system.
        reciprocal_shift_bound: Three axis bounds; ``(0, 0, 0)`` selects explicit
            sums over the shifts, other bounds select factored phase grids over
            that box, from which the shifts are gathered.
        reciprocal_particle_chunk_size: Maximum particles per response tile.
        reciprocal_k_chunk_size: Maximum explicit k-vectors per tile; also sets
            the grid workspace budget.
        reciprocal_dtype: Precision of structure-factor evaluations; ``float32``
            is much faster on GPUs with slow double precision.
    """

    alpha: Table[SystemId, units.Wavenumber]  # (n_graphs,)
    cutoff: Table[SystemId, units.Length]  # (n_graphs,)
    reciprocal_lattice_shifts: Table[SystemId, Array]  # (n_graphs, n_kvecs, 3)
    k_max: Table[SystemId, units.Wavenumber]
    reciprocal_shift_bound: ReciprocalGridBound = field(static=True, default=(0, 0, 0))
    reciprocal_particle_chunk_size: int = field(
        static=True, default_factory=_default_reciprocal_particle_chunk
    )
    reciprocal_k_chunk_size: int = field(
        static=True, default=_DEFAULT_RECIPROCAL_K_CHUNK
    )
    reciprocal_dtype: DTypeLike = field(static=True, default=jnp.float64)

    @classmethod
    @no_jax_tracing
    def make(
        cls,
        charges: Table[ParticleId, IsEwaldPointData],
        cell: Table[SystemId, HasCell[Periodic3D]],
        epsilon_total: SquaredChargePerLength,
        real_cutoff: units.Length | None = None,
        *,
        reciprocal_particle_chunk_size: int | None = None,
        reciprocal_k_chunk_size: int = _DEFAULT_RECIPROCAL_K_CHUNK,
        reciprocal_dtype: DTypeLike = jnp.float64,
    ) -> EwaldParameters:
        """Estimate each system's parameters and prepare its reciprocal shifts.

        Args:
            charges: Particle table with charges and system assignments.
            cell: System table with periodic cells.
            epsilon_total: Target combined real-space and reciprocal-space error,
                an energy over the Coulomb constant.
            real_cutoff: Optional fixed real-space cutoff.
            reciprocal_particle_chunk_size: Maximum particles per response tile;
                defaults to 8192 on GPU and 1024 elsewhere.
            reciprocal_k_chunk_size: Maximum explicit k-vectors per tile; also sets the grid workspace budget.
            reciprocal_dtype: Precision of structure-factor evaluations.

        Returns:
            Ewald parameters with stored integer shifts and reciprocal cutoffs.
        """
        n_systems = len(cell.keys)
        sys_idx = np.asarray(charges.data.system.indices_in(cell.keys))
        charge_values = charges.data.charges
        cell_list = [cell.data.cell[i] for i in range(n_systems)]
        estimates = [
            estimate_ewald_parameters(
                charge_values[sys_idx == i],
                system_cell,
                real_cutoff=real_cutoff,
                epsilon_total=epsilon_total,
            )
            for i, system_cell in enumerate(cell_list)
        ]
        return cls.from_cutoffs(
            cell,
            alpha=cell.set_data(units.stack([est.alpha for est in estimates])),
            cutoff=cell.set_data(units.stack([est.real_cutoff for est in estimates])),
            k_max=cell.set_data(units.stack([est.k_max for est in estimates])),
            reciprocal_particle_chunk_size=reciprocal_particle_chunk_size,
            reciprocal_k_chunk_size=reciprocal_k_chunk_size,
            reciprocal_dtype=reciprocal_dtype,
        )

    @classmethod
    @no_jax_tracing
    def from_cutoffs(
        cls,
        cell: Table[SystemId, HasCell[Periodic3D]],
        alpha: Table[SystemId, units.Wavenumber],
        cutoff: Table[SystemId, units.Length],
        k_max: Table[SystemId, units.Wavenumber],
        *,
        reciprocal_particle_chunk_size: int | None = None,
        reciprocal_k_chunk_size: int = _DEFAULT_RECIPROCAL_K_CHUNK,
        reciprocal_dtype: DTypeLike = jnp.float64,
    ) -> EwaldParameters:
        """Prepare reciprocal shifts once from supplied physical cutoffs.

        Args:
            cell: System table with periodic cells.
            alpha: Screening parameters.
            cutoff: Real-space cutoffs.
            k_max: Reciprocal cutoffs.
            reciprocal_particle_chunk_size: Maximum particles per response tile;
                defaults to 8192 on GPU and 1024 elsewhere.
            reciprocal_k_chunk_size: Maximum k-vectors per tile.
            reciprocal_dtype: Precision of structure-factor evaluations.

        Returns:
            Parameters with stored integer shifts and reciprocal cutoffs.
        """
        if reciprocal_particle_chunk_size is None:
            reciprocal_particle_chunk_size = _default_reciprocal_particle_chunk()
        shifts, bound = prepare_reciprocal_shifts(
            cell.data.cell,
            k_max[cell.index],
            grid=jax.default_backend() in RECIPROCAL_GRID_BACKENDS,
        )
        return cls(
            alpha=alpha,
            cutoff=cutoff,
            reciprocal_lattice_shifts=cell.set_data(shifts),
            k_max=k_max,
            reciprocal_shift_bound=bound,
            reciprocal_particle_chunk_size=reciprocal_particle_chunk_size,
            reciprocal_k_chunk_size=reciprocal_k_chunk_size,
            reciprocal_dtype=reciprocal_dtype,
        )


@dataclass
class EwaldParameterEstimates:
    """Estimated optimal Ewald parameters for given accuracy.

    Attributes:
        alpha: Screening parameter in 1/Å.
        real_cutoff: Real-space cutoff in Å.
        k_max: Reciprocal cutoff in 1/Å.
        error_real: Estimated real-space error.
        error_recip: Estimated reciprocal-space error.
    """

    alpha: units.Wavenumber
    real_cutoff: units.Length
    k_max: units.Wavenumber
    error_real: SquaredChargePerLength
    error_recip: SquaredChargePerLength


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
    charges: units.Charge,
    cell: Cell[AnyPeriodicity],
    /,
    real_cutoff: units.Length | None = None,
    alpha: units.Wavenumber | None = None,
    *,
    epsilon_total: SquaredChargePerLength,
) -> EwaldParameterEstimates:
    """Estimate cutoffs and screening for one system before JIT compilation.

    Split ``epsilon_total`` equally between real and reciprocal error targets.
    The search runs on magnitudes in the charges' base.

    Args:
        charges: Charges of one system, shaped ``(n_particles,)``.
        cell: Single simulation cell.
        real_cutoff: Optional fixed real-space cutoff; otherwise optimized.
        alpha: Optional screening parameter; ignored with a fixed real cutoff.
        epsilon_total: Target combined real-space and reciprocal-space error,
            an energy over the Coulomb constant.

    Returns:
        Screening, cutoffs and estimated errors.
    """
    bs = units.base_of(charges)

    def host(q: units.Quantity[int, int, int, int, int]) -> np.ndarray:
        return np.asarray(q.rebase(bs).value)

    fixed_cutoff = None if real_cutoff is None else float(host(real_cutoff))
    fixed_alpha = None if alpha is None else float(host(alpha))
    charges_np = host(charges)
    net_charge = float(charges_np.astype(np.float64).sum()) / float(
        host(units.constants.e)
    )
    if abs(net_charge) > 1e-6:
        warnings.warn(
            f"System is not charge neutral (net charge {net_charge:.4g} e); "
            "the neutralizing-background correction will be applied.",
            stacklevel=2,
        )
    volume = host(cell.volume)
    squared_charge = np.vdot(charges_np, charges_np)
    n_particles = charges_np.size
    max_radius = float(np.min(host(cell.perpendicular_lengths), axis=0)) / 2
    target = float(host(epsilon_total)) / 2
    scale = float(host(_SEARCH_LENGTH))
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
                lambda r: real_error(r, alpha),
                (0.01 * scale, min(20.0 * scale, max_radius)),
                target,
            ),
            _cutoff_for_error(
                lambda k: reciprocal_error(k, alpha), (0.1 / scale, 5.0 / scale), target
            ),
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

    if fixed_cutoff is not None:
        # erfc(alpha * cutoff) = cutoff * target fixes alpha for this cutoff.
        def erfc_residual(z: float) -> float:
            return abs(erfc(z) - fixed_cutoff * target)

        result = scipy.optimize.minimize_scalar(erfc_residual)
        if not result.success:
            raise ValueError("Failed to find alpha for given real space cutoff.")
        alpha_opt = result.x / fixed_cutoff
        rc_opt = fixed_cutoff
        kc_opt = 2 * alpha_opt * np.sqrt(-np.log(target))
    else:
        alpha_opt = (
            _minimize_on_grid(total_cost, (0.001 / scale, 2 / scale), samples=400)
            if fixed_alpha is None
            else fixed_alpha
        )
        rc_opt, kc_opt = cutoffs(alpha_opt)
        if rc_opt <= 0 or kc_opt <= 0 or rc_opt == np.inf or kc_opt == np.inf:
            raise ValueError("Invalid cutoff values computed")

    error_dim = units.dim_of(epsilon_total)
    return EwaldParameterEstimates(
        alpha=units.from_dim(alpha_opt, units.WAVENUMBER, bs),
        real_cutoff=units.from_dim(rc_opt, units.LENGTH, bs),
        k_max=units.from_dim(kc_opt, units.WAVENUMBER, bs),
        error_real=units.from_dim(real_error(rc_opt, alpha_opt), error_dim, bs),
        error_recip=units.from_dim(reciprocal_error(kc_opt, alpha_opt), error_dim, bs),
    )


def _reciprocal_vectors(cell: Cell[AnyPeriodicity], bs: units.BaseSystem) -> np.ndarray:
    """Reciprocal lattice vectors in ``bs`` (rows, with the 2π factor) on the host."""
    return np.asarray(cell.inverse_vectors.rebase(bs).value).mT * 2 * np.pi


@no_jax_tracing
def reciprocal_grid_bound(cell: Cell[AnyPeriodicity], kmax: units.Wavenumber) -> int:
    """Uniform coefficient bound enclosing the reciprocal kmax sphere.

    Args:
        cell: Cell defining the reciprocal lattice.
        kmax: Maximum reciprocal-vector magnitude.

    Returns:
        Integer bound enclosing the reciprocal cutoff sphere.
    """
    bs = units.base_of(kmax)
    min_length = np.linalg.svd(_reciprocal_vectors(cell, bs), compute_uv=False).min()
    return int(np.ceil(float(kmax.value) / min_length))


@no_jax_tracing
def kvecs_from_kmax(cell: Cell[AnyPeriodicity], kmax: units.Wavenumber) -> np.ndarray:
    """Integer half-space lattice coefficients within the reciprocal kmax sphere.

    Args:
        cell: Cell defining the reciprocal lattice.
        kmax: Maximum reciprocal-vector magnitude.

    Returns:
        Integer half-space lattice coefficients shaped ``(n_kvecs, 3)``.
    """
    k = float(kmax.value)
    vecs = reciprocal_grid_shifts(reciprocal_grid_bound(cell, kmax))
    kvecs = vecs @ _reciprocal_vectors(cell, units.base_of(kmax))
    return vecs[np.sum(kvecs**2, axis=-1) <= k**2]


def reciprocal_grid_index(
    shifts: Array, bound: ReciprocalGridBound
) -> Array | np.ndarray:
    """Flat positions of shifts in ``reciprocal_grid_shifts(bound)``.

    Args:
        shifts: Integer shifts inside ``bound``, shaped ``(..., n_kvecs, 3)``.
        bound: Grid axis bounds.

    Returns:
        Grid indices shaped ``(..., n_kvecs)``; a NumPy array for concrete
        shifts, so kernels can specialize on them.
    """
    _, by, bz = bound
    xp = jnp if is_traced(shifts) else np
    ints = xp.asarray(shifts).astype(np.int32)
    return (ints[..., 0] * (2 * by + 1) + ints[..., 1] + by) * (2 * bz + 1) + (
        ints[..., 2] + bz
    )


def reciprocal_grid_shifts(bound: int | ReciprocalGridBound) -> np.ndarray:
    """Integer half-space grid: nonnegative x, signed y/z, x-major.

    Args:
        bound: Nonnegative maximum absolute lattice coefficients, or one bound
            for all three axes.

    Returns:
        Integer coefficients shaped ``((bx + 1) * (2*by + 1) * (2*bz + 1), 3)``.
    """
    bx, by, bz = (bound, bound, bound) if isinstance(bound, int) else bound
    axes = (
        np.arange(0, bx + 1),
        np.arange(-by, by + 1),
        np.arange(-bz, bz + 1),
    )
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


@no_jax_tracing
def prepare_reciprocal_shifts(
    cells: Cell[Periodic3D], k_max: units.Wavenumber, *, grid: bool
) -> tuple[Array, ReciprocalGridBound]:
    """Batch sphere-filtered shifts, padded to a common count.

    With ``grid``, the bound is the tightest common enclosing grid of all
    systems, including batches with different cells; otherwise ``(0, 0, 0)``.

    Args:
        cells: Batched cells, one per system.
        k_max: Reciprocal cutoffs, shaped ``(n_systems,)``.
        grid: Whether to return the enclosing phase-grid bound.

    Returns:
        Padded integer shifts ``(n_systems, n_kvecs, 3)`` and the grid bound.
    """
    cutoffs = np.asarray(units.magnitudes(k_max))
    if np.any(~np.isfinite(cutoffs)) or np.any(cutoffs < 0):
        raise ValueError("Reciprocal cutoffs must be finite and nonnegative")
    shifts = [kvecs_from_kmax(cells[i], k_max[i]) for i in range(len(cutoffs))]
    if not shifts:
        return jnp.zeros((0, 0, 3), dtype=int), (0, 0, 0)
    capacity = max(map(len, shifts))
    padded = jnp.asarray(
        np.stack([np.pad(s, ((0, capacity - len(s)), (0, 0))) for s in shifts])
    )
    if not grid:
        return padded, (0, 0, 0)
    bx, by, bz = np.max([np.max(np.abs(s), axis=0) for s in shifts], axis=0)
    return padded, (int(bx), int(by), int(bz))


# a_m of the P3M force error estimate per spline order, Deserno and Holm, J. Chem.
# Phys. 109, 7694 (1998), for ik differentiation and the optimal influence function.
_P3M_ERROR_COEFFICIENTS = {
    4: (1 / 4320, 3 / 1936, 7601 / 2271360, 143 / 28800),
    6: (
        691 / 68140800,
        13 / 57600,
        47021 / 35512320,
        9694607 / 2095994880,
        733191589 / 59609088000,
        326190917 / 11700633600,
    ),
}
# PME's RMS force error per order: the P3M estimate with (h alpha)^(p-1) instead of
# (h alpha)^p, times a pair factor, and each charge's self force times a self factor,
# added in quadrature; the factors bound direct-Ewald errors of random systems.
_PME_ERROR_FACTORS = {4: (1.8, 1.3), 6: (1.6, 1.0)}


@dataclass
class PMEParameters:
    """Per-system screening and one particle mesh shared by all systems.

    The mesh and the spline order are static, as FFT shapes must be. Rebuild
    the parameters when a cell changes enough to need another mesh.

    Attributes:
        alpha: Screening parameter per system.
        cutoff: Real-space cutoff per system, read by the real-space term.
        mesh: Mesh points along each cell vector.
        order: Even cardinal B-spline order, at most the smallest mesh size.
        reciprocal_dtype: Precision of the charge mesh and its FFT.
    """

    alpha: Table[SystemId, units.Wavenumber]
    cutoff: Table[SystemId, units.Length]
    mesh: tuple[int, int, int] = field(static=True)
    order: int = field(static=True, default=6)
    reciprocal_dtype: DTypeLike = field(static=True, default=jnp.float64)

    def __post_init__(self) -> None:
        if self.order < 2 or self.order % 2 or min(self.mesh) < self.order:
            raise ValueError(
                "PME needs an even spline order of at most the mesh size; got order "
                f"{self.order} and mesh {self.mesh}."
            )

    @classmethod
    @no_jax_tracing
    def make(
        cls,
        charges: Table[ParticleId, IsEwaldPointData],
        cell: Table[SystemId, HasCell[Periodic3D]],
        epsilon_total: SquaredChargePerLength,
        real_cutoff: units.Length | None = None,
        *,
        reciprocal_dtype: DTypeLike = jnp.float64,
    ) -> PMEParameters:
        """Estimate screening, cutoffs, mesh and spline order from a target error.

        Screening and cutoffs follow ``estimate_ewald_parameters``. The mesh is
        about as accurate as the direct sum at the estimated ``k_max``; the
        order is the cheaper of 4 and 6, and the mesh is the per-axis maximum
        over systems.

        Args:
            charges: Particle table with charges and system assignments.
            cell: System table with periodic cells.
            epsilon_total: Target combined real-space and reciprocal-space error,
                an energy over the Coulomb constant.
            real_cutoff: Optional fixed real-space cutoff.
            reciprocal_dtype: Precision of the charge mesh and its FFT.

        Returns:
            Parameters whose mesh meets the target in every system.
        """
        n_systems = len(cell.keys)
        sys_idx = np.asarray(charges.data.system.indices_in(cell.keys))
        charge_values = charges.data.charges
        cell_list = [cell.data.cell[i] for i in range(n_systems)]
        estimates = [
            estimate_ewald_parameters(
                charge_values[sys_idx == i],
                system_cell,
                real_cutoff=real_cutoff,
                epsilon_total=epsilon_total,
            )
            for i, system_cell in enumerate(cell_list)
        ]

        def mesh(order: int) -> tuple[int, int, int]:
            sizes = [
                estimate_pme_mesh(
                    charge_values[sys_idx == i],
                    system_cell,
                    est.alpha,
                    est.k_max,
                    order,
                )
                for i, (system_cell, est) in enumerate(
                    zip(cell_list, estimates, strict=True)
                )
            ]
            mx, my, mz = np.max(sizes, axis=0).tolist()
            return mx, my, mz

        def cost(order: int) -> float:
            points = np.prod(mesh(order))
            n_particles = np.sum(sys_idx < n_systems)
            return n_particles * order**3 + n_systems * points * np.log2(points)

        order = min(_PME_ERROR_FACTORS, key=cost)
        return cls(
            alpha=cell.set_data(units.stack([est.alpha for est in estimates])),
            cutoff=cell.set_data(units.stack([est.real_cutoff for est in estimates])),
            mesh=mesh(order),
            order=order,
            reciprocal_dtype=reciprocal_dtype,
        )


@no_jax_tracing
def estimate_pme_mesh(
    charges: units.Charge,
    cell: Cell[AnyPeriodicity],
    /,
    alpha: units.Wavenumber,
    k_max: units.Wavenumber,
    order: int,
) -> tuple[int, int, int]:
    """Smallest 2-3-5 mesh about as accurate as the direct sum cut at ``k_max``.

    Both are RMS force errors: the direct sum's from Kolafa and Perram, Mol.
    Simul. 9, 351 (1992), the mesh's from the P3M estimate of Deserno and Holm
    (1998), fitted to PME. The search runs on magnitudes in the charges' base.

    Args:
        charges: Charges of one system, shaped ``(n_particles,)``.
        cell: Its cell, single or as a batch of one.
        alpha: Screening parameter.
        k_max: Reciprocal cutoff of the direct sum.
        order: Spline order, 4 or 6.

    Returns:
        Mesh points along each cell vector, each at least ``order``.
    """
    bs = units.base_of(charges)

    def host(q: units.Quantity[int, int, int, int, int]) -> np.ndarray:
        return np.asarray(q.rebase(bs).value)

    q = host(charges).astype(np.float64)
    if not np.any(q):
        return order, order, order
    widths = host(cell.perpendicular_lengths).reshape(3)
    volume = float(host(cell.volume).squeeze())
    screening = float(host(alpha))
    cutoff = float(host(k_max))
    length = volume ** (1 / 3)
    pair = q @ q / np.sqrt(q.size)
    decay = np.exp(-(cutoff**2) / (4 * screening**2))
    target = pair * screening * np.sqrt(8 / (volume * cutoff)) * decay
    pair_factor, self_factor = _PME_ERROR_FACTORS[order]
    scale = np.hypot(
        pair_factor
        * pair
        * np.sqrt(screening * length * np.sqrt(2 * np.pi))
        / length**2,
        self_factor * np.sqrt(np.mean(q**4)) * screening**2,
    )

    def error(spacing: float) -> float:
        x = spacing * screening
        series = sum(
            a * x ** (2 * m) for m, a in enumerate(_P3M_ERROR_COEFFICIENTS[order])
        )
        return scale * x ** (order - 1) * np.sqrt(series)

    def excess(spacing: float) -> float:
        return np.log(error(spacing) / target)

    # The error grows with the spacing; the coarsest mesh has ``order`` points.
    spacing = widths.min() / order
    if error(spacing) > target:
        spacing = scipy.optimize.brentq(excess, 1e-9 * spacing, spacing)
    mx, my, mz = (_next_2_3_5(max(order, int(np.ceil(w / spacing)))) for w in widths)
    return mx, my, mz


def _next_2_3_5(n: int) -> int:
    """Smallest integer of at least ``n`` without prime factors above 5."""
    while True:
        rest = n
        for prime in (2, 3, 5):
            while rest % prime == 0:
                rest //= prime
        if rest == 1:
            return n
        n += 1
