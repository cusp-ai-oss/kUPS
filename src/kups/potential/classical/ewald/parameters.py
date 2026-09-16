# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Ewald summation for long-range electrostatics in periodic systems.

Splits the Coulomb potential into short-range (real-space), long-range
(reciprocal-space), and self-interaction terms. Supports incremental
updates via cached structure factors for efficient Monte Carlo.
"""

from __future__ import annotations

import warnings
from typing import (
    Callable,
    Protocol,
)

import einops
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.special
from jax import Array
from scipy.special import erfc

from kups.core.cell import AnyPeriodicity, Cell, Periodic3D
from kups.core.data import Table
from kups.core.typing import (
    HasCell,
    HasCharges,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import (
    dataclass,
    no_jax_tracing,
)
from kups.potential.common.graph import (
    IsRadiusGraphPoints,
)


@dataclass
class EwaldParameters:
    """Ewald summation parameters: convergence settings and reciprocal lattice vectors.

    Attributes:
        alpha: Ewald screening parameter [1/Ang], shape `(n_graphs,)`.
        cutoff: Real-space cutoff radius [Ang], shape `(n_graphs,)`.
        reciprocal_lattice_shifts: Integer k-vector coefficients,
            shape `(n_graphs, n_kvecs, 3)`.
    """

    alpha: Table[SystemId, Array]  # (n_graphs,)
    cutoff: Table[SystemId, Array]  # (n_graphs,)
    reciprocal_lattice_shifts: Table[
        SystemId, Array
    ]  # (n_graphs, n_rec_shifts, 3) integers

    @classmethod
    @no_jax_tracing
    def make(
        cls,
        charges: Table[ParticleId, IsEwaldPointData],
        cell: Table[SystemId, HasCell[Periodic3D]],
        epsilon_total: float = 1e-8,
        real_cutoff: float | None = None,
    ) -> EwaldParameters:
        """Estimate Ewald parameters from indexed particles and systems.

        Splits particles by system index, estimates per-system parameters,
        and zero-pads k-vectors to the maximum count across systems.

        Args:
            charges: Indexed particles with charges and system assignment.
            cell: Indexed systems with cells.
            epsilon_total: Target total accuracy for the Ewald sum.
            real_cutoff: Optional real-space cutoff override; estimated if not given.

        Returns:
            ``EwaldParameters`` with estimated physics parameters.
        """
        # Unpack Indexed into per-system lists
        n_systems = len(cell.keys)
        sys_idx = charges.data.system.indices
        charges_list = [charges.data.charges[sys_idx == i] for i in range(n_systems)]
        cell_list = [cell.data.cell[i] for i in range(n_systems)]

        estimates_list = [
            estimate_ewald_parameters(
                c, u, real_cutoff=real_cutoff, epsilon_total=epsilon_total
            )
            for c, u in zip(charges_list, cell_list)
        ]
        shifts_list = [
            kvecs_from_kmax(u, est.k_max) for u, est in zip(cell_list, estimates_list)
        ]
        max_n_kvecs = max(len(s) for s in shifts_list)
        padded_shifts = jnp.stack(
            [jnp.pad(s, [(0, max_n_kvecs - len(s)), (0, 0)]) for s in shifts_list]
        )
        return cls(
            alpha=Table(
                cell.keys,
                jnp.asarray([est.alpha for est in estimates_list]),
            ),
            cutoff=Table(
                cell.keys,
                jnp.asarray([est.real_cutoff for est in estimates_list]),
            ),
            reciprocal_lattice_shifts=Table(cell.keys, padded_shifts),
        )


class IsEwaldPointData(HasCharges, IsRadiusGraphPoints, Protocol):
    """Particle data required by Ewald: charges, positions, system/inclusion/exclusion indices."""

    ...


@dataclass
class EwaldParameterEstimates:
    """Estimated optimal Ewald parameters for given accuracy."""

    alpha: float
    real_cutoff: float
    k_max: float
    error_real: float
    error_recip: float
    kvecs: Array


@no_jax_tracing
def estimate_ewald_parameters(
    charges: Array,
    cell: Cell[AnyPeriodicity],
    /,
    real_cutoff: float | None = None,
    alpha: float | None = None,
    epsilon_total: float = 1e-8,
) -> EwaldParameterEstimates:
    """Estimate optimal Ewald parameters for target accuracy.

    Not JAX-compatible (uses scipy); call before JIT compilation.
    Only works on single systems, not batched.

    Args:
        charges: Particle charges [e], shape `(n_particles,)`.
        cell: Cell parameters.
        real_cutoff: Real-space cutoff [Ang]; optimized if ``None``.
        alpha: Screening parameter [1/Ang]; optimized if ``None``.
        epsilon_total: Target total error (split equally between real/reciprocal).

    Returns:
        Optimized Ewald summation parameters.
    """
    # Note: only runs on a single system, not a batch of systems.
    # Input validation
    charges_np = np.asarray(charges)
    net_charge = float(charges_np.astype(np.float64).sum())
    if abs(net_charge) > 1e-6:
        warnings.warn(
            f"System is not charge neutral (net charge {net_charge:.4g} e); "
            "the neutralizing-background correction will be applied.",
            stacklevel=2,
        )
    volume = np.asarray(cell.volume)
    Q2 = np.vdot(charges_np, charges_np)
    N = charges.size

    # smallest side length spanned by the cell
    max_radius = np.min(cell.perpendicular_lengths, axis=0) / 2

    # Split error budget equally
    eps_target = epsilon_total / 2

    # Length of a box shaped like the cell
    lattice_length = volume ** (1 / 3)

    def minimize(
        f: Callable[[float], float], bounds: tuple[float, float], n: int = 200
    ) -> float:
        attempts = np.linspace(bounds[0], bounds[1], n)  # type: ignore
        return attempts[np.argmin([f(x) for x in attempts])]

    def real_space_error(rc: float, alpha: float):
        """Standard Ewald real space error estimate"""
        if rc <= 0 or alpha <= 0:
            return np.inf
        return (Q2 / np.sqrt(N)) * (erfc(alpha * rc) / rc)

    def recip_space_error(kc: float, alpha: float):
        """Standard Ewald reciprocal space error estimate"""
        if kc <= 0 or alpha <= 0:
            return np.inf
        exp_arg = -(kc**2) / (4 * alpha**2)
        # Prevent numerical overflow/underflow
        if exp_arg < -700:  # exp(-700) ≈ 0
            return 0.0
        return (Q2 * alpha / (np.sqrt(N) * np.pi)) * np.exp(exp_arg)

    def optimal_rc(alpha: float) -> float:
        # Solve for rc s.t. real_space_error(rc, alpha) ≈ eps_target
        def real_error_diff(rc: float) -> float:
            return abs(real_space_error(rc, alpha) - eps_target)

        # Use reasonable bounds for rc based on system size
        if real_cutoff is None:
            rc_max = min(20.0, max_radius)  # Increased max rc
            rc_bounds = (0.01, rc_max)
            rc_opt = minimize(real_error_diff, rc_bounds)
        else:
            rc_opt = real_cutoff
        if real_space_error(rc_opt, alpha) > eps_target * 2:
            return np.inf
        return rc_opt

    def optimal_kc(alpha: float) -> float:
        """Find kc that gives the target reciprocal space error"""

        def recip_error_diff(kc: float) -> float:
            # Also minimize kc if all things being equal
            return abs(recip_space_error(kc, alpha) - eps_target)  # + kc * eps_target

        # Reasonable bounds for kc
        kc_bounds = (0.1, 5.0)  # Allow larger kc values
        kc_opt = minimize(recip_error_diff, kc_bounds)
        if recip_space_error(kc_opt, alpha) > eps_target * 2:
            return np.inf
        return kc_opt

    def total_cost(alpha: float) -> float:
        if alpha <= 0:
            return np.inf

        rc_opt = optimal_rc(alpha)
        if rc_opt == np.inf:
            return np.inf

        kc_opt = optimal_kc(alpha)
        if kc_opt == np.inf:
            return np.inf

        # Cost function: computational effort scales with rc^3 for real space
        # and with number of k-vectors for reciprocal space
        # Simple cost model: real space scales as rc^3, reciprocal as kc^3
        n_kvecs = (2 * kc_opt * lattice_length / 2 / np.pi + 1) ** 3
        rc_cost = rc_opt**3 * N / volume * 4 / 3 * np.pi
        cost_per_particle = rc_cost + n_kvecs
        return float(cost_per_particle)

    # Fast path without optimization
    if real_cutoff is not None:
        # First we solve for alpha given the real space cutoff by solving
        # erfc(alpha * rc) = rc * eps_target
        target_erfc = real_cutoff * eps_target
        alpha_result: scipy.optimize.OptimizeResult = scipy.optimize.minimize_scalar(  # type: ignore
            lambda z: abs(scipy.special.erfc(z) - target_erfc)
        )
        if not alpha_result.success:
            raise ValueError("Failed to find alpha for given real space cutoff.")
        alpha_opt: float = alpha_result.x / real_cutoff
        # For this alpha, we solve for kmax via the reciprocal space error estimate
        # exp(-k^2/4alpha^2) <= eps_target -> k >= 2 alpha sqrt(-ln(eps_target))
        kmax = 2 * alpha_opt * np.sqrt(-np.log(eps_target))
        return EwaldParameterEstimates(
            alpha=alpha_opt,
            real_cutoff=real_cutoff,
            k_max=kmax,
            error_real=real_space_error(real_cutoff, alpha_opt),
            error_recip=recip_space_error(kmax, alpha_opt),
            kvecs=kvecs_from_kmax(cell, kmax),
        )

    if alpha is None:
        alpha_opt = minimize(total_cost, (0.001, 2), n=400)
    else:
        alpha_opt = alpha
    rc_opt = optimal_rc(alpha_opt)
    kc_opt = optimal_kc(alpha_opt)

    # Verify the cutoffs are reasonable
    if rc_opt <= 0 or kc_opt <= 0 or rc_opt == np.inf or kc_opt == np.inf:
        raise ValueError("Invalid cutoff values computed")

    return EwaldParameterEstimates(
        alpha=alpha_opt,
        real_cutoff=rc_opt,
        k_max=kc_opt,
        error_real=real_space_error(rc_opt, alpha_opt),
        error_recip=recip_space_error(kc_opt, alpha_opt),
        kvecs=kvecs_from_kmax(cell, kc_opt),
    )


@no_jax_tracing
def kvecs_from_kmax(cell: Cell[AnyPeriodicity], kmax: float) -> Array:
    """Generate integer k-vector coefficients within a sphere of radius ``kmax``.

    Args:
        cell: Cell defining the reciprocal lattice.
        kmax: Maximum k-vector magnitude cutoff.

    Returns:
        Integer k-vector coefficients, shape ``(n_kvecs, 3)``.
    """
    rvecs = cell.inverse_vectors.mT * 2 * jnp.pi
    min_length = jnp.min(jnp.linalg.svd(rvecs)[1])
    n = jnp.ceil(kmax / min_length).astype(int)
    lattice = (jnp.arange(0, n + 1), jnp.arange(-n, n + 1), jnp.arange(-n, n + 1))
    vecs = jnp.stack(jnp.meshgrid(*lattice), axis=-1).reshape(-1, 3)
    kvecs = einops.einsum(vecs, rvecs, "kvecs dim1, dim1 dim2 -> kvecs dim2")
    return vecs[jnp.linalg.norm(kvecs, axis=-1) <= kmax]
