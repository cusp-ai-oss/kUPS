# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Fugacity calculations for real gas mixtures.

This module provides fugacity coefficient calculations for non-ideal gases and mixtures.
Fugacity is a measure of chemical potential that accounts for deviations from ideal gas
behavior, essential for accurate phase equilibria and grand canonical Monte Carlo simulations.

Currently implements the Peng-Robinson cubic equation of state, which provides good
accuracy for hydrocarbons and many other fluids. The framework supports extension to
other equations of state (e.g., Soave-Redlich-Kwong, virial equations).

Fugacities come back in the log domain, as a [units.Log][kups.core.units.Log],
because a reservoir at a few bar and one at a few hundred are hundreds of decades
apart in activity. The logs carry the unit they are measured against (pascal), so
``log_fugacity.exp()`` recovers the fugacity in whatever unit a caller names.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from kups.core import units
from kups.core.units import catalog as u
from kups.core.utils.jax import jit, vectorize
from kups.core.utils.math import cubic_roots


class LogFugacityResult(NamedTuple):
    """Result of fugacity calculation.

    Attributes:
        log_fugacity: Fugacity of each component, in the log domain against
            pascal.
        compressibility: Compressibility factor Z = PV/(nRT) [-]
    """

    log_fugacity: units.Log[units.PressureUnit]
    compressibility: Array


@jit
@vectorize(signature="(n),(n),(n),(n),(n,n)->(n),(n)")
def _peng_robinson_log_coefficient(
    reduced_pressure: Array,
    reduced_temperature: Array,
    acentric_factor: Array,
    fraction: Array,
    interaction: Array,
) -> tuple[Array, Array]:
    """Log fugacity coefficients and compressibility factors from Peng-Robinson.

    The equation of state is a correlation in the reduced variables ``P/Pc`` and
    ``T/Tc``, so everything here is a dimensionless ratio and the pressure the
    fugacity is proportional to enters at the caller.

    The computation involves:
    1. Computing the temperature-dependent parameter m from acentric factors
    2. Determining attractive (A) and repulsive (B) parameters for each component
    3. Mixing rules for multi-component systems with binary interaction parameters
    4. Solving the cubic equation for compressibility factor Z
    5. Computing fugacity coefficients from the equation of state derivatives
    6. Selecting the appropriate root (gas phase) based on thermodynamic stability

    Args:
        reduced_pressure: Pressure over each component's critical pressure [-],
            shape (n,)
        reduced_temperature: Temperature over each component's critical
            temperature [-], shape (n,)
        acentric_factor: Acentric factor for each component [-], shape (n,)
        fraction: Mole fraction of each component [-], shape (n,)
        interaction: Binary interaction parameters [-], shape (n,n)

    Returns:
        tuple containing:
            - log_coefficient: Natural logarithm of the fugacity coefficient of
              each component [-], shape (n,)
            - compressibility: Compressibility factor Z [-], shape (n,)

    Notes:
        - Fugacity is related to chemical potential: μᵢ = μᵢ⁰ + RT ln(fᵢ)
        - For component i in a mixture: fᵢ = φᵢ * yᵢ * P, where φᵢ is the fugacity coefficient
        - The gas phase root is selected by choosing the solution with minimum fugacity coefficient
        - Binary interaction parameters kᵢⱼ modify the attractive parameter: aᵢⱼ = (1-kᵢⱼ)√(aᵢaⱼ)
    """
    n = reduced_pressure.size
    m = 0.37464 + 1.54226 * acentric_factor - 0.26992 * acentric_factor**2
    sqrt_alpha = 1 + m * (1 - (reduced_temperature**0.5))

    A = 0.45724 * reduced_pressure / reduced_temperature**2 * sqrt_alpha**2
    B = 0.0778 * reduced_pressure / reduced_temperature

    Aij = (1 - interaction) * jnp.sqrt(jnp.outer(A, A))
    Bmix = jnp.vdot(fraction, B)
    Amix = jnp.vdot(Aij, jnp.outer(fraction, fraction))

    coefficients = jnp.array(
        [
            1,
            Bmix - 1,
            Amix - 3.0 * Bmix**2 - 2 * Bmix,
            -(Amix * Bmix - Bmix**2 - Bmix**3),
        ]
    )

    solutions = cubic_roots(coefficients)
    Z_real = solutions.real
    # A root is "physical" iff it's (nearly) real and compressibility-feasible
    # (Z > Bmix so the log(Z − Bmix) below is finite). Tolerances derive from
    # the working dtype: eps^(1/2) bounds the imaginary-part noise of real
    # roots, eps^(3/4) is the strictly-positive margin of log's argument.
    eps = jnp.finfo(Z_real.dtype).eps
    is_real = jnp.abs(solutions.imag) < eps**0.5
    valid_solutions = is_real & (Z_real > Bmix + eps**0.75)

    # Double-where: substitute a safe placeholder Z for unphysical branches so
    # `jnp.log` and friends stay finite. The placeholder value is never
    # selected by the argmin below because the mask pins it to +inf, but
    # using it avoids NaN gradients flowing back through unused branches.
    safe_Z = jnp.where(valid_solutions, Z_real, Bmix + 1.0)

    SQ2 = jnp.sqrt(2)
    dA_dyi = 2 * jnp.einsum("ab,b->a", Aij, fraction)[:, None]
    log_coefficient = (
        (B[:, None] / Bmix) * (safe_Z - 1)
        - jnp.log(safe_Z - Bmix)
        - (Amix / (2 * SQ2 * Bmix))
        * (dA_dyi / Amix - B[:, None] / Bmix)
        * jnp.log((safe_Z + (1 + SQ2) * Bmix) / (safe_Z + (1 - SQ2) * Bmix))
    )
    # Select gas phase fugacity: lowest log_coefficient among valid roots.
    # https://github.com/iRASPA/RASPA2/blob/6498ab1eec9c8e0f063dcd0d71dd7add372c529b/src/equations_of_state.c#L364
    solution_idx = jnp.argmin(
        jnp.where(valid_solutions, log_coefficient, jnp.inf), axis=1
    )
    return log_coefficient[jnp.arange(n), solution_idx], safe_Z[solution_idx]


def peng_robinson_log_fugacity(
    pressure: units.Pressure,
    temperature: units.Temperature,
    critical_pressure: units.Pressure,
    critical_temperature: units.Temperature,
    acentric_factor: Array,
    composition: Array = jnp.ones((1,)),
    interaction: Array = jnp.zeros((1, 1)),
) -> LogFugacityResult:
    """Compute log fugacity coefficients using Peng-Robinson equation of state.

    Public API for Peng-Robinson fugacity calculations with automatic vectorization
    over batched inputs. This function wraps the internal implementation with
    convenient defaults for single-component systems.

    The Peng-Robinson equation of state is:

    $$
    P = \\frac{RT}{V-b} - \\frac{a\\alpha(T)}{V^2 + 2bV - b^2}
    $$

    where $a$ and $b$ are component-specific parameters, and $\\alpha(T)$ is a temperature-
    dependent correction factor based on the acentric factor.

    Args:
        pressure: System pressure
        temperature: System temperature
        critical_pressure: Critical pressure for each component, shape `(n,)`
        critical_temperature: Critical temperature for each component, shape `(n,)`
        acentric_factor: Acentric factor for each component [-], shape `(n,)`
        composition: Mole fraction of each component [-], shape `(n,)`. Default: pure component
        interaction: Binary interaction parameters [-], shape `(n, n)`. Default: ideal mixing

    Returns:
        LogFugacityResult containing log fugacity and compressibility factor

    Example:
        ```python
        import jax.numpy as jnp
        from kups.core import units
        from kups.core.units import catalog as u
        from kups.mcmc.fugacity import peng_robinson_log_fugacity

        # CO2 properties
        result = peng_robinson_log_fugacity(
            pressure=10.0 * u.bar,
            temperature=300.0 * u.K,
            critical_pressure=jnp.array([7.38e6]) * u.Pa,
            critical_temperature=jnp.array([304.2]) * u.K,
            acentric_factor=jnp.array([0.228]),
        )
        fugacity = result.log_fugacity.exp().to(u.bar)  # in bar
        Z = result.compressibility
        ```
    """
    log_coefficient, compressibility = _peng_robinson_log_coefficient(
        pressure[..., None] / critical_pressure,
        temperature[..., None] / critical_temperature,
        acentric_factor,
        composition,
        interaction,
    )
    # f_i = phi_i * y_i * P, whose factors add in the log domain; the pressure is
    # the one that carries a dimension, so its log sets the reference.
    log_pressure = units.Log.of(pressure, u.Pa).log[..., None]
    return LogFugacityResult(
        units.Log.from_logs(
            log_pressure + (log_coefficient + jnp.log(composition)), u.Pa
        ),
        compressibility,
    )
