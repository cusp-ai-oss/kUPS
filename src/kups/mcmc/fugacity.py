# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Fugacity calculations for real gas mixtures.

This module provides fugacity coefficient calculations for non-ideal gases and mixtures.
Fugacity is a measure of chemical potential that accounts for deviations from ideal gas
behavior, essential for accurate phase equilibria and grand canonical Monte Carlo simulations.

Currently implements the Peng-Robinson cubic equation of state, which provides good
accuracy for hydrocarbons and many other fluids. The framework supports extension to
other equations of state (e.g., Soave-Redlich-Kwong, virial equations).
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from kups.core.utils.jax import jit, vectorize
from kups.core.utils.math import cubic_roots


class LogFugacityResult(NamedTuple):
    """Result of fugacity calculation.

    Attributes:
        log_fugacity: Natural logarithm of fugacity for each component [ln(Pa)]
        compressibility: Compressibility factor Z = PV/(nRT) [-]
    """

    log_fugacity: Array
    compressibility: Array


@jit
@vectorize(signature="(),(),(n),(n),(n),(n),(n,n)->(n),(n)")
def _peng_robinson_log_fugacity(
    pressure: Array,
    temperature: Array,
    critical_pressure: Array,
    critical_temperature: Array,
    acentric_factor: Array,
    fraction: Array,
    interaction: Array,
) -> tuple[Array, Array]:
    """Compute log fugacity coefficients and compressibility factors using the Peng-Robinson equation of state.

    This function implements the Peng-Robinson cubic equation of state for gas mixtures to calculate
    the natural logarithm of fugacity coefficients and compressibility factors. The Peng-Robinson EOS
    is particularly suitable for hydrocarbon systems and provides good accuracy for both liquid and
    vapor phases.

    The computation involves:
    1. Calculating reduced pressure (Pr) and temperature (Tr) using critical properties
    2. Computing the temperature-dependent parameter m from acentric factors
    3. Determining attractive (A) and repulsive (B) parameters for each component
    4. Mixing rules for multi-component systems with binary interaction parameters
    5. Solving the cubic equation for compressibility factor Z
    6. Computing fugacity coefficients from the equation of state derivatives
    7. Selecting the appropriate root (gas phase) based on thermodynamic stability

    Args:
        pressure: System pressure [Pa]
        temperature: System temperature [K]
        critical_pressure: Critical pressure for each component [Pa], shape (n,)
        critical_temperature: Critical temperature for each component [K], shape (n,)
        acentric_factor: Acentric factor for each component [-], shape (n,)
        fraction: Mole fraction of each component [-], shape (n,)
        interaction: Binary interaction parameters [-], shape (n,n)

    Returns:
        tuple containing:
            - log_fugacity: Natural logarithm of fugacity for each component [ln(Pa)], shape (n,)
            - compressibility: Compressibility factor Z [-], scalar

    Notes:
        - Fugacity is related to chemical potential: μᵢ = μᵢ⁰ + RT ln(fᵢ)
        - For component i in a mixture: fᵢ = φᵢ * yᵢ * P, where φᵢ is the fugacity coefficient
        - The gas phase root is selected by choosing the solution with minimum fugacity coefficient
        - Binary interaction parameters kᵢⱼ modify the attractive parameter: aᵢⱼ = (1-kᵢⱼ)√(aᵢaⱼ)
    """
    n = critical_pressure.size
    Pr = pressure / critical_pressure
    Tr = temperature / critical_temperature
    m = 0.37464 + 1.54226 * acentric_factor - 0.26992 * acentric_factor**2
    sqrt_alpha = 1 + m * (1 - (Tr**0.5))

    A = 0.45724 * Pr / Tr**2 * sqrt_alpha**2
    B = 0.0778 * Pr / Tr

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
    # (Z > Bmix so the log(Z − Bmix) below is finite).
    is_real = jnp.abs(solutions.imag) < 1e-8
    valid_solutions = is_real & (Z_real > Bmix + 1e-12)

    # Double-where: substitute a safe placeholder Z for unphysical branches so
    # `jnp.log` and friends stay finite. The placeholder value is never
    # selected by the argmin below because the mask pins it to +inf, but
    # using it avoids NaN gradients flowing back through unused branches.
    safe_Z = jnp.where(valid_solutions, Z_real, Bmix + 1.0)

    SQ2 = jnp.sqrt(2)
    dA_dyi = 2 * jnp.einsum("ab,b->a", Aij, fraction)[:, None]
    ln_fugacity_coeff = (
        (B[:, None] / Bmix) * (safe_Z - 1)
        - jnp.log(safe_Z - Bmix)
        - (Amix / (2 * SQ2 * Bmix))
        * (dA_dyi / Amix - B[:, None] / Bmix)
        * jnp.log((safe_Z + (1 + SQ2) * Bmix) / (safe_Z + (1 - SQ2) * Bmix))
    )
    # For component i in a mixture: f_i = phi_i * y_i * p
    ln_fugacity = ln_fugacity_coeff + jnp.log(pressure) + jnp.log(fraction)[:, None]
    # Select gas phase fugacity: lowest ln_fugacity_coeff among valid roots.
    # https://github.com/iRASPA/RASPA2/blob/6498ab1eec9c8e0f063dcd0d71dd7add372c529b/src/equations_of_state.c#L364
    solution_idx = jnp.argmin(
        jnp.where(valid_solutions, ln_fugacity_coeff, jnp.inf), axis=1
    )
    ln_fugacity = ln_fugacity[jnp.arange(n), solution_idx]
    Z_selected = safe_Z[solution_idx]
    return ln_fugacity, Z_selected


def peng_robinson_log_fugacity(
    pressure: Array,
    temperature: Array,
    critical_pressure: Array,
    critical_temperature: Array,
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
        pressure: System pressure [Pa]
        temperature: System temperature [K]
        critical_pressure: Critical pressure for each component [Pa], shape `(n,)`
        critical_temperature: Critical temperature for each component [K], shape `(n,)`
        acentric_factor: Acentric factor for each component [-], shape `(n,)`
        composition: Mole fraction of each component [-], shape `(n,)`. Default: pure component
        interaction: Binary interaction parameters [-], shape `(n, n)`. Default: ideal mixing

    Returns:
        LogFugacityResult containing log fugacity and compressibility factor

    Example:
        ```python
        import jax.numpy as jnp
        from kups.mcmc.fugacity import peng_robinson_log_fugacity

        # CO2 properties
        P = 1e6  # 10 bar in Pa
        T = 300.0  # K
        Pc = jnp.array([7.38e6])  # Critical pressure
        Tc = jnp.array([304.2])   # Critical temperature
        omega = jnp.array([0.228])  # Acentric factor

        result = peng_robinson_log_fugacity(P, T, Pc, Tc, omega)
        fugacity = jnp.exp(result.log_fugacity)  # Convert from log
        Z = result.compressibility
        ```
    """
    return LogFugacityResult(
        *_peng_robinson_log_fugacity(
            pressure,
            temperature,
            critical_pressure,
            critical_temperature,
            acentric_factor,
            composition,
            interaction,
        )
    )
