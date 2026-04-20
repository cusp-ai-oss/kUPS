# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Molecular dynamics observable utilities.

Pure utility functions for computing MD-specific observables from momenta,
forces, and other MD quantities. These are used internally by integrators
and are distinct from the StateProperty-based observables in kups.observables.
"""

import jax.numpy as jnp
from jax import Array

from kups.core.utils.jax import vectorize


def particle_kinetic_energy(momentum: Array, mass: Array) -> Array:
    """Compute the per-particle kinetic energy from momentum and mass.

    Calculates the kinetic energy for each particle using:

    $$K_i = \\frac{\\mathbf{p}_i^2}{2m_i} = \\frac{p_{i,x}^2 + p_{i,y}^2 + p_{i,z}^2}{2m_i}$$

    where $\\mathbf{p}_i$ is the momentum vector and $m_i$ is the particle mass.

    Args:
        momentum: Momentum vector $\\mathbf{p}$ (units: mass·length/time), shape `(..., 3)`
        mass: Particle mass $m$ (units: mass), shape `(...,)`

    Returns:
        Per-particle kinetic energy $K$ (units: energy), shape `(...,)`
    """
    # K = p²/(2m) [energy]
    return 0.5 * jnp.sum(jnp.square(momentum), axis=-1) / mass


@vectorize(signature="(),(3,3),()->()")
def instantaneous_pressure(
    kinetic_energy: Array,
    cauchy_stress: Array,
    volume: Array,
) -> Array:
    """Compute instantaneous pressure from kinetic energy and Cauchy stress.

    $$P = \\frac{2K}{dV} + \\frac{\\text{Tr}(\\boldsymbol{\\sigma})}{d}$$

    where $K$ is the total kinetic energy, $d$ is the spatial dimensionality,
    $V$ is the volume, and $\\boldsymbol{\\sigma}$ is the Cauchy stress tensor
    (units: energy/length³).

    Args:
        kinetic_energy: Total kinetic energy $K$ (units: energy), scalar or array.
        cauchy_stress: Cauchy stress tensor $\\boldsymbol{\\sigma}$
            (units: energy/length³), shape ``(d, d)``.
        volume: System volume $V$ (units: length³), scalar or array.

    Returns:
        Instantaneous pressure $P$ (units: energy/length³), scalar or array.
    """
    d = cauchy_stress.shape[0]
    return (2.0 * kinetic_energy) / (d * volume) + jnp.trace(cauchy_stress) / d
