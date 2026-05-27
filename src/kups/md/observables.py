# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Molecular dynamics observable utilities.

Pure utility functions for computing MD-specific observables from momenta,
forces, and other MD quantities. These are used internally by integrators
and are distinct from the StateProperty-based observables in kups.observables.
"""

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.typing import HasMasses, HasMomenta, ParticleId, SystemId
from kups.core.utils.jax import vectorize
from kups.observables.stress import (
    IsVirialParticles,
    IsVirialSystems,
    stress_via_virial_theorem,
)


@runtime_checkable
class _PressureTensorParticles(IsVirialParticles, HasMomenta, HasMasses, Protocol): ...


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


def remove_center_of_mass_momentum(
    momenta: Array, masses: Array, system: Index[SystemId]
) -> Array:
    """Project momenta onto the zero-total-momentum subspace per system.

    The projection subtracts the center-of-mass velocity from each particle,
    ``p_i <- p_i - m_i * sum_j(p_j) / sum_j(m_j)``, independently for each
    system index. This is the mass-metric projection for the canonical ensemble
    conditioned on zero total momentum.
    """
    total_momentum = system.sum_over(momenta).data
    total_mass = system.sum_over(masses).data
    com_velocity = total_momentum / total_mass[..., None]
    return momenta - masses[..., None] * com_velocity[system.indices]


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


def instantaneous_pressure_tensor(
    particles: Table[ParticleId, _PressureTensorParticles],
    systems: Table[SystemId, IsVirialSystems],
) -> Array:
    r"""Compute the symmetric instantaneous pressure tensor.

    $$P_{\text{ins}} = \frac{1}{V}\sum_i \frac{\mathbf{p}_i \otimes \mathbf{p}_i}{m_i}
                      + \boldsymbol{\sigma},$$

    where the second term is the symmetric Cauchy stress from the virial
    theorem (`stress_via_virial_theorem`), which includes both the
    pair-force contribution ``sym(Σ Fᵢ ⊗ rᵢ)/V`` and the lattice-gradient
    contribution ``h^T·∂U/∂h / V`` needed for periodic potentials such as
    Ewald and PME. This is the tensorial generalisation of
    [`instantaneous_pressure`][kups.md.observables.instantaneous_pressure] used
    by extended-variable NPT integrators (e.g. Gao–Fang–Wang BAOAB NPT
    Langevin, Eq. 9).

    Args:
        particles: Per-particle table providing ``positions``, ``momenta``,
            ``masses``, ``system`` index and ``position_gradients``.
        systems: Per-system table providing ``cell`` and ``cell_gradients``.

    Returns:
        Symmetric pressure tensor per system, shape ``(n_systems, 3, 3)``,
        units of energy/length³.
    """
    p = particles.data.momenta
    m = particles.data.masses
    # Per-particle kinetic outer product, summed per-system via the kUPS-native
    # Index.sum_over (same pattern as stress_via_virial_theorem itself).
    per_particle = p[..., :, None] * p[..., None, :] / m[..., None, None]
    ke_tensor = particles.data.system.sum_over(per_particle).data
    volume = systems.data.cell.volume
    sigma = stress_via_virial_theorem(particles, systems).data
    return ke_tensor / volume[..., None, None] + sigma
