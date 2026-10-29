# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Molecular dynamics observable utilities.

Pure utility functions for computing MD-specific observables from momenta,
forces, and other MD quantities. These are used internally by integrators
and are distinct from the StateProperty-based observables in kups.observables.
"""

from typing import Protocol, runtime_checkable

from kups.core import units
from kups.core.data import Index, Table
from kups.core.typing import HasMasses, HasMomenta, ParticleId, SystemId
from kups.observables.stress import (
    IsVirialParticles,
    IsVirialSystems,
    stress_via_virial_theorem,
)


@runtime_checkable
class _PressureTensorParticles(IsVirialParticles, HasMomenta, HasMasses, Protocol): ...


def particle_kinetic_energy(momentum: units.Momentum, mass: units.Mass) -> units.Energy:
    """Compute the per-particle kinetic energy from momentum and mass.

    Calculates the kinetic energy for each particle using:

    $$K_i = \\frac{\\mathbf{p}_i^2}{2m_i} = \\frac{p_{i,x}^2 + p_{i,y}^2 + p_{i,z}^2}{2m_i}$$

    where $\\mathbf{p}_i$ is the momentum vector and $m_i$ is the particle mass.

    Args:
        momentum: Momentum vector $\\mathbf{p}$, shape `(..., 3)`
        mass: Particle mass $m$, shape `(...,)`

    Returns:
        Per-particle kinetic energy $K$, shape `(...,)`
    """
    # K = p²/(2m)
    return 0.5 * units.sum(momentum**2, axis=-1) / mass


def remove_center_of_mass_momentum(
    momenta: units.Momentum, masses: units.Mass, system: Index[SystemId]
) -> units.Momentum:
    """Project momenta onto the zero-total-momentum subspace per system.

    The projection subtracts the center-of-mass velocity from each particle,
    ``p_i <- p_i - m_i * sum_j(p_j) / sum_j(m_j)``, independently for each
    system index. This is the mass-metric projection for the canonical ensemble
    conditioned on zero total momentum.

    Args:
        momenta: Particle momenta, shape ``(n_atoms, 3)``.
        masses: Particle masses, shape ``(n_atoms,)``.
        system: Per-atom system index.

    Returns:
        Momenta with each system's total momentum removed, shape ``(n_atoms, 3)``.
    """
    total_momentum = system.sum_over(momenta)
    total_mass = system.sum_over(masses).data
    com_velocity = total_momentum.set_data(total_momentum.data / total_mass[..., None])
    return momenta - masses[..., None] * com_velocity[system]


def system_kinetic_energy(
    momenta: units.Momentum,
    masses: units.Mass,
    system: Index[SystemId],
    *,
    remove_com: bool = False,
) -> units.Energy:
    """Total kinetic energy per system.

    Args:
        momenta: Particle momenta, shape ``(n_atoms, 3)``.
        masses: Particle masses, shape ``(n_atoms,)``.
        system: Per-atom system index.
        remove_com: If ``True``, subtract each system's center-of-mass motion
            first, yielding the internal (thermal) kinetic energy used for
            temperature.

    Returns:
        Kinetic energy per system, shape ``(n_systems,)``.
    """
    if remove_com:
        momenta = remove_center_of_mass_momentum(momenta, masses, system)
    return system.sum_over(particle_kinetic_energy(momenta, masses)).data


def instantaneous_pressure(
    kinetic_energy: units.Energy,
    cauchy_stress: units.Pressure,
    volume: units.Volume,
) -> units.Pressure:
    """Compute instantaneous pressure from kinetic energy and Cauchy stress.

    $$P = \\frac{2K}{dV} + \\frac{\\text{Tr}(\\boldsymbol{\\sigma})}{d}$$

    where $K$ is the total kinetic energy, $d$ is the spatial dimensionality,
    $V$ is the volume, and $\\boldsymbol{\\sigma}$ is the Cauchy stress tensor.

    Args:
        kinetic_energy: Total kinetic energy $K$, scalar or array.
        cauchy_stress: Cauchy stress tensor $\\boldsymbol{\\sigma}$,
            shape ``(d, d)``.
        volume: System volume $V$, scalar or array.

    Returns:
        Instantaneous pressure $P$, scalar or array.
    """
    d = cauchy_stress.shape[-1]
    trace = cauchy_stress.trace(axis1=-2, axis2=-1)
    return (2.0 * kinetic_energy) / (d * volume) + trace / d


def instantaneous_pressure_tensor(
    particles: Table[ParticleId, _PressureTensorParticles],
    systems: Table[SystemId, IsVirialSystems],
) -> units.Pressure:
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
        Symmetric pressure tensor per system, shape ``(n_systems, 3, 3)``.
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
