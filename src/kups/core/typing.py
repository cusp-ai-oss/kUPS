# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Structural typing protocols and sentinel ID types for simulation entities.

Defines :class:`ParticleId`, :class:`SystemId`, and other sentinel ``int``/``str``
sub-types used as type-safe index labels, plus ``Has*`` protocols for structural
duck-typing of simulation data (positions, momenta, forces, etc.).
"""

from typing import Any, Protocol, Union, runtime_checkable

import numpy as np
from jax import Array

from kups.core.data import Index, Table
from kups.core.unitcell import UnitCell

DType = np.dtype
type PyTree = Any


class SystemId(int):
    """Sentinel type for simulation system indices."""


class ParticleId(int):
    """Sentinel type for particle indices."""


class MotifParticleId(int):
    """Sentinel type for motif particle indices."""


class MotifId(int):
    """Sentinel type for motif template indices."""


class InclusionId(int):
    """Sentinel type for inclusion group indices."""


class ExclusionId(int):
    """Sentinel type for exclusion group indices."""


class GroupId(int):
    """Sentinel type for molecular group indices."""


class Label(str):
    """Sentinel type for species/molecule string labels."""


@runtime_checkable
class SupportsDType(Protocol):
    @property
    def dtype(self, /) -> DType: ...


DTypeLike = Union[
    str,  # like 'float32', 'int32'
    type[Any],  # like np.float32, np.int32, float, int
    np.dtype,  # like np.dtype('float32'), np.dtype('int32')
    SupportsDType,  # like jnp.float32, jnp.int32
]


@runtime_checkable
class HasGroupIndex(Protocol):
    """Protocol for entities with molecular group index.

    Attributes:
        group: Index mapping particles to molecular groups.
    """

    @property
    def group(self) -> Index[GroupId]: ...


@runtime_checkable
class HasSystemIndex(Protocol):
    """Protocol for entities with simulation system index.

    Attributes:
        system: Index mapping entities to independent simulation boxes.
    """

    @property
    def system(self) -> Index[SystemId]: ...


@runtime_checkable
class HasExclusionIndex(Protocol):
    """Protocol for entities with exclusion group index.

    Attributes:
        exclusion: Index mapping particles to exclusion groups.
    """

    @property
    def exclusion(self) -> Index[ExclusionId]: ...


@runtime_checkable
class HasInclusionIndex(Protocol):
    """Protocol for entities with inclusion group index.

    Attributes:
        inclusion: Index mapping particles to inclusion groups.
    """

    @property
    def inclusion(self) -> Index[InclusionId]: ...


@runtime_checkable
class HasMotifIndex(Protocol):
    """Protocol for entities with molecular motif index.

    Attributes:
        motif: Index mapping particles to motif templates.
    """

    @property
    def motif(self) -> Index[MotifId]: ...


@runtime_checkable
class HasPositions(Protocol):
    """Protocol for entities with position data.

    Attributes:
        positions: Array of positions for each entity (e.g., atoms).
    """

    @property
    def positions(self) -> Array: ...


@runtime_checkable
class HasMomenta(Protocol):
    """Protocol for entities with momentum data.

    Attributes:
        momenta: Array of momenta for each entity (e.g., atoms).
    """

    @property
    def momenta(self) -> Array: ...


@runtime_checkable
class HasVelocities(Protocol):
    """Protocol for entities with velocity data.

    Attributes:
        velocities: Array of velocities for each entity (e.g., atoms).
    """

    @property
    def velocities(self) -> Array: ...


@runtime_checkable
class HasForces(Protocol):
    """Protocol for entities with force data.

    Attributes:
        forces: Array of forces for each entity (e.g., atoms).
    """

    @property
    def forces(self) -> Array: ...


@runtime_checkable
class HasWeights(Protocol):
    """Protocol for entities with weight/mass data.

    Attributes:
        weights: Array of weights/masses for each entity (e.g., atoms).
    """

    @property
    def weights(self) -> Array: ...


@runtime_checkable
class HasLabels(Protocol):
    """Protocol for entities with string labels.

    Attributes:
        labels: Species identifier for each group (e.g., molecule type).
    """

    @property
    def labels(self) -> Index[Label]: ...


@runtime_checkable
class HasCharges(Protocol):
    """Protocol for entities with charge data.

    Attributes:
        charges: Array of charges for each entity (e.g., atoms).
    """

    @property
    def charges(self) -> Array: ...


@runtime_checkable
class HasMasses(Protocol):
    """Protocol for entities with mass data.

    Attributes:
        masses: Array of masses for each entity (e.g., atoms).
    """

    @property
    def masses(self) -> Array: ...


@runtime_checkable
class HasAtomicNumbers(Protocol):
    """Protocol for entities with atomic numbers.

    Attributes:
        atomic_numbers: Array of atomic numbers for each atom.
    """

    @property
    def atomic_numbers(self) -> Array: ...


@runtime_checkable
class HasTemperature(Protocol):
    """Protocol for entities with temperature parameters.

    Attributes:
        temperature: Temperature array [energy units], shape `(n_systems,)`.
    """

    @property
    def temperature(self) -> Array: ...


@runtime_checkable
class HasLogActivity(Protocol):
    """Protocol for entities with chemical activity parameters.

    Attributes:
        log_activity: Natural log of chemical activity per species [dimensionless],
            shape `(n_systems, n_species)`.
    """

    @property
    def log_activity(self) -> Array: ...


@runtime_checkable
class HasUnitCell(Protocol):
    """Protocol for entities with unit cell parameters.

    Attributes:
        unitcell: Unit cell parameters for each system (lattice vectors, volume).
    """

    @property
    def unitcell(self) -> UnitCell: ...


@runtime_checkable
class HasTimeStep(Protocol):
    r"""Protocol for systems with integration time step.

    Attributes:
        time_step: Time step $\Delta t$ (units: time).
    """

    @property
    def time_step(self) -> Array: ...


@runtime_checkable
class HasThermalEnergy(Protocol):
    r"""Protocol for systems with target thermal energy.

    Attributes:
        thermal_energy: Thermal energy $k_B T$ (units: energy).
    """

    @property
    def thermal_energy(self) -> Array: ...


@runtime_checkable
class HasFrictionCoefficient(Protocol):
    r"""Protocol for systems with Langevin friction coefficient.

    Attributes:
        friction_coefficient: Friction coefficient $\gamma$ (units: 1/time).
    """

    @property
    def friction_coefficient(self) -> Array: ...


@runtime_checkable
class HasThermostatTimeConstant(Protocol):
    r"""Protocol for systems with thermostat coupling time constant.

    Attributes:
        thermostat_time_constant: Thermostat time constant $\tau$ (units: time).
    """

    @property
    def thermostat_time_constant(self) -> Array: ...


@runtime_checkable
class HasStressTensor(Protocol):
    r"""Protocol for systems with computed virial stress tensor.

    Attributes:
        stress_tensor: Virial stress tensor $\mathbf{W}$ (units: energy).
    """

    @property
    def stress_tensor(self) -> Array: ...


@runtime_checkable
class HasTargetPressure(Protocol):
    r"""Protocol for systems with target pressure for barostat.

    Attributes:
        target_pressure: Target pressure $P_0$ (units: energy/length³).
    """

    @property
    def target_pressure(self) -> Array: ...


@runtime_checkable
class HasPressureCouplingTime(Protocol):
    r"""Protocol for systems with barostat coupling time constant.

    Attributes:
        pressure_coupling_time: Barostat time constant $\tau_P$ (units: time).
    """

    @property
    def pressure_coupling_time(self) -> Array: ...


@runtime_checkable
class HasCompressibility(Protocol):
    r"""Protocol for systems with isothermal compressibility.

    Attributes:
        compressibility: Isothermal compressibility $\beta$ (units: length³/energy).
    """

    @property
    def compressibility(self) -> Array: ...


@runtime_checkable
class HasMinimumScaleFactor(Protocol):
    r"""Protocol for systems with minimum barostat scale factor.

    Attributes:
        minimum_scale_factor: Minimum scaling factor $\mu_{\text{min}}$ (dimensionless).
    """

    @property
    def minimum_scale_factor(self) -> Array: ...


@runtime_checkable
class HasPotentialEnergy(Protocol):
    r"""Protocol for systems with a potential energy.

    Attributes:
        potential_energy: Potential energy in eV.
    """

    @property
    def potential_energy(self) -> Array: ...


@runtime_checkable
class HasCache[Data, Cache](Protocol):
    """Protocol for objects carrying primary data alongside a cache."""

    @property
    def data(self) -> Data: ...
    @property
    def cache(self) -> Cache: ...


type MaybeCached[P, C] = P | HasCache[P, C]


@runtime_checkable
class HasMotifAndSystemIndex(HasMotifIndex, HasSystemIndex, Protocol):
    """Protocol for entities with both motif and system indices."""


@runtime_checkable
class HasPositionsAndLabels(HasPositions, HasLabels, Protocol):
    """Protocol for entities with both position and label data."""

    ...


@runtime_checkable
class HasPositionsAndAtomicNumbers(HasPositions, HasAtomicNumbers, Protocol):
    """Protocol for entities with both position and atomic number data."""

    ...


@runtime_checkable
class HasPositionsAndSystemIndex(HasPositions, HasSystemIndex, Protocol):
    """Protocol for entities with both position data and system index."""


@runtime_checkable
class HasPositionsAndGroupIndex(HasPositions, HasGroupIndex, Protocol):
    """Protocol for entities with both position data and group index."""


@runtime_checkable
class HasCutoff(Protocol):
    """Protocol for entities with a distance cutoff.

    Attributes:
        cutoff: Cutoff distance (units: length).
    """

    @property
    def cutoff(self) -> Array: ...


@runtime_checkable
class HasParticles[Particles](Protocol):
    @property
    def particles(self) -> Table[ParticleId, Particles]: ...


@runtime_checkable
class HasSystems[Systems](Protocol):
    @property
    def systems(self) -> Table[SystemId, Systems]: ...


@runtime_checkable
class IsState[Particles, Systems](
    HasParticles[Particles], HasSystems[Systems], Protocol
): ...
