# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Protocol

from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Table
from kups.core.data.index import SupportsSorting
from kups.core.lens import Lens, View, bind
from kups.core.propagator import Propagator, SequentialPropagator
from kups.core.typing import (
    HasCompressibility,
    HasDegreesOfFreedom,
    HasForces,
    HasFrictionCoefficient,
    HasMasses,
    HasMinimumScaleFactor,
    HasMomenta,
    HasPositions,
    HasPressureCouplingTime,
    HasSystemIndex,
    HasTargetPressure,
    HasTemperature,
    HasThermostatTimeConstant,
    HasTimeStep,
    HasUnitCell,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import UnitCell
from kups.core.utils.functools import pipe
from kups.core.utils.jax import dataclass, field, tree_map, vectorize
from kups.core.utils.random import sample_like
from kups.md.observables import instantaneous_pressure, particle_kinetic_energy
from kups.observables.stress import stress_via_virial_theorem

type Time = Array
type Mass = Array
type Energy = Array
type Temperature = Array
type Pressure = Array
type Stress = Array

type Integrator = Literal["verlet", "baoab_langevin", "csvr", "csvr_npt"]


@runtime_checkable
class Flow[State, PyTree](Protocol):
    """Protocol for position update flows with boundary conditions.

    A flow defines how positions evolve under velocity updates, potentially
    including boundary conditions like periodic wrapping or reflections.
    """

    def __call__(
        self, state: State, dt: Time, primal: PyTree, tangent: PyTree
    ) -> PyTree:
        """Apply flow to update positions.

        Args:
            state: Current simulation state.
            dt: Timestep $\\Delta t$ (units: time).
            primal: Position $\\mathbf{r}$ (units: length).
            tangent: Velocity $\\mathbf{v}$  (units: length/time).

        Returns:
            Updated position (units: length).
        """
        ...


@vectorize(signature=("(),(n),(n)->(n)"), excluded=frozenset({0}))
def euclidean_flow(
    state: Any,
    dt: Time,
    primal: Array,
    tangent: Array,
) -> Array:
    """Flow in unbounded Euclidean space without boundary conditions.

    Implements simple kinematic update:

    $$\\mathbf{r}_{\\text{new}} = \\mathbf{r} + \\mathbf{v} \\cdot \\Delta t$$

    Use this for non-periodic systems or when positions are handled differently.

    Args:
        state: Current simulation state (unused but required by Flow protocol)
        dt: Timestep $\\Delta t$ (units: time)
        primal: Position $\\mathbf{r}$ (units: length)
        tangent: Velocity $\\mathbf{v}$ (units: length/time)

    Returns:
        Updated position $\\mathbf{r}_{\\text{new}}$ (units: length)
    """
    return primal + tangent * dt


@dataclass
class MinimumImageConventionFlow[State, PyTree](Flow[State, PyTree]):
    """Flow with periodic boundary conditions using minimum image convention.

    Wraps the base flow to apply periodic boundary conditions, ensuring particles
    remain within the primary simulation cell. After updating positions via the
    underlying flow, applies the unit cell's `wrap` method to fold positions
    back into the box.

    Type Parameters:
        State: Simulation state type
        PyTree: JAX PyTree type for positions

    Attributes:
        unitcell: View to extract the [UnitCell][kups.core.unitcell.UnitCell] from state
        flow: Underlying flow operator (typically [euclidean_flow][kups.md.integrators.euclidean_flow])

    Example:
        ```python
        from kups.md.integrators import MinimumImageConventionFlow, euclidean_flow

        # Create PBC flow
        pbc_flow = MinimumImageConventionFlow(
            unitcell=lambda s: s.unitcell,
            flow=euclidean_flow
        )
        ```
    """

    unitcell: View[State, UnitCell] = field(static=True)
    flow: Flow[State, PyTree] = field(static=True)

    def __call__(
        self, state: State, dt: Time, primal: PyTree, tangent: PyTree
    ) -> PyTree:
        return tree_map(
            self.unitcell(state).wrap, self.flow(state, dt, primal, tangent)
        )


def half_time[S: HasTimeStep](sys: Table[SystemId, S]) -> Table[SystemId, S]:
    """View that halves the time_step of a system.

    Used by every integrator factory (atomic and rigid) to compose the half-step
    momentum kicks of velocity-Verlet / BAOAB. Public because the rigid-body
    factories in :mod:`kups.md.rigid` need to share it.

    Args:
        sys: Indexed system with time_step attribute

    Returns:
        New Indexed system with time_step halved
    """
    return bind(sys, lambda x: x.data.time_step).apply(lambda x: x / 2)


_half_time = half_time
"""Backwards-compatible alias for :func:`half_time`."""


@runtime_checkable
class _PositionStepData(
    HasMomenta, HasPositions, HasMasses, HasSystemIndex, Protocol
): ...


@dataclass
class PositionStep[State, Key: SupportsSorting](Propagator[State]):
    """Update positions using velocities in molecular dynamics.

    Implements the 'A' operator in splitting schemes, propagating positions
    forward in time using the current velocities. This is the kinematic update
    step in velocity Verlet and related integrators.

    The position update follows:

    $$\\mathbf{r}(t+\\Delta t) = \\mathbf{r}(t) + \\mathbf{v}(t) \\cdot \\Delta t$$

    where $\\mathbf{v} = \\mathbf{p}/m$ is the velocity derived from momentum.

    Type Parameters:
        State: Simulation state type
        Key: Table key type (e.g. :class:`ParticleId` for atomic MD,
            :class:`GroupId` for rigid-body COM drift).

    Attributes:
        entries: Lens to get/set the indexed table whose entries are drifted
            (atom positions in atomic MD, group COM positions in rigid-body MD).
            Each entry must expose ``positions``, ``momenta``, ``masses``, and
            ``system``.
        systems: View to extract system data with time step $\\Delta t$
        flow: Flow operator defining how positions evolve (handles boundary conditions)
    """

    entries: Lens[State, Table[Key, _PositionStepData]] = field(static=True)
    systems: View[State, Table[SystemId, HasTimeStep]] = field(static=True)
    flow: Flow[State, Array] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Apply position update step.

        Args:
            key: JAX PRNG key (unused in this deterministic step).
            state: Current simulation state.

        Returns:
            Updated state with new positions.
        """
        del key  # Deterministic step
        entries_lens = self.entries.bind(state)
        entries = entries_lens.get()
        sys = self.systems(state)[entries.data.system]
        # r_new = r + (p/m) · Δt
        velocity = entries.data.momenta / entries.data.masses[..., None]
        new_positions = self.flow(
            state, sys.time_step, entries.data.positions, velocity
        )
        assert new_positions.shape == entries.data.positions.shape
        return entries_lens.focus(lambda x: x.data.positions).set(new_positions)


@runtime_checkable
class IsMomentumStepData(HasMomenta, HasForces, HasSystemIndex, Protocol): ...


@dataclass
class MomentumStep[State, Key: SupportsSorting](Propagator[State]):
    """Update momenta using forces according to Newton's second law.

    Implements the 'B' operator in splitting schemes, applying forces to
    update particle momenta. This is the dynamical update step that couples
    to the potential energy landscape.

    The momentum update follows:

    $$\\mathbf{p}(t+\\Delta t) = \\mathbf{p}(t) + \\mathbf{F}(t) \\cdot \\Delta t$$

    where $\\mathbf{F} = -\\nabla U$ is the force derived from potential energy $U$.

    Type Parameters:
        State: Simulation state type
        Key: Table key type (e.g. :class:`ParticleId` for atomic MD,
            :class:`GroupId` for rigid-body COM and rotational kicks).

    Attributes:
        entries: Lens to get/set the indexed table whose momenta are kicked
            (atoms in atomic MD, rigid groups in rigid-body MD). Each entry
            must expose ``momenta``, ``forces``, and ``system``.
        systems: View to extract system data with time step $\\Delta t$
    """

    entries: Lens[State, Table[Key, IsMomentumStepData]] = field(static=True)
    systems: View[State, Table[SystemId, HasTimeStep]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Apply momentum update step.

        Args:
            key: JAX PRNG key (unused in this deterministic step).
            state: Current simulation state.

        Returns:
            Updated state with new momenta.
        """
        del key  # Deterministic step
        entries_lens = self.entries.bind(state)
        entries = entries_lens.get()
        sys = self.systems(state)[entries.data.system]
        new_momenta = (
            entries.data.momenta + entries.data.forces * sys.time_step[..., None]
        )
        assert new_momenta.shape == entries.data.momenta.shape
        return entries_lens.focus(lambda x: x.data.momenta).set(new_momenta)


@runtime_checkable
class _MDParticleData(
    HasMomenta, HasPositions, HasForces, HasMasses, HasSystemIndex, Protocol
):
    @property
    def position_gradients(self) -> Array: ...


def make_velocity_verlet_step[State](
    particles: Lens[State, Table[ParticleId, _MDParticleData]],
    systems: View[State, Table[SystemId, HasTimeStep]],
    derivative_computation: Propagator[State],
    flow: Flow[State, Array],
) -> SequentialPropagator[State]:
    r"""Create a velocity Verlet integrator for molecular dynamics (NVE ensemble).

    The velocity Verlet algorithm is a symplectic, time-reversible integrator
    that provides second-order accuracy in both positions and velocities. It
    conserves total energy and samples the microcanonical (NVE) ensemble.

    Algorithm steps:

    1. $\mathbf{p}(t+\Delta t/2) = \mathbf{p}(t) + \mathbf{F}(t) \cdot \Delta t/2$ — momentum half-step
    2. $\mathbf{r}(t+\Delta t) = \mathbf{r}(t) + \mathbf{p}(t+\Delta t/2)/m \cdot \Delta t$ — position full-step
    3. Compute $\mathbf{F}(t+\Delta t)$ — force evaluation
    4. $\mathbf{p}(t+\Delta t) = \mathbf{p}(t+\Delta t/2) + \mathbf{F}(t+\Delta t) \cdot \Delta t/2$ — momentum half-step

    Args:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, positions $\\mathbf{r}$,
            forces $\\mathbf{F}$, masses $m$)
        systems: View to extract system data with time step $\\Delta t$
        derivative_computation: Propagator to compute forces $\\mathbf{F}$ from state
        flow: Flow operator for position updates (handles boundary conditions)

    Returns:
        SequentialPropagator implementing the velocity Verlet algorithm

    References:
        Swope, W. C., Andersen, H. C., Berens, P. H., & Wilson, K. R. (1982).
        A computer simulation method for the calculation of equilibrium
        constants for the formation of physical clusters of molecules:
        Application to small water clusters. J. Chem. Phys., 76(1), 637-649.
        DOI: 10.1063/1.442716
    """
    sys_with_half_time = pipe(systems, _half_time)  # Δt/2 [time]
    return SequentialPropagator(
        (
            MomentumStep(particles, sys_with_half_time),
            PositionStep(particles, systems, flow),
            derivative_computation,
            MomentumStep(particles, sys_with_half_time),
        )
    )


@runtime_checkable
class IsStochasticEntryData(HasMomenta, HasMasses, HasSystemIndex, Protocol): ...


@runtime_checkable
class _StochasticSysData(
    HasTimeStep, HasTemperature, HasFrictionCoefficient, Protocol
): ...


@dataclass
class StochasticStep[State, Key: SupportsSorting](Propagator[State]):
    """Langevin thermostat stochastic step with exact Ornstein-Uhlenbeck solution.

    Implements the 'O' operator in the BAOAB splitting scheme. This step
    exactly solves the Ornstein-Uhlenbeck stochastic differential equation:

    $$d\\mathbf{p} = -\\gamma\\mathbf{p}\\,dt + \\sqrt{2\\gamma k_B T m}\\,dW$$

    The exact solution for timestep $\\Delta t$ is:

    $$\\mathbf{p}(t+\\Delta t) = e^{-\\gamma\\Delta t} \\mathbf{p}(t) + \\sqrt{k_B T(1-e^{-2\\gamma\\Delta t})} \\sqrt{m}\\,\\eta$$

    where $\\eta \\sim \\mathcal{N}(0,1)$ is Gaussian white noise. This preserves the correct
    Maxwell-Boltzmann distribution at temperature $T$.

    Type Parameters:
        State: Simulation state type
        Key: Table key type (e.g. :class:`ParticleId`, :class:`GroupId`).

    Attributes:
        entries: Lens to get/set the indexed table whose momenta are
            thermostatted (atoms in atomic MD, rigid groups in rigid-body
            MD). Each entry must expose ``momenta``, ``masses``, ``system``.
        system: View to extract system data (time step $\\Delta t$, temperature $T$,
            friction coefficient $\\gamma$)

    References:
        Leimkuhler, B., & Matthews, C. (2013). Rational construction of
        stochastic numerical methods for molecular sampling.
        Appl. Math. Res. Express, 2013(1), 34-56.
        DOI: 10.1093/amrx/abs010
    """

    entries: Lens[State, Table[Key, IsStochasticEntryData]] = field(static=True)
    system: View[State, Table[SystemId, _StochasticSysData]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Apply stochastic Ornstein-Uhlenbeck thermostat step.

        Args:
            key: JAX PRNG key for generating random noise
            state: Current simulation state

        Returns:
            Updated state with thermostated momenta
        """
        entries_lens = self.entries.bind(state)
        entries = entries_lens.get()
        sys = self.system(state)[entries.data.system]
        thermal_energy_per_entry = sys.temperature * BOLTZMANN_CONSTANT
        # Ornstein-Uhlenbeck coefficients
        # c₁ = e^(-γΔt)
        damping_factor = jax.numpy.exp(-sys.friction_coefficient * sys.time_step)
        # c₂ = √(kT(1-e^(-2γΔt)))
        noise_amplitude = jax.numpy.sqrt(
            thermal_energy_per_entry * (1 - damping_factor**2)
        )

        noise = sample_like(jax.random.normal, key, entries.data.momenta)

        # Exact OU solution: p_new = c₁·p + c₂·√m·η
        new_momenta = (
            damping_factor[..., None] * entries.data.momenta
            + (noise_amplitude * jnp.sqrt(entries.data.masses))[..., None] * noise
        )

        assert new_momenta.shape == entries.data.momenta.shape
        return entries_lens.focus(lambda p: p.data.momenta).set(new_momenta)


def make_baoab_langevin_step[State](
    particles: Lens[State, Table[ParticleId, _MDParticleData]],
    systems: View[State, Table[SystemId, _StochasticSysData]],
    derivative_computation: Propagator[State],
    flow: Flow[State, Array],
) -> SequentialPropagator[State]:
    r"""Create BAOAB Langevin integrator for canonical (NVT) ensemble sampling.

    BAOAB is a second-order splitting scheme for Langevin dynamics that provides
    efficient sampling of the canonical ensemble. The name comes from the sequence
    of operators: B (momentum kick), A (position update), O (Ornstein-Uhlenbeck),
    A (position update), B (momentum kick).

    Algorithm steps:

    1. **B**: $\mathbf{p}(t+\Delta t/4) = \mathbf{p}(t) + \mathbf{F}(t) \cdot \Delta t/2$ — half momentum step
    2. **A**: $\mathbf{r}(t+\Delta t/2) = \mathbf{r}(t) + \mathbf{p}(t+\Delta t/4)/m \cdot \Delta t/2$ — half position step
    3. **O**: $\mathbf{p}(t+3\Delta t/4) = $ exact OU solution — stochastic thermostat
    4. **A**: $\mathbf{r}(t+\Delta t) = \mathbf{r}(t+\Delta t/2) + \mathbf{p}(t+3\Delta t/4)/m \cdot \Delta t/2$ — half position step
    5. Compute $\mathbf{F}(t+\Delta t)$ — force evaluation
    6. **B**: $\mathbf{p}(t+\Delta t) = \mathbf{p}(t+3\Delta t/4) + \mathbf{F}(t+\Delta t) \cdot \Delta t/2$ — half momentum step

    Args:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, positions $\\mathbf{r}$,
            forces $\\mathbf{F}$, masses $m$)
        systems: View to extract system data (time step $\\Delta t$, thermal energy $k_B T$,
            friction coefficient $\\gamma$)
        derivative_computation: Propagator to compute forces $\\mathbf{F}$ from state
        flow: Flow operator for position updates (handles boundary conditions)

    Returns:
        SequentialPropagator implementing the BAOAB algorithm

    References:
        Leimkuhler, B., & Matthews, C. (2013). Rational construction of
        stochastic numerical methods for molecular sampling.
        Appl. Math. Res. Express, 2013(1), 34-56. DOI: 10.1093/amrx/abs010
    """
    sys_with_half_time = pipe(systems, _half_time)
    return SequentialPropagator(
        (
            MomentumStep(particles, sys_with_half_time),  # B
            PositionStep(particles, sys_with_half_time, flow),  # A
            StochasticStep(particles, systems),  # O
            PositionStep(particles, sys_with_half_time, flow),  # A
            derivative_computation,
            MomentumStep(particles, sys_with_half_time),  # B
        )
    )


@runtime_checkable
class _CSVRSystemData(
    HasTimeStep,
    HasTemperature,
    HasThermostatTimeConstant,
    HasDegreesOfFreedom,
    Protocol,
): ...


IsCSVRParticleData = IsStochasticEntryData
"""Alias kept for backwards compatibility; structurally identical to
:class:`IsStochasticEntryData`."""


def csvr_scale_factor(
    key: Array,
    kinetic_energy: Array,
    degrees_of_freedom: Array,
    target_thermal_energy: Array,
    timestep: Array,
    thermostat_timescale: Array,
) -> Array:
    r"""Bussi–Donadio–Parrinello stochastic velocity-rescaling factor.

    Pure function, useful both for atomic and rigid-body kinetic energies.
    Returns one $\alpha$ per system: scaling momenta by $\alpha$ drives the
    distribution of total KE toward the canonical $\chi^2(N_{\text{dof}})$.

    Args:
        key: PRNG key.
        kinetic_energy: Current per-system kinetic energy $K$ (units: energy),
            shape ``(n_systems,)``.
        degrees_of_freedom: Per-system DOF count $N_{\text{dof}}$, shape ``(n_systems,)``.
        target_thermal_energy: $k_B T$ per system (units: energy), shape ``(n_systems,)``.
        timestep: $\Delta t$ per system (units: time), shape ``(n_systems,)``.
        thermostat_timescale: $\tau$ per system (units: time), shape ``(n_systems,)``.

    Returns:
        Scaling factor $\alpha$ per system, shape ``(n_systems,)``.
    """
    kinetic_energy_target = degrees_of_freedom * target_thermal_energy / 2

    key1, key2 = jax.random.split(key)
    gaussian_noise = jax.random.normal(key1, dtype=float)

    dof_minus_one = degrees_of_freedom - 1
    chi_squared_noise = jnp.where(
        dof_minus_one > 0,
        jax.random.chisquare(key2, df=dof_minus_one, dtype=float),
        0.0,
    )

    exponential_decay = jnp.exp(-timestep / thermostat_timescale)
    correction_factor = (
        (1 - exponential_decay)
        * kinetic_energy_target
        / (kinetic_energy * degrees_of_freedom)
    )

    scaling_squared = (
        exponential_decay
        + correction_factor * (gaussian_noise**2 + chi_squared_noise)
        + 2 * gaussian_noise * jnp.sqrt(exponential_decay * correction_factor)
    )
    return jnp.sqrt(jnp.maximum(scaling_squared, 0.0))


@dataclass
class CSVRStep[State](Propagator[State]):
    r"""Canonical Sampling through Velocity Rescaling (CSVR) thermostat step.

    Implements the Bussi-Donadio-Parrinello algorithm for canonical sampling
    by stochastically rescaling velocities to maintain the target temperature.
    This produces correct canonical ensemble sampling unlike deterministic
    velocity rescaling (Berendsen thermostat).

    The scaling factor $\alpha^2$ is sampled from the conditional distribution:

    $$\alpha^2 \sim (K'/K) \text{ where } K' \text{ follows the target kinetic energy distribution}$$

    The algorithm uses:

    $$\alpha^2 = c_1 + c_2(R_1^2 + R_2) + 2R_1\sqrt{c_1 c_2}$$

    where:

    - $c_1 = e^{-\Delta t/\tau}$ — exponential decay factor
    - $c_2 = (1-c_1) \cdot K_{\text{target}}/(K_{\text{current}} \cdot N_{\text{dof}})$ — correction factor
    - $R_1 \sim \mathcal{N}(0,1)$ — Gaussian random variable
    - $R_2 \sim \chi^2(N_{\text{dof}}-1)$ — chi-squared random variable

    Type Parameters:
        State: Simulation state type

    Attributes:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, masses $m$)
        systems: View to extract system data (time step $\\Delta t$, temperature $T$,
            degrees of freedom $N_{\\text{dof}}$, thermostat time constant $\\tau$)

    References:
        Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling
        through velocity rescaling. J. Chem. Phys., 126(1), 014101.
        DOI: 10.1063/1.2408420
    """

    particles: Lens[State, Table[ParticleId, IsCSVRParticleData]] = field(static=True)
    systems: View[State, Table[SystemId, _CSVRSystemData]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Apply CSVR stochastic velocity rescaling.

        Args:
            key: JAX PRNG key for generating random noise
            state: Current simulation state

        Returns:
            Updated state with rescaled momenta matching target temperature distribution
        """
        system = self.systems(state)
        particles = self.particles.get(state)

        per_particle_ke = particle_kinetic_energy(
            particles.data.momenta, particles.data.masses
        )
        kinetic_energy_current = jax.ops.segment_sum(
            per_particle_ke,
            particles.data.system.indices,
            particles.data.system.num_labels,
        )

        velocity_scale = csvr_scale_factor(
            key,
            kinetic_energy=kinetic_energy_current,
            degrees_of_freedom=system.data.degrees_of_freedom,
            target_thermal_energy=system.data.temperature * BOLTZMANN_CONSTANT,
            timestep=system.data.time_step,
            thermostat_timescale=system.data.thermostat_time_constant,
        )

        scale_per_system = velocity_scale[particles.data.system.indices]
        new_momenta = particles.data.momenta * scale_per_system[..., None]

        assert new_momenta.shape == particles.data.momenta.shape
        return (
            self.particles.bind(state).focus(lambda x: x.data.momenta).set(new_momenta)
        )


def make_csvr_step[State](
    particles: Lens[State, Table[ParticleId, _MDParticleData]],
    systems: View[State, Table[SystemId, _CSVRSystemData]],
    derivative_computation: Propagator[State],
    flow: Flow[State, Array],
) -> SequentialPropagator[State]:
    r"""Create CSVR integrator for canonical (NVT) ensemble sampling.

    Combines the CSVR thermostat with velocity Verlet integration to sample
    the canonical ensemble at constant temperature. The algorithm applies
    stochastic velocity rescaling before each velocity Verlet step.

    Algorithm steps:

    1. Apply CSVR velocity rescaling (thermostat)
    2. Velocity Verlet integration:
        - $\mathbf{p}(t+\Delta t/2) = \mathbf{p}(t) + \mathbf{F}(t) \cdot \Delta t/2$ — half momentum step
        - $\mathbf{r}(t+\Delta t) = \mathbf{r}(t) + \mathbf{p}(t+\Delta t/2)/m \cdot \Delta t$ — full position step
        - Compute $\mathbf{F}(t+\Delta t)$ — force evaluation
        - $\mathbf{p}(t+\Delta t) = \mathbf{p}(t+\Delta t/2) + \mathbf{F}(t+\Delta t) \cdot \Delta t/2$ — half momentum step

    Args:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, positions $\\mathbf{r}$,
            forces $\\mathbf{F}$, masses $m$)
        systems: View to extract system data (time step $\\Delta t$, temperature $T$,
            degrees of freedom $N_{\\text{dof}}$, thermostat time constant $\\tau$)
        derivative_computation: Propagator to compute forces $\\mathbf{F}$ from state
        flow: Flow operator for position updates (handles boundary conditions)

    Returns:
        SequentialPropagator implementing the CSVR+Verlet algorithm

    References:
        Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling
        through velocity rescaling. J. Chem. Phys., 126(1), 014101.
        DOI: 10.1063/1.2408420
    """
    systems_with_half_time = pipe(systems, _half_time)
    return SequentialPropagator(
        (
            CSVRStep(particles, systems),
            MomentumStep(particles, systems_with_half_time),
            PositionStep(particles, systems, flow),
            derivative_computation,
            MomentumStep(particles, systems_with_half_time),
        )
    )


@runtime_checkable
class _StochasticCellRescalingSystemData(
    HasUnitCell,
    HasTimeStep,
    HasTemperature,
    HasTargetPressure,
    HasPressureCouplingTime,
    HasCompressibility,
    HasMinimumScaleFactor,
    Protocol,
):
    @property
    def unitcell_gradients(self) -> UnitCell: ...


@runtime_checkable
class _BarostatEntryData(
    HasMomenta, HasPositions, HasForces, HasMasses, HasSystemIndex, Protocol
):
    @property
    def position_gradients(self) -> Array: ...


def stochastic_cell_rescaling_factor(
    key: Array,
    kinetic_energy: Array,
    cauchy_stress: Array,
    systems_data: _StochasticCellRescalingSystemData,
) -> Array:
    r"""Bernetti–Bussi 2020 isotropic linear cell-scaling factor $\mu$.

    Pure function shared by atomic and rigid-body NPT.

    $$\mu = \exp\!\left(\frac{1}{3}\big[\beta\,\frac{\Delta t}{\tau_P}\,(P - P_0) + \sqrt{\tfrac{2 k_B T \beta \Delta t}{\tau_P V}}\,R\big]\right),
       \quad R \sim \mathcal N(0, 1)$$

    Args:
        key: PRNG key.
        kinetic_energy: Per-system kinetic energy entering the pressure
            expression (translational only for rigid bodies), shape ``(n_systems,)``.
        cauchy_stress: Per-system Cauchy stress tensor, shape ``(n_systems, 3, 3)``.
        systems_data: Per-system NPT parameters.

    Returns:
        Linear scaling factor $\mu$ per system, clipped to
        ``[minimum_scale_factor, 1/minimum_scale_factor]``.
    """
    timestep = systems_data.time_step
    thermal_energy = systems_data.temperature * BOLTZMANN_CONSTANT
    barostat_timescale = systems_data.pressure_coupling_time
    compressibility = systems_data.compressibility
    volume = systems_data.unitcell.volume

    current_pressure = instantaneous_pressure(kinetic_energy, cauchy_stress, volume)
    pressure_deviation = current_pressure - systems_data.target_pressure
    depsilon_det = (
        (timestep / barostat_timescale) * compressibility * pressure_deviation
    )
    random_noise = jax.random.normal(key, dtype=volume.dtype)
    depsilon_stoch = (
        jnp.sqrt(
            2.0
            * thermal_energy
            * compressibility
            * timestep
            / (barostat_timescale * volume)
        )
        * random_noise
    )
    scaling_factor = jnp.exp((depsilon_det + depsilon_stoch) / 3.0)
    min_scaling = systems_data.minimum_scale_factor
    return jnp.clip(scaling_factor, min_scaling, 1.0 / min_scaling)


@dataclass
class StochasticCellRescalingStep[State, Key: SupportsSorting](Propagator[State]):
    """Stochastic cell rescaling barostat for atomic NPT (Bernetti & Bussi, 2020).

    Computes the per-system kinetic energy and virial stress from ``entries``,
    rescales the unit cell and ``entries.data.positions`` by the same factor.
    For rigid-body NPT (translational-KE-only pressure, molecular virial)
    use :class:`kups.md.rigid.RigidStochasticCellRescalingStep` instead.

    **Important:** The :class:`UnitCell` must be reconstructed after scaling
    to ensure the cached volume is recomputed correctly.

    Type Parameters:
        State: Simulation state type
        Key: Table key type for ``entries`` (typically :class:`ParticleId`).

    Attributes:
        entries: Lens to the table whose positions are rescaled and whose
            momenta/masses/positions feed the kinetic and virial expressions.
        systems: Lens to per-system NPT parameters and unit cell.

    References:
        Bernetti, M., & Bussi, G. (2020). Pressure control using stochastic
        cell rescaling. J. Chem. Phys., 153(11), 114107.
        DOI: 10.1063/5.0020514
    """

    entries: Lens[State, Table[Key, _BarostatEntryData]] = field(static=True)
    systems: Lens[State, Table[SystemId, _StochasticCellRescalingSystemData]] = field(
        static=True
    )

    def __call__(self, key: Array, state: State) -> State:
        entries = self.entries.get(state)
        systems = self.systems.get(state)

        per_entry_ke = particle_kinetic_energy(
            entries.data.momenta, entries.data.masses
        )
        kinetic_energy = jax.ops.segment_sum(
            per_entry_ke,
            entries.data.system.indices,
            entries.data.system.num_labels,
        )
        cauchy_stress = stress_via_virial_theorem(entries, systems).data

        scaling_factor = stochastic_cell_rescaling_factor(
            key, kinetic_energy, cauchy_stress, systems.data
        )

        new_unitcell = systems.data.unitcell * scaling_factor
        state = self.systems.focus(lambda x: x.data.unitcell).set(state, new_unitcell)

        entries_lens = self.entries.bind(state)
        entries = entries_lens.get()
        scaling_per_entry = scaling_factor[entries.data.system.indices]
        new_positions = entries.data.positions * scaling_per_entry[..., None]
        return entries_lens.focus(lambda p: p.data.positions).set(new_positions)


@runtime_checkable
class IsCSVRNPTSystemData(
    HasUnitCell,
    HasTimeStep,
    HasTemperature,
    HasTargetPressure,
    HasPressureCouplingTime,
    HasCompressibility,
    HasMinimumScaleFactor,
    HasThermostatTimeConstant,
    HasDegreesOfFreedom,
    Protocol,
):
    @property
    def unitcell_gradients(self) -> UnitCell: ...


def make_csvr_npt_step[State](
    particles: Lens[State, Table[ParticleId, _BarostatEntryData]],
    systems: Lens[State, Table[SystemId, IsCSVRNPTSystemData]],
    derivative_computation: Propagator[State],
    flow: Flow[State, Array],
) -> SequentialPropagator[State]:
    r"""Create NPT integrator for isothermal-isobaric (NPT) ensemble sampling.

    Combines CSVR thermostat for temperature control with stochastic cell
    rescaling (Bernetti-Bussi 2020) for pressure control, integrated with
    velocity Verlet dynamics. This correctly samples the NPT ensemble with
    proper volume fluctuations.

    Algorithm sequence per timestep:

    1. Apply CSVR velocity rescaling (temperature control)
    2. Velocity Verlet integration:
        - $\mathbf{p}(t+\Delta t/2) = \mathbf{p}(t) + \mathbf{F}(t) \cdot \Delta t/2$ — half momentum step
        - $\mathbf{r}(t+\Delta t) = \mathbf{r}(t) + \mathbf{p}(t+\Delta t/2)/m \cdot \Delta t$ — full position step
        - Compute $\mathbf{F}(t+\Delta t)$ — force evaluation
        - $\mathbf{p}(t+\Delta t) = \mathbf{p}(t+\Delta t/2) + \mathbf{F}(t+\Delta t) \cdot \Delta t/2$ — half momentum step
    3. Stochastic cell rescaling (pressure control)
    4. Recompute forces and stress after box/position scaling

    Args:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, positions $\\mathbf{r}$,
            forces $\\mathbf{F}$, masses $m$)
        systems: Lens to get/set system data (lattice vectors $\\mathbf{L}$, stress tensor $\\mathbf{W}$,
            time step $\\Delta t$, temperature $T$, target pressure $P_0$,
            barostat time constant $\\tau_P$, compressibility $\\beta$, minimum scale factor $\\mu_{\\text{min}}$,
            degrees of freedom $N_{\\text{dof}}$, thermostat time constant $\\tau_T$)
        derivative_computation: Propagator to compute forces $\\mathbf{F}$ and stress tensor $\\mathbf{W}$ from state
        flow: Flow operator for position updates (handles boundary conditions)

    Returns:
        SequentialPropagator implementing the CSVR-NPT algorithm

    References:
        CSVR: Bussi, G., Donadio, D., & Parrinello, M. (2007).
              Canonical sampling through velocity rescaling.
              J. Chem. Phys., 126(1), 014101. DOI: 10.1063/1.2408420
        SCR: Bernetti, M., & Bussi, G. (2020). Pressure control using
             stochastic cell rescaling. J. Chem. Phys., 153(11), 114107.
             DOI: 10.1063/5.0020514
    """
    sys_view: View[State, Table[SystemId, IsCSVRNPTSystemData]] = systems.get
    sys_half_view: View[State, Table[SystemId, IsCSVRNPTSystemData]] = pipe(
        systems.get, _half_time
    )
    return SequentialPropagator(
        (
            CSVRStep(particles, sys_view),
            MomentumStep(particles, sys_half_view),
            PositionStep(particles, sys_view, flow),
            derivative_computation,
            MomentumStep(particles, sys_half_view),
            StochasticCellRescalingStep(particles, systems),
            derivative_computation,
        )
    )


@runtime_checkable
class IsMDSystem(HasFrictionCoefficient, IsCSVRNPTSystemData, Protocol): ...


class IsMDState(Protocol):
    """State protocol for molecular dynamics step computation."""

    @property
    def particles(self) -> Table[ParticleId, _MDParticleData]: ...
    @property
    def systems(self) -> Table[SystemId, IsMDSystem]: ...


def make_md_step_from_state[State](
    state: Lens[State, IsMDState],
    derivative_computation: Propagator[State],
    integrator: Integrator,
) -> Propagator[State]:
    """Build a single MD integration step from a typed state.

    Constructs the appropriate integrator propagator by extracting views for
    particles and systems from ``state`` and wrapping them with a
    [MinimumImageConventionFlow][kups.md.integrators.MinimumImageConventionFlow]
    for periodic-boundary-condition-aware distance computations.

    Supported integrators:

    - ``"verlet"`` — [Velocity Verlet][kups.md.integrators.make_velocity_verlet_step]
      (NVE ensemble, no thermostat).
    - ``"baoab_langevin"`` — [BAOAB Langevin][kups.md.integrators.make_baoab_langevin_step]
      (NVT via Langevin friction/noise).
    - ``"csvr"`` — [CSVR][kups.md.integrators.make_csvr_step]
      (NVT via canonical-sampling velocity rescaling, constant volume).
    - ``"csvr_npt"`` — [CSVR-NPT][kups.md.integrators.make_csvr_npt_step]
      (NPT via CSVR thermostat with barostat).

    Args:
        state: Lens into the sub-state satisfying
            [IsMDState][kups.md.integrators.IsMDState] (needs ``particles`` and
            ``systems``).
        derivative_computation: Propagator that computes forces/gradients and
            updates the state (e.g. a wrapped potential).
        integrator: String key selecting the integration algorithm.

    Returns:
        [Propagator][kups.core.propagator.Propagator] that advances the
        simulation by one time step.

    Raises:
        ValueError: If ``integrator`` is not one of the supported keys.
    """
    flow = MinimumImageConventionFlow(
        state.focus(lambda x: x.systems[x.particles.data.system].unitcell),
        euclidean_flow,
    )
    match integrator:
        case "verlet":
            integrator_fn = make_velocity_verlet_step
        case "baoab_langevin":
            integrator_fn = make_baoab_langevin_step
        case "csvr":
            integrator_fn = make_csvr_step
        case "csvr_npt":
            integrator_fn = make_csvr_npt_step
        case _:
            raise ValueError(f"Unknown integrator: {integrator}")
    return integrator_fn(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        derivative_computation,
        flow,
    )
