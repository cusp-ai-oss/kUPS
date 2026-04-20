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
from kups.core.lens import Lens, View, bind
from kups.core.propagator import Propagator, SequentialPropagator
from kups.core.typing import (
    HasCompressibility,
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


def _half_time[S: HasTimeStep](sys: Table[SystemId, S]) -> Table[SystemId, S]:
    """View that halves the time_step of a system.

    Args:
        sys: Indexed system with time_step attribute

    Returns:
        New Indexed system with time_step halved
    """
    return bind(sys, lambda x: x.data.time_step).apply(lambda x: x / 2)


@runtime_checkable
class _PositionStepData(
    HasMomenta, HasPositions, HasMasses, HasSystemIndex, Protocol
): ...


@dataclass
class PositionStep[State, Data: _PositionStepData](Propagator[State]):
    """Update positions using velocities in molecular dynamics.

    Implements the 'A' operator in splitting schemes, propagating positions
    forward in time using the current velocities. This is the kinematic update
    step in velocity Verlet and related integrators.

    The position update follows:

    $$\\mathbf{r}(t+\\Delta t) = \\mathbf{r}(t) + \\mathbf{v}(t) \\cdot \\Delta t$$

    where $\\mathbf{v} = \\mathbf{p}/m$ is the velocity derived from momentum.

    Type Parameters:
        State: Simulation state type
        Data: Particle data type (must have momenta, positions, masses, system index)

    Attributes:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, positions $\\mathbf{r}$, masses $m$)
        systems: View to extract system data with time step $\\Delta t$
        flow: Flow operator defining how positions evolve (handles boundary conditions)
    """

    particles: Lens[State, Table[ParticleId, Data]] = field(static=True)
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
        # Extract current state
        particle_lens = self.particles.bind(state)
        particles = particle_lens.get()
        sys = self.systems(state)[particles.data.system]
        # Update particles: r_new = r + (p/m)·Δt
        velocity = particles.data.momenta / particles.data.masses[..., None]
        new_positions = self.flow(
            state, sys.time_step, particles.data.positions, velocity
        )
        assert new_positions.shape == particles.data.positions.shape
        return particle_lens.focus(lambda x: x.data.positions).set(new_positions)


@runtime_checkable
class IsMomentumStepData(HasMomenta, HasForces, HasSystemIndex, Protocol): ...


@dataclass
class MomentumStep[State, Data: IsMomentumStepData](Propagator[State]):
    """Update momenta using forces according to Newton's second law.

    Implements the 'B' operator in splitting schemes, applying forces to
    update particle momenta. This is the dynamical update step that couples
    to the potential energy landscape.

    The momentum update follows:

    $$\\mathbf{p}(t+\\Delta t) = \\mathbf{p}(t) + \\mathbf{F}(t) \\cdot \\Delta t$$

    where $\\mathbf{F} = -\\nabla U$ is the force derived from potential energy $U$.

    Type Parameters:
        State: Simulation state type
        Data: Particle data type (must have momenta, forces, system index)

    Attributes:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, forces $\\mathbf{F}$)
        systems: View to extract system data with time step $\\Delta t$
    """

    particles: Lens[State, Table[ParticleId, Data]] = field(static=True)
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
        # Extract current state
        particle_lens = self.particles.bind(state)
        particles = particle_lens.get()
        sys = self.systems(state)[particles.data.system]

        new_momenta = (
            particles.data.momenta + particles.data.forces * sys.time_step[..., None]
        )
        assert new_momenta.shape == particles.data.momenta.shape
        return particle_lens.focus(lambda x: x.data.momenta).set(new_momenta)


@runtime_checkable
class _MDParticleData(
    HasMomenta, HasPositions, HasForces, HasMasses, HasSystemIndex, Protocol
):
    @property
    def position_gradients(self) -> Array: ...


def make_velocity_verlet_step[State, Data: _MDParticleData](
    particles: Lens[State, Table[ParticleId, Data]],
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
class IsStochasticParticleData(HasMomenta, HasMasses, HasSystemIndex, Protocol): ...


@runtime_checkable
class _StochasticSysData(
    HasTimeStep, HasTemperature, HasFrictionCoefficient, Protocol
): ...


@dataclass
class StochasticStep[State, Data: IsStochasticParticleData](Propagator[State]):
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
        Data: Particle data type (must have momenta, masses, system index)

    Attributes:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, masses $m$)
        system: View to extract system data (time step $\\Delta t$, temperature $T$,
            friction coefficient $\\gamma$)

    References:
        Leimkuhler, B., & Matthews, C. (2013). Rational construction of
        stochastic numerical methods for molecular sampling.
        Appl. Math. Res. Express, 2013(1), 34-56.
        DOI: 10.1093/amrx/abs010
    """

    particles: Lens[State, Table[ParticleId, Data]] = field(static=True)
    system: View[State, Table[SystemId, _StochasticSysData]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Apply stochastic Ornstein-Uhlenbeck thermostat step.

        Args:
            key: JAX PRNG key for generating random noise
            state: Current simulation state

        Returns:
            Updated state with thermostated momenta
        """
        # Extract current state
        particle_lens = self.particles.bind(state)
        particles = particle_lens.get()
        sys = self.system(state)[particles.data.system]
        # kT: thermal energy [energy]
        thermal_energy_per_particle = sys.temperature * BOLTZMANN_CONSTANT
        # Ornstein-Uhlenbeck coefficients
        # c₁ = e^(-γΔt) [dimensionless]
        damping_factor = jax.numpy.exp(-sys.friction_coefficient * sys.time_step)
        # c₂ = √(kT(1-e^(-2γΔt))) [√energy]
        noise_amplitude = jax.numpy.sqrt(
            thermal_energy_per_particle * (1 - damping_factor**2)
        )

        # η ~ N(0,1) [dimensionless]
        noise = sample_like(jax.random.normal, key, particles.data.momenta)

        # Exact OU solution: p_new = c₁·p + c₂·√m·η
        new_momenta = (
            damping_factor[..., None] * particles.data.momenta
            + (noise_amplitude * jnp.sqrt(particles.data.masses))[..., None] * noise
        )

        assert new_momenta.shape == particles.data.momenta.shape
        return (
            self.particles.bind(state).focus(lambda p: p.data.momenta).set(new_momenta)
        )


def make_baoab_langevin_step[State, Data: _MDParticleData](
    particles: Lens[State, Table[ParticleId, Data]],
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
    Protocol,
): ...


@runtime_checkable
class IsCSVRParticleData(HasMomenta, HasMasses, HasSystemIndex, Protocol): ...


@dataclass
class CSVRStep[
    State,
    Data: IsCSVRParticleData,
](Propagator[State]):
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
        Data: Particle data type (must have momenta, masses, system index)

    Attributes:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, masses $m$)
        systems: View to extract system data (time step $\\Delta t$, temperature $T$,
            degrees of freedom $N_{\\text{dof}}$, thermostat time constant $\\tau$)

    References:
        Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling
        through velocity rescaling. J. Chem. Phys., 126(1), 014101.
        DOI: 10.1063/1.2408420
    """

    particles: Lens[State, Table[ParticleId, Data]] = field(static=True)
    systems: View[State, Table[SystemId, _CSVRSystemData]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Apply CSVR stochastic velocity rescaling.

        Args:
            key: JAX PRNG key for generating random noise
            state: Current simulation state

        Returns:
            Updated state with rescaled momenta matching target temperature distribution
        """
        # Extract parameters
        system = self.systems(state)
        particles = self.particles.get(state)
        # Δt: timestep [time]
        timestep = system.data.time_step
        # kT: thermal energy [energy]
        target_thermal_energy = system.data.temperature * BOLTZMANN_CONSTANT
        # τ: thermostat time constant [time]
        thermostat_timescale = system.data.thermostat_time_constant
        # N_dof: degrees of freedom [dimensionless]
        # TODO: Update once we have constraints that could limit the degrees of freedom
        degrees_of_freedom = particles.data.system.counts.data * 3 - 3

        # Compute current kinetic energy from particles
        per_particle_ke = particle_kinetic_energy(
            particles.data.momenta, particles.data.masses
        )
        # K: total kinetic energy per system [energy]
        kinetic_energy_current = jax.ops.segment_sum(
            per_particle_ke,
            particles.data.system.indices,
            particles.data.system.num_labels,
        )
        # K_target = N_dof·kT/2 [energy]
        kinetic_energy_target = degrees_of_freedom * target_thermal_energy / 2

        # Generate random numbers for scaling
        key1, key2 = jax.random.split(key)
        # R₁ ~ N(0,1) [dimensionless]
        gaussian_noise = jax.random.normal(key1, dtype=float)

        # R₂ ~ χ²(N_dof-1) [dimensionless]
        dof_minus_one = degrees_of_freedom - 1
        chi_squared_noise = jnp.where(
            dof_minus_one > 0,
            jax.random.chisquare(key2, df=dof_minus_one, dtype=float),
            0.0,
        )

        # CSVR scaling coefficients
        # c₁ = e^(-Δt/τ) [dimensionless]
        exponential_decay = jnp.exp(-timestep / thermostat_timescale)
        # c₂ = (1-c₁)·K_target/(K_current·N_dof) [dimensionless]
        correction_factor = (
            (1 - exponential_decay)
            * kinetic_energy_target
            / (kinetic_energy_current * degrees_of_freedom)
        )

        # α² = c₁ + c₂(R₁² + R₂) + 2R₁√(c₁c₂) [dimensionless]
        scaling_squared = (
            exponential_decay
            + correction_factor * (gaussian_noise**2 + chi_squared_noise)
            + 2 * gaussian_noise * jnp.sqrt(exponential_decay * correction_factor)
        )
        # α = √(α²), ensure non-negative [dimensionless]
        velocity_scale = jnp.sqrt(jnp.maximum(scaling_squared, 0.0))

        # Scale momenta by system
        scale_per_system = velocity_scale[particles.data.system.indices]
        new_momenta = particles.data.momenta * scale_per_system[..., None]

        assert new_momenta.shape == particles.data.momenta.shape
        return (
            self.particles.bind(state).focus(lambda x: x.data.momenta).set(new_momenta)
        )


def make_csvr_step[State, Data: _MDParticleData](
    particles: Lens[State, Table[ParticleId, Data]],
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
class _BarostatParticleData(_MDParticleData, Protocol): ...


@dataclass
class StochasticCellRescalingStep[
    State,
    Data: _BarostatParticleData,
    SData: _StochasticCellRescalingSystemData,
](Propagator[State]):
    """Stochastic cell rescaling barostat for NPT ensemble sampling.

    Implements the isotropic stochastic cell rescaling algorithm (Bernetti & Bussi, 2020)
    that correctly samples the NPT ensemble. This first-order barostat includes a
    stochastic term to ensure proper volume fluctuations, unlike the Berendsen
    barostat which artificially suppresses fluctuations.

    The algorithm scales both the simulation box and particle positions by a
    factor $\\mu$ determined by:

    $$\\mu \\approx 1 + \\frac{\\Delta t}{\\tau_P} \\beta (P - P_0) + \\sqrt{\\frac{2k_B T \\beta \\Delta t}{\\tau_P V}} \\, R$$

    where:

    - $\\tau_P$ = pressure coupling time constant
    - $P$ = instantaneous pressure
    - $P_0$ = target pressure
    - $\\beta$ = isothermal compressibility
    - $k_B T$ = thermal energy
    - $V$ = box volume
    - $R \\sim \\mathcal{N}(0,1)$ = Gaussian random noise

    The scaling is applied to both box and positions:

    $$\\mathbf{L}_{\\text{new}} = \\mu \\mathbf{L}, \\quad \\mathbf{r}_{\\text{new}} = \\mu \\mathbf{r}$$

    **Important:** The [UnitCell][kups.core.unitcell.UnitCell] must be reconstructed after
    scaling to ensure the cached volume is recomputed correctly.

    Type Parameters:
        State: Simulation state type
        Data: Particle data type (must have positions, momenta, masses, system index)
        SData: System data type with barostat parameters

    Attributes:
        particles: Lens to get/set indexed particle data (positions $\\mathbf{r}$, momenta $\\mathbf{p}$, masses $m$)
        systems: Lens to get/set system data (lattice vectors $\\mathbf{L}$, stress tensor $\\mathbf{W}$,
            time step $\\Delta t$, temperature $T$, target pressure $P_0$,
            barostat time constant $\\tau_P$, compressibility $\\beta$, minimum scale factor $\\mu_{\\text{min}}$)

    References:
        Bernetti, M., & Bussi, G. (2020). Pressure control using stochastic
        cell rescaling. J. Chem. Phys., 153(11), 114107.
        DOI: 10.1063/5.0020514
    """

    particles: Lens[State, Table[ParticleId, Data]] = field(static=True)
    systems: Lens[State, Table[SystemId, SData]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Apply stochastic cell rescaling for pressure control.

        Scales the simulation box and particle positions by a factor determined
        from pressure deviation and stochastic fluctuations. The UnitCell is
        reconstructed to ensure cached volume is updated correctly.

        Args:
            key: JAX PRNG key for generating volume fluctuation noise
            state: Current simulation state

        Returns:
            Updated state with rescaled box and positions matching NPT ensemble
        """
        # Extract parameters
        systems = self.systems.get(state)
        # Δt: timestep [time]
        timestep = systems.data.time_step
        # kT: thermal energy [energy]
        thermal_energy = systems.data.temperature * BOLTZMANN_CONSTANT
        # P₀: target pressure [pressure]
        target_pressure = systems.data.target_pressure
        # τP: barostat time constant [time]
        barostat_timescale = systems.data.pressure_coupling_time
        # β: isothermal compressibility [1/pressure]
        compressibility = systems.data.compressibility

        # Get current state
        # Unit cell with lattice vectors L
        unitcell = systems.data.unitcell
        # V: volume [length³]
        volume = unitcell.volume
        # Compute kinetic energy from particles
        particles = self.particles.bind(state).get()
        per_particle_ke = particle_kinetic_energy(
            particles.data.momenta, particles.data.masses
        )
        # K: total kinetic energy per system [energy]
        kinetic_energy = jax.ops.segment_sum(
            per_particle_ke,
            particles.data.system.indices,
            particles.data.system.num_labels,
        )

        # Full Cauchy stress via virial theorem:
        # σ = -(1/V)(Σ ∂U/∂r_i ⊗ r_i + h^T · ∂U/∂h)
        cauchy_stress = stress_via_virial_theorem(particles, systems).data
        # P = 2K/(dV) + Tr(σ)/d
        current_pressure = instantaneous_pressure(kinetic_energy, cauchy_stress, volume)

        # Stochastic cell rescaling (Bernetti & Bussi 2020)
        # Linearized form for small timesteps:
        # μ ≈ 1 + (Δt/τP)·β·(P - P₀) + √(2kT·β·Δt/(τP·V))·R
        # where R ~ N(0,1)

        # Stochastic cell rescaling (Bernetti & Bussi 2020, Eq. in reference impl)
        # dε = -β/τP·Δt·(P₀ - P) + √(2kT·β·Δt/(τP·V))·R
        # where dε = d(ln V) is the log-volume change. The LINEAR scaling
        # factor for lattice vectors is exp(dε/3) = (V_new/V)^(1/3).

        pressure_deviation = current_pressure - target_pressure
        # dε: log-volume change [dimensionless]
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

        depsilon = depsilon_det + depsilon_stoch
        # Linear scaling: exp(dε/3) — cube root of volume scaling
        scaling_factor = jnp.exp(depsilon / 3.0)

        # Safety clamp to prevent extreme scaling
        # μ ∈ [μ_min, μ_max]
        min_scaling = systems.data.minimum_scale_factor
        max_scaling = 1.0 / min_scaling
        scaling_factor = jnp.clip(scaling_factor, min_scaling, max_scaling)

        # Scale unit cell: L_new = μ·L
        # CRITICAL: Must reconstruct UnitCell to recompute cached volume
        # L_new = μ·L [length]
        new_unitcell = unitcell * scaling_factor
        state = self.systems.focus(lambda x: x.data.unitcell).set(state, new_unitcell)

        # Scale positions: r_new = μ·r
        particle_lens = self.particles.bind(state)
        particles = particle_lens.get()
        # μ_i: scaling factor per system [dimensionless]
        scaling_per_system = scaling_factor[particles.data.system.indices]

        new_positions = particles.data.positions * scaling_per_system[..., None]
        assert new_positions.shape == particles.data.positions.shape
        return particle_lens.focus(lambda p: p.data.positions).set(new_positions)


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
    Protocol,
):
    @property
    def unitcell_gradients(self) -> UnitCell: ...


def make_csvr_npt_step[
    State,
    Data: _BarostatParticleData,
    SData: IsCSVRNPTSystemData,
](
    particles: Lens[State, Table[ParticleId, Data]],
    systems: Lens[State, Table[SystemId, SData]],
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
    sys_view: View[State, Table[SystemId, SData]] = systems.get
    sys_half_view: View[State, Table[SystemId, SData]] = pipe(systems.get, _half_time)
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


def make_md_step_from_state[State, InpState: IsMDState](
    state: Lens[State, InpState],
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
