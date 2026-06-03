# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, overload, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Protocol

from kups.core.cell import (
    AnyPeriodicity,
    Cell,
    Periodic3D,
    require_periodic_3d_triclinic,
)
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Table
from kups.core.lens import Lens, View, bind
from kups.core.propagator import Propagator, SequentialPropagator
from kups.core.typing import (
    HasBarostatFriction,
    HasBarostatMass,
    HasCell,
    HasCellMomentum,
    HasCompressibility,
    HasForces,
    HasFrictionCoefficient,
    HasIntegratorParams,
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
    IsState,
    ParticleId,
    SystemId,
)
from kups.core.utils.functools import pipe
from kups.core.utils.jax import dataclass, field, tree_map, vectorize
from kups.core.utils.math import solve_affine_ode
from kups.core.utils.random import sample_like
from kups.md.observables import (
    instantaneous_pressure,
    instantaneous_pressure_tensor,
    particle_kinetic_energy,
    remove_center_of_mass_momentum,
)
from kups.observables.stress import stress_via_virial_theorem

type Time = Array
type Mass = Array
type Energy = Array
type Temperature = Array
type Pressure = Array
type Stress = Array

type Integrator = Literal[
    "verlet", "baoab_langevin", "csvr", "csvr_npt", "baoab_npt_langevin"
]

# χ constant in the Gao-Fang-Wang NPT Langevin barostat-pressure term
# (Appendix A, ν/d − 1). For 3D lower-triangular cell with ν = d(d+1)/2 = 6.
_GAO_CHI: int = 1


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
class WrapFlow[State, PyTree](Flow[State, PyTree]):
    """Flow that applies the cell's wrap to updated positions.

    After the base flow updates positions, applies the cell's ``wrap`` method.
    On periodic axes this folds positions back into the box (minimum image
    convention); on non-periodic axes ``wrap`` is the identity and positions
    pass through unchanged.

    Type Parameters:
        State: Simulation state type
        PyTree: JAX PyTree type for positions

    Attributes:
        cell: View to extract the [Cell][kups.core.cell.Cell] from state
        flow: Underlying flow operator (typically [euclidean_flow][kups.md.integrators.euclidean_flow])

    Example:
        ```python
        from kups.md.integrators import WrapFlow, euclidean_flow

        wrap_flow = WrapFlow(
            cell=lambda s: s.cell,
            flow=euclidean_flow
        )
        ```
    """

    cell: View[State, Cell[AnyPeriodicity]] = field(static=True)
    flow: Flow[State, PyTree] = field(static=True)

    def __call__(
        self, state: State, dt: Time, primal: PyTree, tangent: PyTree
    ) -> PyTree:
        return tree_map(self.cell(state).wrap, self.flow(state, dt, primal, tangent))


def _half_time[S: HasIntegratorParams[HasTimeStep]](
    sys: Table[SystemId, S],
) -> Table[SystemId, S]:
    """View that halves the time_step nested in ``integrator_params``.

    Args:
        sys: Indexed system whose ``integrator_params`` exposes ``time_step``.

    Returns:
        New Indexed system with ``integrator_params.time_step`` halved.
    """
    return bind(sys, lambda x: x.data.integrator_params.time_step).apply(
        lambda x: x / 2
    )


def _to_params[P](
    sys: Table[SystemId, HasIntegratorParams[P]],
) -> Table[SystemId, P]:
    """Project a Table of systems down to its ``integrator_params`` bundle.

    The resulting table shares the same primary keys but exposes the inner
    integrator parameter pytree as its ``data``.
    """
    return Table(sys.keys, sys.data.integrator_params, _cls=sys._cls)


def _half_time_params[P: HasTimeStep](
    sys: Table[SystemId, HasIntegratorParams[P]],
) -> Table[SystemId, P]:
    """Project to ``integrator_params`` with ``time_step`` halved (for splitting schemes)."""
    return _to_params(_half_time(sys))


@runtime_checkable
class _PositionStepData(
    HasMomenta, HasPositions, HasMasses, HasSystemIndex, Protocol
): ...


@dataclass
class PositionStep[State](Propagator[State]):
    """Update positions using velocities in molecular dynamics.

    Implements the 'A' operator in splitting schemes, propagating positions
    forward in time using the current velocities. This is the kinematic update
    step in velocity Verlet and related integrators.

    The position update follows:

    $$\\mathbf{r}(t+\\Delta t) = \\mathbf{r}(t) + \\mathbf{v}(t) \\cdot \\Delta t$$

    where $\\mathbf{v} = \\mathbf{p}/m$ is the velocity derived from momentum.

    Type Parameters:
        State: Simulation state type

    Attributes:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, positions $\\mathbf{r}$, masses $m$)
        systems: View to extract system data with time step $\\Delta t$
        flow: Flow operator defining how positions evolve (handles boundary conditions)
    """

    particles: Lens[State, Table[ParticleId, _PositionStepData]] = field(static=True)
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
class MomentumStep[State](Propagator[State]):
    """Update momenta using forces according to Newton's second law.

    Implements the 'B' operator in splitting schemes, applying forces to
    update particle momenta. This is the dynamical update step that couples
    to the potential energy landscape.

    The momentum update follows:

    $$\\mathbf{p}(t+\\Delta t) = \\mathbf{p}(t) + \\mathbf{F}(t) \\cdot \\Delta t$$

    where $\\mathbf{F} = -\\nabla U$ is the force derived from potential energy $U$.

    Type Parameters:
        State: Simulation state type

    Attributes:
        particles: Lens to get/set indexed particle data (momenta $\\mathbf{p}$, forces $\\mathbf{F}$)
        systems: View to extract system data with time step $\\Delta t$
    """

    particles: Lens[State, Table[ParticleId, IsMomentumStepData]] = field(static=True)
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


@runtime_checkable
class IsVerletParams(HasTimeStep, Protocol):
    r"""Integrator-params shape for :func:`make_velocity_verlet_step`."""


def make_velocity_verlet_step[State](
    particles: Lens[State, Table[ParticleId, _MDParticleData]],
    systems: View[State, Table[SystemId, HasIntegratorParams[IsVerletParams]]],
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
    params = pipe(systems, _to_params)
    params_half_time = pipe(systems, _half_time_params)  # Δt/2 [time]
    return SequentialPropagator(
        (
            MomentumStep(particles, params_half_time),
            PositionStep(particles, params, flow),
            derivative_computation,
            MomentumStep(particles, params_half_time),
        )
    )


@runtime_checkable
class IsStochasticParticleData(HasMomenta, HasMasses, HasSystemIndex, Protocol): ...


@runtime_checkable
class IsBAOABLangevinParams(
    HasTimeStep, HasTemperature, HasFrictionCoefficient, Protocol
):
    r"""Integrator-params shape for :func:`make_baoab_langevin_step`."""


@dataclass
class StochasticStep[State](Propagator[State]):
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

    particles: Lens[State, Table[ParticleId, IsStochasticParticleData]] = field(
        static=True
    )
    system: View[State, Table[SystemId, IsBAOABLangevinParams]] = field(static=True)

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
        constrained_momenta = remove_center_of_mass_momentum(
            new_momenta, particles.data.masses, particles.data.system
        )
        # Deterministic gamma=0 or dt=0 steps should remain no-ops. For
        # active Langevin thermostats, keep the sampled ensemble in the
        # zero-total-momentum subspace used by the MD temperature analysis.
        should_project = (sys.friction_coefficient > 0) & (sys.time_step > 0)
        new_momenta = jnp.where(
            should_project[..., None], constrained_momenta, new_momenta
        )

        assert new_momenta.shape == particles.data.momenta.shape
        return (
            self.particles.bind(state).focus(lambda p: p.data.momenta).set(new_momenta)
        )


def make_baoab_langevin_step[State](
    particles: Lens[State, Table[ParticleId, _MDParticleData]],
    systems: View[State, Table[SystemId, HasIntegratorParams[IsBAOABLangevinParams]]],
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
    params = pipe(systems, _to_params)
    params_half_time = pipe(systems, _half_time_params)
    return SequentialPropagator(
        (
            MomentumStep(particles, params_half_time),  # B
            PositionStep(particles, params_half_time, flow),  # A
            StochasticStep(particles, params),  # O
            PositionStep(particles, params_half_time, flow),  # A
            derivative_computation,
            MomentumStep(particles, params_half_time),  # B
        )
    )


@runtime_checkable
class IsCSVRParams(
    HasTimeStep,
    HasTemperature,
    HasThermostatTimeConstant,
    Protocol,
):
    r"""Integrator-params shape for :func:`make_csvr_step`."""


@runtime_checkable
class IsCSVRParticleData(HasMomenta, HasMasses, HasSystemIndex, Protocol): ...


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
    systems: View[State, Table[SystemId, IsCSVRParams]] = field(static=True)

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


def make_csvr_step[State](
    particles: Lens[State, Table[ParticleId, _MDParticleData]],
    systems: View[State, Table[SystemId, HasIntegratorParams[IsCSVRParams]]],
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
    params = pipe(systems, _to_params)
    params_half_time = pipe(systems, _half_time_params)
    return SequentialPropagator(
        (
            CSVRStep(particles, params),
            MomentumStep(particles, params_half_time),
            PositionStep(particles, params, flow),
            derivative_computation,
            MomentumStep(particles, params_half_time),
        )
    )


@runtime_checkable
class IsCSVRNPTParams(
    HasTimeStep,
    HasTemperature,
    HasThermostatTimeConstant,
    HasTargetPressure,
    HasPressureCouplingTime,
    HasCompressibility,
    HasMinimumScaleFactor,
    Protocol,
):
    r"""Integrator-params shape for :func:`make_csvr_npt_step`."""


@runtime_checkable
class IsMDSystem[P](HasCell[Periodic3D], HasIntegratorParams[P], Protocol):
    r"""Protocol for an MD system row: a periodic cell plus bundled integrator parameters."""


@runtime_checkable
class IsMDSystemNPT(IsMDSystem[IsCSVRNPTParams], Protocol):
    r"""NPT MD system row: adds the cell-gradient leaf needed for the barostat."""

    @property
    def cell_gradients(self) -> Cell[Periodic3D]: ...


@runtime_checkable
class _BarostatParticleData(_MDParticleData, Protocol): ...


@dataclass
class StochasticCellRescalingStep[State](Propagator[State]):
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

    **Important:** The [Cell][kups.core.cell.Cell] must be reconstructed after
    scaling to ensure the cached volume is recomputed correctly.

    Type Parameters:
        State: Simulation state type

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

    particles: Lens[State, Table[ParticleId, _BarostatParticleData]] = field(
        static=True
    )
    systems: Lens[State, Table[SystemId, IsMDSystemNPT]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Apply stochastic cell rescaling for pressure control.

        Scales the simulation box and particle positions by a factor determined
        from pressure deviation and stochastic fluctuations. The Cell is
        reconstructed to ensure cached volume is updated correctly.

        Args:
            key: JAX PRNG key for generating volume fluctuation noise
            state: Current simulation state

        Returns:
            Updated state with rescaled box and positions matching NPT ensemble
        """
        # Extract parameters
        systems = self.systems.get(state)
        params = systems.data.integrator_params
        # Δt: timestep [time]
        timestep = params.time_step
        # kT: thermal energy [energy]
        thermal_energy = params.temperature * BOLTZMANN_CONSTANT
        # P₀: target pressure [pressure]
        target_pressure = params.target_pressure
        # τP: barostat time constant [time]
        barostat_timescale = params.pressure_coupling_time
        # β: isothermal compressibility [1/pressure]
        compressibility = params.compressibility

        # Get current state
        # Cell with lattice vectors L
        cell = systems.data.cell
        # V: volume [length³]
        volume = cell.volume
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
        min_scaling = params.minimum_scale_factor
        max_scaling = 1.0 / min_scaling
        scaling_factor = jnp.clip(scaling_factor, min_scaling, max_scaling)

        # Scale cell: L_new = μ·L
        # CRITICAL: Must reconstruct Cell to recompute cached volume
        # L_new = μ·L [length]
        new_cell = cell * scaling_factor
        state = self.systems.focus(lambda x: x.data.cell).set(state, new_cell)

        # Scale positions: r_new = μ·r
        particle_lens = self.particles.bind(state)
        particles = particle_lens.get()
        # μ_i: scaling factor per system [dimensionless]
        scaling_per_system = scaling_factor[particles.data.system.indices]

        new_positions = particles.data.positions * scaling_per_system[..., None]
        assert new_positions.shape == particles.data.positions.shape
        return particle_lens.focus(lambda p: p.data.positions).set(new_positions)


def make_csvr_npt_step[State](
    particles: Lens[State, Table[ParticleId, _BarostatParticleData]],
    systems: Lens[State, Table[SystemId, IsMDSystemNPT]],
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
    params: View[State, Table[SystemId, IsCSVRNPTParams]] = pipe(
        systems.get, _to_params
    )
    params_half: View[State, Table[SystemId, IsCSVRNPTParams]] = pipe(
        systems.get, _half_time_params
    )
    return SequentialPropagator(
        (
            CSVRStep(particles, params),
            MomentumStep(particles, params_half),
            PositionStep(particles, params, flow),
            derivative_computation,
            MomentumStep(particles, params_half),
            StochasticCellRescalingStep(particles, systems),
            derivative_computation,
        )
    )


# =============================================================================
# Gao-Fang-Wang BAOAB NPT Langevin (JCP 2016, arxiv 1601.01044)
# Fully-flexible-cell extended-variable NPT Langevin dynamics.
# =============================================================================


@runtime_checkable
class IsBAOABNPTLangevinParams(
    HasTimeStep,
    HasTemperature,
    HasFrictionCoefficient,
    HasTargetPressure,
    HasPressureCouplingTime,
    HasCompressibility,
    HasBarostatMass,
    HasBarostatFriction,
    Protocol,
):
    r"""Integrator-params shape for :func:`make_baoab_npt_langevin_step`."""


@runtime_checkable
class IsBAOABNPTLangevinSystem(
    HasCell[Periodic3D],
    HasCellMomentum,
    HasIntegratorParams[IsBAOABNPTLangevinParams],
    Protocol,
):
    r"""NPT Langevin MD system row: periodic cell, cell-momentum tensor, integrator
    params, and a cell-gradient leaf for the virial."""

    @property
    def cell_gradients(self) -> Cell[Periodic3D]: ...


def _cell_velocity(cell_momentum: Array, barostat_mass: Array) -> Array:
    """Element-wise ``V_dot = p^h / M`` on the lower-triangular DOFs.

    Both inputs have zeros in the strict-upper triangle by construction; a
    naive division would propagate ``0/0 = NaN`` there. Mask the strict-upper
    triangle to zero.
    """
    safe_M = jnp.where(barostat_mass != 0, barostat_mass, 1.0)
    return jnp.tril(jnp.where(barostat_mass != 0, cell_momentum / safe_M, 0.0))


@dataclass
class CellMomentumKick[State](Propagator[State]):
    r"""B$^h$ kick (Gao Algorithm lines 2 & 13).

    Updates the cell-momentum tensor by the pressure-deviation virial. In
    paper convention (Eq. 8d) the deterministic term is

    $$\text{kick}^{\text{paper}} = \det(h)\,(P_{\text{ins}} - P_0 I)\,h^{-T}
                                   - \chi k_B T\, h^{-T}.$$

    In kUPS convention $V = h^T$, so $h^{-T} = V^{-1}$, and $p^h_{\text{kUPS}}
    = (p^h_{\text{paper}})^T$. Transposing the kick and projecting to the
    lower triangle gives the kUPS-side update applied here:

    $$p^h \leftarrow p^h + \Delta t \cdot \text{tril}\!\left(
        \det(V)\,V^{-T}(P_{\text{ins}} - P_0 I) - \chi k_B T\, V^{-T}
    \right),$$

    The integration uses $\Delta t / 2$ — the algorithm calls this kick four
    times per step, twice at the start and twice at the end of the BAOAB
    palindrome, each at a half time-step. $P_{\text{ins}}$ is the symmetric
    pressure tensor from
    [instantaneous_pressure_tensor][kups.md.observables.instantaneous_pressure_tensor]
    and includes the lattice-gradient contribution. $\chi = 1$ for the
    3D lower-triangular cell parameterisation (Appendix A, $\nu/d-1$).
    """

    particles: Lens[State, Table[ParticleId, _BarostatParticleData]] = field(
        static=True
    )
    systems: Lens[State, Table[SystemId, IsBAOABNPTLangevinSystem]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key  # Deterministic step
        sys = self.systems.get(state)
        par = self.particles.get(state)
        params = sys.data.integrator_params

        V_inv = sys.data.cell.inverse_vectors  # lower-tri
        V_det = sys.data.cell.volume  # (n,)
        chi_kT = _GAO_CHI * params.temperature * BOLTZMANN_CONSTANT  # (n,)
        P_target = params.target_pressure  # (n,)
        P_ins = instantaneous_pressure_tensor(par, sys)  # (n, 3, 3) symmetric
        P_diff = P_ins - P_target[..., None, None] * jnp.eye(3)

        # Paper-side kick = det(h)·(P_diff @ V_inv) − χkT·V_inv.
        # kUPS-side kick = (paper kick)^T  (since kUPS p^h = paper p^h transposed).
        kick_paper = (
            V_det[..., None, None] * (P_diff @ V_inv) - chi_kT[..., None, None] * V_inv
        )
        kick_kups = jnp.tril(jnp.swapaxes(kick_paper, -1, -2))

        # Half time-step: this kick fires twice per BAOAB step on each side.
        dt_half = params.time_step[..., None, None] / 2
        new_cell_momentum = sys.data.cell_momentum + dt_half * kick_kups
        return self.systems.focus(lambda x: x.data.cell_momentum).set(
            state, new_cell_momentum
        )


@dataclass
class CellPositionStep[State](Propagator[State]):
    r"""A$^h$ drift (Gao Algorithm lines 4 & 9): cell-only drift.

    Implements the paper's :math:`\mathcal{F}^h_K = \sum_{\alpha\beta}
    (p^h_{\alpha\beta}/M_{\alpha\beta})\,\partial/\partial h_{\alpha\beta}`,
    which moves only $h$:

    $$V \leftarrow V + \frac{\Delta t}{2}\,p^h / M.$$

    **Atomic positions are not modified here.** The convective response of
    particle positions to cell motion ($\dot{h} h^{-1} r$, paper Eq. 8a) is
    handled in :class:`CoupledPositionStep` via the analytic ODE solver;
    rescaling positions in this step too would double-count the convective
    term and break the symplectic Trotter splitting.

    The new cell is reconstructed as a :class:`TriclinicFrame` from the
    drifted lower-triangular vectors, which forces a fresh ``volume`` cache.
    """

    systems: Lens[State, Table[SystemId, IsBAOABNPTLangevinSystem]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key  # Deterministic step
        sys = self.systems.get(state)
        params = sys.data.integrator_params
        V_old = sys.data.cell.vectors  # lower-tri
        V_dot = _cell_velocity(sys.data.cell_momentum, params.barostat_mass)
        # Half time-step (BAOAB: drifts twice per side, each at Δt/2).
        # V_old, V_dot both lower-tri ⇒ V_new lower-tri.
        V_new = V_old + (params.time_step[..., None, None] / 2) * V_dot
        # Reconstruct TriclinicFrame so cached volume refreshes.
        return self.systems.focus(lambda x: x.data.cell.frame).apply(
            state, lambda frame: frame.from_matrix(V_new)
        )


@dataclass
class CellStochasticStep[State](Propagator[State]):
    r"""O$^h$ Ornstein–Uhlenbeck thermostat on the cell-momentum tensor
    (Gao Algorithm line 6).

    Exact OU solution per lower-triangular component:

    $$p^h_{\alpha\beta} \leftarrow e^{-\gamma_{\alpha\beta}\Delta t}\,p^h_{\alpha\beta}
        + \sqrt{M_{\alpha\beta}\,k_B T\,(1 - e^{-2\gamma_{\alpha\beta}\Delta t})}\,R_{\alpha\beta},$$

    with $R_{\alpha\beta} \sim \mathcal{N}(0, 1)$. Noise is masked to the
    lower triangle so strict-upper components of $p^h$ remain zero.
    """

    systems: Lens[State, Table[SystemId, IsBAOABNPTLangevinSystem]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        sys = self.systems.get(state)
        params = sys.data.integrator_params
        gamma = params.barostat_friction
        M = params.barostat_mass
        kT = params.temperature * BOLTZMANN_CONSTANT
        dt = params.time_step
        damp = jnp.exp(-gamma * dt[..., None, None])
        noise_amp = jnp.sqrt(M * kT[..., None, None] * (1.0 - damp**2))
        R = jnp.tril(jax.random.normal(key, M.shape, dtype=M.dtype))
        new_cell_momentum = jnp.tril(damp * sys.data.cell_momentum + noise_amp * R)
        return self.systems.focus(lambda x: x.data.cell_momentum).set(
            state, new_cell_momentum
        )


@dataclass
class CoupledMomentumStep[State](Propagator[State]):
    r"""B kick with cell coupling (Gao Algorithm lines 3 & 11).

    Solves $\dot{p}_i = F_i - h^{-\top}\dot{h}^{\top} p_i$ exactly over the
    params time-step via :func:`solve_affine_ode`. In kUPS row-vector form
    the coupling becomes $\dot{p}_{\text{row}} = F_{\text{row}} - p_{\text{row}}\cdot M^{\top}$
    with $M = V^{-1}\dot{V}$ (lower-triangular); the equivalent column-vector
    ODE has $A = -M$.

    Half time-step: the BAOAB palindrome calls this kick four times per
    full step, each at $\Delta t / 2$.
    """

    particles: Lens[State, Table[ParticleId, _BarostatParticleData]] = field(
        static=True
    )
    systems: Lens[State, Table[SystemId, IsBAOABNPTLangevinSystem]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        sys = self.systems.get(state)
        par_lens = self.particles.bind(state)
        par = par_lens.get()
        params = sys.data.integrator_params
        V_dot = _cell_velocity(sys.data.cell_momentum, params.barostat_mass)
        M_pos = sys.data.cell.inverse_vectors @ V_dot  # lower-tri
        A_per_p = -M_pos[par.data.system.indices]  # column-form A
        dt_half_per_p = params.time_step[par.data.system.indices] / 2
        new_momenta = jax.vmap(solve_affine_ode)(
            A_per_p, par.data.forces, par.data.momenta, dt_half_per_p
        )
        return par_lens.focus(lambda p: p.data.momenta).set(new_momenta)


@dataclass
class CoupledPositionStep[State](Propagator[State]):
    r"""A drift with cell coupling (Gao Algorithm lines 5 & 8).

    Solves $\dot{r}_i = p_i/m_i + \dot{h}h^{-1} r_i$ exactly over the params
    time-step via :func:`solve_affine_ode`. In kUPS row-vector form the
    coupling becomes $\dot{r}_{\text{row}} = v_{\text{row}} + r_{\text{row}}\cdot M$
    with $M = V^{-1}\dot{V}$ (lower-triangular); the column-vector ODE has
    $A = M^{\top}$ (upper-triangular).

    Half time-step: the BAOAB palindrome calls this drift four times per
    full step, each at $\Delta t / 2$.
    """

    particles: Lens[State, Table[ParticleId, _BarostatParticleData]] = field(
        static=True
    )
    systems: Lens[State, Table[SystemId, IsBAOABNPTLangevinSystem]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        sys = self.systems.get(state)
        par_lens = self.particles.bind(state)
        par = par_lens.get()
        params = sys.data.integrator_params
        V_dot = _cell_velocity(sys.data.cell_momentum, params.barostat_mass)
        M_pos = sys.data.cell.inverse_vectors @ V_dot  # lower-tri
        A_per_p = jnp.swapaxes(M_pos, -1, -2)[par.data.system.indices]  # upper-tri
        dt_half_per_p = params.time_step[par.data.system.indices] / 2
        v = par.data.momenta / par.data.masses[..., None]
        new_positions = jax.vmap(solve_affine_ode)(
            A_per_p, v, par.data.positions, dt_half_per_p
        )
        return par_lens.focus(lambda p: p.data.positions).set(new_positions)


@dataclass
class WrapStep[State](Propagator[State]):
    """Apply the cell's wrap operation to particle positions.

    Standalone version of :class:`WrapFlow` for use at the end of a coupled
    integration step where the per-A-step wrap would break the analytic
    affine-ODE solution.
    """

    particles: Lens[State, Table[ParticleId, _BarostatParticleData]] = field(
        static=True
    )
    systems: View[State, Table[SystemId, HasCell[Periodic3D]]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        par_lens = self.particles.bind(state)
        par = par_lens.get()
        cells = self.systems(state)[par.data.system]
        new_positions = cells.cell.wrap(par.data.positions)
        return par_lens.focus(lambda p: p.data.positions).set(new_positions)


def make_baoab_npt_langevin_step[State](
    particles: Lens[State, Table[ParticleId, _BarostatParticleData]],
    systems: Lens[State, Table[SystemId, IsBAOABNPTLangevinSystem]],
    derivative_computation: Propagator[State],
    flow: Flow[State, Array],
) -> SequentialPropagator[State]:
    r"""Create the fully-flexible-cell BAOAB NPT Langevin integrator.

    Implements the Trotter-split scheme of Gao, Fang & Wang
    (*Sampling the isothermal-isobaric ensemble by Langevin dynamics*,
    arxiv 1601.01044, JCP 2016). Each step is a palindrome of 12
    sub-operators around a single OU pair, with one force/stress
    evaluation per step:

    ``B^h ▸ B ▸ A^h ▸ A ▸ O^h ▸ O ▸ A ▸ A^h ▸ wrap ▸ F ▸ B ▸ B^h``

    Algorithm (paper §III lines 2–13):

    1. **B^h** ($\Delta t/2$): kick cell-momentum by the pressure deviation.
    2. **B** ($\Delta t/2$): coupled atom-momentum kick, $\dot p = F - h^{-T}\dot h^{T} p$.
    3. **A^h** ($\Delta t/2$): drift the cell matrix only.
    4. **A** ($\Delta t/2$): coupled atom-position drift, $\dot r = p/m + \dot h h^{-1} r$.
    5. **O^h** ($\Delta t$): exact OU thermostat on cell-momentum.
    6. **O** ($\Delta t$): exact OU thermostat on atom-momentum (reuses
       :class:`StochasticStep`).
    7. **A** ($\Delta t/2$): repeat.
    8. **A^h** ($\Delta t/2$): repeat.
    9. **wrap**: single periodic wrap of all positions.
    10. **F**: recompute forces and cell-gradients at the wrapped coordinates
        retained in the state.
    11. **B** ($\Delta t/2$): repeat.
    12. **B^h** ($\Delta t/2$): repeat.

    The atom-side OU step is the existing :class:`StochasticStep`
    (unchanged); only the cell-side machinery is new.

    The ``flow`` argument is unused (kept for signature compatibility with
    the other ``make_*_step`` factories); periodic wrapping is handled
    in-step by a single :class:`WrapStep` just before the force evaluation.

    Args:
        particles: Lens onto the per-particle table (momenta, positions,
            forces, masses, system).
        systems: Lens onto the per-system table providing the cell, the
            extended-variable cell-momentum, ``cell_gradients`` and the
            integrator-params bundle (:class:`BAOABNPTLangevinParams`).
        derivative_computation: Propagator that updates ``position_gradients``
            and ``cell_gradients`` from the current state.
        flow: Unused — present only for API parity with sibling factories.

    Returns:
        :class:`SequentialPropagator` implementing the Gao–Fang–Wang scheme.

    References:
        Gao, X., Fang, J., & Wang, H. (2016). Sampling the
        isothermal-isobaric ensemble by Langevin dynamics.
        J. Chem. Phys., 144, 124113. arxiv 1601.01044.
    """
    del flow  # Wrapping is handled by an explicit WrapStep.

    # The atom-side OU step is keyed off a HasFrictionCoefficient/HasTemperature
    # params view; project to the params slot so the existing StochasticStep
    # works without modification.
    params: View[State, Table[SystemId, IsBAOABNPTLangevinParams]] = pipe(
        systems.get, _to_params
    )
    return SequentialPropagator(
        (
            CellMomentumKick(particles, systems),  # B^h  Δt/2
            CoupledMomentumStep(particles, systems),  # B    Δt/2
            CellPositionStep(systems),  # A^h  Δt/2
            CoupledPositionStep(particles, systems),  # A    Δt/2
            CellStochasticStep(systems),  # O^h  Δt
            StochasticStep(particles, params),  # O    Δt  (reused)
            CoupledPositionStep(particles, systems),  # A    Δt/2
            CellPositionStep(systems),  # A^h  Δt/2
            WrapStep(particles, systems.get),  # wrap
            derivative_computation,  # F
            CoupledMomentumStep(particles, systems),  # B    Δt/2
            CellMomentumKick(particles, systems),  # B^h  Δt/2
        )
    )


def require_baoab_npt_langevin_state(
    systems: Table[SystemId, IsBAOABNPTLangevinSystem],
) -> None:
    """Runtime check that the system table satisfies the BAOAB NPT Langevin shape.

    Call this once on the initial state (outside of jit) before constructing
    the integrator. Verifies that the cell is a 3D-periodic
    :class:`TriclinicFrame` and that the integrator-params is a
    :class:`BAOABNPTLangevinParams`-shaped bundle.
    """
    require_periodic_3d_triclinic(systems.data.cell)


@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsMDSystem[IsVerletParams]]],
    derivative_computation: Propagator[State],
    integrator: Literal["verlet"],
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsMDSystem[IsBAOABLangevinParams]]],
    derivative_computation: Propagator[State],
    integrator: Literal["baoab_langevin"],
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsMDSystem[IsCSVRParams]]],
    derivative_computation: Propagator[State],
    integrator: Literal["csvr"],
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsMDSystemNPT]],
    derivative_computation: Propagator[State],
    integrator: Literal["csvr_npt"],
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, IsState[_MDParticleData, IsBAOABNPTLangevinSystem]],
    derivative_computation: Propagator[State],
    integrator: Literal["baoab_npt_langevin"],
) -> Propagator[State]: ...
@overload
def make_md_step_from_state[State](
    state: Lens[State, Any],
    derivative_computation: Propagator[State],
    integrator: Integrator,
) -> Propagator[State]: ...


def make_md_step_from_state[State](
    state: Lens[State, Any],
    derivative_computation: Propagator[State],
    integrator: Integrator,
) -> Propagator[State]:
    """Build a single MD integration step from a typed state.

    Constructs the appropriate integrator propagator by extracting views for
    particles and systems from ``state`` and wrapping them with a
    [WrapFlow][kups.md.integrators.WrapFlow]
    for periodic-boundary-condition-aware distance computations.

    Supported integrators:

    - ``"verlet"`` — [Velocity Verlet][kups.md.integrators.make_velocity_verlet_step]
      (NVE ensemble, no thermostat). Requires ``integrator_params: IsVerletParams``.
    - ``"baoab_langevin"`` — [BAOAB Langevin][kups.md.integrators.make_baoab_langevin_step]
      (NVT via Langevin friction/noise). Requires
      ``integrator_params: IsBAOABLangevinParams``.
    - ``"csvr"`` — [CSVR][kups.md.integrators.make_csvr_step]
      (NVT via canonical-sampling velocity rescaling, constant volume). Requires
      ``integrator_params: IsCSVRParams``.
    - ``"csvr_npt"`` — [CSVR-NPT][kups.md.integrators.make_csvr_npt_step]
      (NPT via CSVR thermostat with barostat). Requires
      ``integrator_params: IsCSVRNPTParams`` and a ``cell_gradients`` leaf on each system.
    - ``"baoab_npt_langevin"`` — [BAOAB NPT Langevin][kups.md.integrators.make_baoab_npt_langevin_step]
      (Gao–Fang–Wang JCP 2016 fully-flexible-cell NPT). Requires
      ``integrator_params: IsBAOABNPTLangevinParams`` plus ``cell_momentum``
      and ``cell_gradients`` leaves on each system, and a
      :class:`~kups.core.cell.TriclinicFrame` cell.

    The overloads narrow the required state protocol per ``integrator`` literal.

    Args:
        state: Lens into the sub-state with ``particles`` and ``systems``.
        derivative_computation: Propagator that computes forces/gradients and
            updates the state (e.g. a wrapped potential).
        integrator: String key selecting the integration algorithm.

    Returns:
        [Propagator][kups.core.propagator.Propagator] that advances the
        simulation by one time step.

    Raises:
        ValueError: If ``integrator`` is not one of the supported keys.
    """
    flow = WrapFlow(
        state.focus(lambda x: x.systems[x.particles.data.system].cell),
        euclidean_flow,
    )
    particles_lens = state.focus(lambda x: x.particles)
    systems_lens = state.focus(lambda x: x.systems)
    match integrator:
        case "verlet":
            return make_velocity_verlet_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case "baoab_langevin":
            return make_baoab_langevin_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case "csvr":
            return make_csvr_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case "csvr_npt":
            return make_csvr_npt_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case "baoab_npt_langevin":
            return make_baoab_npt_langevin_step(
                particles_lens, systems_lens, derivative_computation, flow
            )
        case _:
            raise ValueError(f"Unknown integrator: {integrator}")
