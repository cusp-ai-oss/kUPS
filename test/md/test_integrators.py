# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MD integrators."""

from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import HasLensFields, LensField, lens
from kups.core.propagator import CachePropagator, Propagator
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, jit
from kups.md.integrators import (
    CellMomentumKick,
    CellPositionStep,
    CellStochasticStep,
    CoupledMomentumStep,
    CoupledPositionStep,
    CSVRStep,
    MomentumStep,
    PositionStep,
    StochasticCellRescalingStep,
    StochasticStep,
    WrapFlow,
    WrapStep,
    euclidean_flow,
    make_baoab_langevin_step,
    make_baoab_npt_langevin_step,
    make_csvr_npt_step,
    make_csvr_step,
    make_velocity_verlet_step,
)
from kups.md.observables import (
    instantaneous_pressure,
    instantaneous_pressure_tensor,
    particle_kinetic_energy,
)

from ..clear_cache import clear_cache  # noqa: F401

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ParticleData:
    positions: Array
    momenta: Array
    forces: Array
    masses: Array
    system: Index[SystemId]
    position_gradients: Array


@dataclass
class SystemParams:
    """Bundled integrator-params for the simple harmonic-system tests.

    Carries the union of fields needed by BAOAB Langevin and CSVR so the same
    state shape can drive both integrators.
    """

    time_step: Array
    temperature: Array
    friction_coefficient: Array
    thermostat_time_constant: Array


@dataclass
class NPTSystemParams:
    """Bundled integrator-params for CSVR-NPT tests."""

    time_step: Array
    temperature: Array
    thermostat_time_constant: Array
    target_pressure: Array
    pressure_coupling_time: Array
    compressibility: Array
    minimum_scale_factor: Array


@dataclass
class SystemData:
    integrator_params: SystemParams


@dataclass
class NPTSystemData:
    cell: Cell
    cell_gradients: Cell
    integrator_params: NPTSystemParams


@dataclass
class SimpleState(HasLensFields):
    particles: LensField[Table[ParticleId, ParticleData]]
    systems: LensField[Table[SystemId, SystemData]]


@dataclass
class NPTState(HasLensFields):
    particles: LensField[Table[ParticleId, ParticleData]]
    systems: LensField[Table[SystemId, NPTSystemData]]


# ============================================================================
# Helpers
# ============================================================================


def compute_temperature(state, dof):
    """Compute instantaneous temperature from state."""
    ke = jnp.sum(
        particle_kinetic_energy(
            state.particles.data.momenta, state.particles.data.masses
        )
    )
    return 2.0 * ke / dof


def get_systems(s: SimpleState) -> Table[SystemId, SystemData]:
    """Extract Table SystemData from state."""
    return s.systems


def get_params(s: SimpleState) -> Table[SystemId, SystemParams]:
    """Project the system table to its integrator_params bundle."""
    sys = s.systems
    return Table(sys.keys, sys.data.integrator_params, _cls=sys._cls)


def run_simulation(integrator, state, key, n_equil, n_sample, extract_fn):
    """Run equilibration + sampling with jax.lax.scan."""

    def step_fn(carry, _):
        key, s = carry
        key, subkey = jax.random.split(key)
        s = integrator(subkey, s)
        return (key, s), extract_fn(s)

    @jit
    def run(key, state):
        (key, state), _ = jax.lax.scan(step_fn, (key, state), None, length=n_equil)
        (_, state), samples = jax.lax.scan(step_fn, (key, state), None, length=n_sample)
        return state, samples

    return run(key, state)


def assert_temperature(mean_temp, kT_target, tolerance, label=""):
    """Assert temperature is within tolerance of target."""
    rel_err = jnp.abs(mean_temp - kT_target) / kT_target
    assert rel_err < tolerance, (
        f"{label}Temperature {mean_temp:.3f} differs from target {kT_target} by {rel_err * 100:.1f}%"
    )


def _virial_stress(positions, forces):
    """Compute virial stress tensor (vectorized)."""
    return jnp.einsum("ij,ik->jk", positions, forces)


# ============================================================================
# Fixtures
# ============================================================================


def create_harmonic_system(
    n_particles=10, k=1.0, m=1.0, kT=1.0, dt=0.01, tau=0.1, gamma=1.0, key=None
):
    """Create harmonic oscillator system for testing."""
    if key is None:
        key = jax.random.key(42)
    key1, key2 = jax.random.split(key)

    positions = jax.random.normal(key1, (n_particles, 3)) * 0.1
    momenta = jax.random.normal(key2, (n_particles, 3)) * jnp.sqrt(m * kT)
    forces = -k * positions
    masses = jnp.full((n_particles,), m)

    system_index = Index.new([SystemId(0)] * n_particles)
    particles = Table.arange(
        ParticleData(
            positions=positions,
            momenta=momenta,
            forces=forces,
            masses=masses,
            system=system_index,
            position_gradients=-forces,
        ),
        label=ParticleId,
    )

    systems = Table.arange(
        SystemData(
            integrator_params=SystemParams(
                time_step=jnp.array([dt]),
                temperature=jnp.array([kT / BOLTZMANN_CONSTANT]),
                friction_coefficient=jnp.array([gamma]),
                thermostat_time_constant=jnp.array([tau]),
            ),
        ),
        label=SystemId,
    )

    state = SimpleState(particles=particles, systems=systems)

    def compute_forces_fn(s):
        forces = -k * s.particles.data.positions
        return Table(
            s.particles.keys,
            ParticleData(
                positions=s.particles.data.positions,
                momenta=s.particles.data.momenta,
                forces=forces,
                masses=s.particles.data.masses,
                system=s.particles.data.system,
                position_gradients=-forces,
            ),
        )

    derivative_computation = CachePropagator(
        lambda key, state: compute_forces_fn(state).data.forces,
        lens(lambda s: s.particles, cls=SimpleState).focus(lambda p: p.data.forces).set,
    )

    return state, derivative_computation, compute_forces_fn


def create_npt_system(
    n_particles=10,
    k=1.0,
    m=1.0,
    box_size=5.0,
    kT=1.0,
    target_pressure=0.0,
    dt=0.01,
    tau_t=0.1,
    tau_p=1.0,
    compressibility=10.0,
    key=None,
):
    """Create NPT system for testing barostat."""
    if key is None:
        key = jax.random.key(42)
    key1, key2 = jax.random.split(key)

    positions = (jax.random.uniform(key1, (n_particles, 3)) - 0.5) * box_size * 0.8
    momenta = jax.random.normal(key2, (n_particles, 3)) * jnp.sqrt(m * kT)
    forces = -k * positions
    masses = jnp.full((n_particles,), m)

    system_index = Index.new([SystemId(0)] * n_particles)
    particles = Table.arange(
        ParticleData(
            positions=positions,
            momenta=momenta,
            forces=forces,
            masses=masses,
            system=system_index,
            position_gradients=-forces,
        ),
        label=ParticleId,
    )

    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * box_size))
    from kups.core.utils.jax import tree_zeros_like

    systems = Table.arange(
        NPTSystemData(
            cell=cell,
            cell_gradients=tree_zeros_like(cell),
            integrator_params=NPTSystemParams(
                time_step=jnp.array([dt]),
                temperature=jnp.array([kT / BOLTZMANN_CONSTANT]),
                thermostat_time_constant=jnp.array([tau_t]),
                target_pressure=jnp.array([target_pressure]),
                pressure_coupling_time=jnp.array([tau_p]),
                compressibility=jnp.array([compressibility]),
                minimum_scale_factor=jnp.array([0.5]),
            ),
        ),
        label=SystemId,
    )

    return NPTState(particles=particles, systems=systems)


def create_npt_derivative_computation():
    """Create derivative computation propagator for NPT tests."""

    def derivative_step(key, s):
        forces = -1.0 * s.particles.data.positions

        new_particles = Table(
            s.particles.keys,
            ParticleData(
                positions=s.particles.data.positions,
                momenta=s.particles.data.momenta,
                forces=forces,
                masses=s.particles.data.masses,
                system=s.particles.data.system,
                position_gradients=-forces,
            ),
            _cls=s.particles._cls,
        )
        return NPTState(particles=new_particles, systems=s.systems)

    @dataclass
    class DerivativeComputation(Propagator[NPTState]):
        def __call__(self, key, state):
            return derivative_step(key, state)

    return DerivativeComputation()


# ============================================================================
# Tests: Individual Components
# ============================================================================


class TestBasicSteps:
    """Tests for PositionStep and MomentumStep (merged to share setup)."""

    def test_position_update(self):
        """Position update correctness and momenta preservation."""
        state, _, _ = create_harmonic_system(n_particles=5, dt=0.01)
        step = PositionStep(
            particles=SimpleState.particles,
            systems=get_params,
            flow=euclidean_flow,
        )
        new_state = step(jax.random.key(0), state)

        velocities = state.particles.data.momenta / state.particles.data.masses[:, None]
        expected = (
            state.particles.data.positions
            + velocities * state.systems.data.integrator_params.time_step[0]
        )
        assert jnp.allclose(new_state.particles.data.positions, expected, rtol=1e-6)
        assert jnp.allclose(
            new_state.particles.data.momenta, state.particles.data.momenta
        )

    def test_momentum_update(self):
        """Momentum update correctness and position preservation."""
        state, _, _ = create_harmonic_system(n_particles=5, dt=0.01)
        step = MomentumStep(particles=SimpleState.particles, systems=get_params)
        new_state = step(jax.random.key(0), state)

        expected = (
            state.particles.data.momenta
            + state.particles.data.forces
            * state.systems.data.integrator_params.time_step[0]
        )
        assert jnp.allclose(new_state.particles.data.momenta, expected, rtol=1e-6)
        assert jnp.allclose(
            new_state.particles.data.positions, state.particles.data.positions
        )


class TestThermostatSteps:
    """Tests for StochasticStep and CSVRStep (merged to share JIT cache)."""

    def test_stochastic_temperature_preservation(self):
        n_particles, kT_target = 10, 1.5
        state, _, _ = create_harmonic_system(
            n_particles=n_particles, kT=kT_target, dt=0.02
        )
        step = StochasticStep(particles=SimpleState.particles, system=get_params)

        _, temps = run_simulation(
            step,
            state,
            jax.random.key(42),
            n_equil=50,
            n_sample=100,
            extract_fn=lambda s: compute_temperature(s, 3 * n_particles - 3),
        )
        assert_temperature(jnp.mean(temps), kT_target, 0.2)

    def test_stochastic_step_removes_center_of_mass_momentum(self):
        state, _, _ = create_harmonic_system(n_particles=6, kT=1.0, dt=0.02)
        drift = jnp.array([0.4, -0.2, 0.1])
        momenta = (
            state.particles.data.momenta + state.particles.data.masses[:, None] * drift
        )
        state = SimpleState.particles.focus(lambda p: p.data.momenta).set(
            state, momenta
        )
        step = StochasticStep(particles=SimpleState.particles, system=get_params)

        out = step(jax.random.key(7), state)

        total_momentum = out.particles.data.system.sum_over(
            out.particles.data.momenta
        ).data
        assert jnp.allclose(total_momentum, 0.0, atol=1e-12)

    def test_stochastic_step_gamma_zero_is_deterministic_noop(self):
        state, _, _ = create_harmonic_system(n_particles=6, kT=1.0, dt=0.02, gamma=0.0)
        step = StochasticStep(particles=SimpleState.particles, system=get_params)

        out = step(jax.random.key(7), state)

        assert jnp.allclose(out.particles.data.momenta, state.particles.data.momenta)

    def test_csvr_velocity_rescaling(self):
        n_particles, kT_target = 10, 2.0
        state, _, _ = create_harmonic_system(
            n_particles=n_particles, kT=kT_target, tau=0.1, dt=0.02
        )
        step = CSVRStep(particles=SimpleState.particles, systems=get_params)

        _, temps = run_simulation(
            step,
            state,
            jax.random.key(123),
            n_equil=0,
            n_sample=100,
            extract_fn=lambda s: compute_temperature(s, 3 * n_particles - 3),
        )
        assert_temperature(jnp.mean(temps), kT_target, 0.15)


class TestBarostatAndMICSteps:
    """Tests for StochasticCellRescalingStep and WrapFlow."""

    def test_cell_volume_updates(self):
        state = create_npt_system(n_particles=3, box_size=5.0)
        step = StochasticCellRescalingStep(
            particles=NPTState.particles, systems=NPTState.systems
        )

        initial_volume = state.systems.data.cell.volume
        new_state = step(jax.random.key(42), state)

        assert not jnp.isclose(
            new_state.systems.data.cell.volume, initial_volume, rtol=1e-8
        ), "CRITICAL BUG: Cell volume did not update"
        expected_volume = jnp.linalg.det(new_state.systems.data.cell.vectors)
        assert jnp.isclose(
            new_state.systems.data.cell.volume, expected_volume, rtol=1e-6
        )

    def test_positions_scale_with_box(self):
        state = create_npt_system(n_particles=3, box_size=5.0)
        step = StochasticCellRescalingStep(
            particles=NPTState.particles, systems=NPTState.systems
        )

        initial_pos = state.particles.data.positions
        initial_box = jnp.mean(jnp.diag(state.systems.data.cell.vectors[0]))
        new_state = step(jax.random.key(42), state)
        new_box = jnp.mean(jnp.diag(new_state.systems.data.cell.vectors[0]))

        expected_pos = initial_pos * (new_box / initial_box)
        assert jnp.allclose(new_state.particles.data.positions, expected_pos, rtol=1e-3)

    def test_pressure_response(self):
        state = create_npt_system(
            n_particles=3, box_size=2.0, tau_p=0.1, compressibility=10.0
        )
        step = StochasticCellRescalingStep(
            particles=NPTState.particles, systems=NPTState.systems
        )

        initial_volume = state.systems.data.cell.volume
        _, volumes = run_simulation(
            step,
            state,
            jax.random.key(42),
            n_equil=0,
            n_sample=10,
            extract_fn=lambda s: s.systems.data.cell.volume,
        )

        assert jnp.mean(volumes[5:]) > initial_volume * 1.01, (
            "Barostat did not expand box in response to high pressure"
        )

    def test_wrapping_positions(self):
        box_size = 5.0
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * box_size))

        @dataclass
        class TestState:
            cell: Cell

        flow = WrapFlow(cell=lambda s: s.cell, flow=euclidean_flow)
        new_pos = flow(
            TestState(cell=cell),
            jnp.array([0.1]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([100.0, 0.0, 0.0]),
        )

        assert jnp.all(new_pos >= 0.0) and jnp.all(new_pos < box_size)


# ============================================================================
# Tests: Physics (NVT integrators merged to share JIT cache)
# ============================================================================


class TestNVTPhysics:
    """Tests for VelocityVerlet, CSVR, and BAOAB physics."""

    def test_vv_energy_conservation(self):
        k = 1.0
        state, deriv, _ = create_harmonic_system(n_particles=5, k=k, dt=0.001)
        integrator = make_velocity_verlet_step(
            particles=SimpleState.particles,
            systems=SimpleState.systems.get,
            derivative_computation=deriv,
            flow=euclidean_flow,
        )

        def total_energy(s):
            ke = jnp.sum(
                particle_kinetic_energy(
                    s.particles.data.momenta, s.particles.data.masses
                )
            )
            pe = 0.5 * k * jnp.sum(s.particles.data.positions**2)
            return ke + pe

        initial_energy = total_energy(state)
        _, energies = run_simulation(
            integrator, state, jax.random.key(0), 0, 100, total_energy
        )

        max_drift = jnp.max(
            jnp.abs(energies - initial_energy) / jnp.abs(initial_energy)
        )
        assert max_drift < 1e-4, (
            f"Energy conservation violated: max drift = {max_drift:.2e}"
        )

    def test_vv_time_reversibility(self):
        state, deriv, _ = create_harmonic_system(n_particles=5, k=1.0, dt=0.01)
        integrator = make_velocity_verlet_step(
            particles=SimpleState.particles,
            systems=SimpleState.systems.get,
            derivative_computation=deriv,
            flow=euclidean_flow,
        )

        initial_pos = state.particles.data.positions.copy()
        initial_mom = state.particles.data.momenta.copy()

        def step_fn(carry, _):
            key, s = carry
            key, subkey = jax.random.split(key)
            return (key, integrator(subkey, s)), None

        @jit
        def run_forward_backward(key, state):
            (key, state), _ = jax.lax.scan(step_fn, (key, state), None, length=50)
            # Reverse momenta
            new_particles = Table(
                state.particles.keys,
                ParticleData(
                    positions=state.particles.data.positions,
                    momenta=-state.particles.data.momenta,
                    forces=state.particles.data.forces,
                    masses=state.particles.data.masses,
                    system=state.particles.data.system,
                    position_gradients=state.particles.data.position_gradients,
                ),
            )
            state = SimpleState(particles=new_particles, systems=state.systems)
            (_, state), _ = jax.lax.scan(step_fn, (key, state), None, length=50)
            return state

        state = run_forward_backward(jax.random.key(0), state)
        assert jnp.allclose(state.particles.data.positions, initial_pos, rtol=1e-4)
        assert jnp.allclose(state.particles.data.momenta, -initial_mom, rtol=1e-4)

    def test_csvr_temperature_and_equipartition(self):
        """Merged: temperature convergence and equipartition theorem."""
        # Temperature convergence (kT=1.5)
        n_particles, kT_target = 10, 1.5
        state, deriv, _ = create_harmonic_system(
            n_particles=n_particles, kT=kT_target, dt=0.01, tau=0.1
        )
        integrator = make_csvr_step(
            particles=SimpleState.particles,
            systems=SimpleState.systems.get,
            derivative_computation=deriv,
            flow=euclidean_flow,
        )

        _, temps = run_simulation(
            integrator,
            state,
            jax.random.key(42),
            n_equil=150,
            n_sample=200,
            extract_fn=lambda s: compute_temperature(s, 3 * n_particles - 3),
        )
        assert_temperature(jnp.mean(temps), kT_target, 0.15)

        # Equipartition theorem (kT=1.0)
        n_particles2, kT_target2 = 10, 1.0
        state2, deriv2, _ = create_harmonic_system(
            n_particles=n_particles2, kT=kT_target2, dt=0.01, tau=0.1
        )
        integrator2 = make_csvr_step(
            particles=SimpleState.particles,
            systems=SimpleState.systems.get,
            derivative_computation=deriv2,
            flow=euclidean_flow,
        )

        def extract_ke_xyz(s):
            p, m = s.particles.data.momenta, s.particles.data.masses
            return jnp.array([jnp.sum(0.5 * p[:, i] ** 2 / m) for i in range(3)])

        _, ke_samples = run_simulation(
            integrator2, state2, jax.random.key(42), 150, 200, extract_ke_xyz
        )
        expected = n_particles2 * kT_target2 / 2

        for i, dim in enumerate("xyz"):
            rel_err = jnp.abs(jnp.mean(ke_samples[:, i]) - expected) / expected
            assert rel_err < 0.30, f"Equipartition violated in {dim}"

    def test_baoab_temperature_control(self):
        n_particles, kT_target = 10, 1.2
        state, derivative_computation, _ = create_harmonic_system(
            n_particles=n_particles, kT=kT_target, dt=0.01, gamma=1.0
        )
        integrator = make_baoab_langevin_step(
            particles=SimpleState.particles,
            systems=SimpleState.systems.get,
            derivative_computation=derivative_computation,
            flow=euclidean_flow,
        )

        _, temps = run_simulation(
            integrator,
            state,
            jax.random.key(456),
            n_equil=150,
            n_sample=150,
            extract_fn=lambda s: compute_temperature(s, 3 * n_particles - 3),
        )
        assert_temperature(jnp.mean(temps), kT_target, 0.15, "BAOAB ")


class TestCSVRNPTPhysics:
    _integrator = None
    _deriv = None

    @classmethod
    def _get_integrator(cls):
        if cls._integrator is None:
            cls._deriv = create_npt_derivative_computation()
            cls._integrator = make_csvr_npt_step(
                particles=NPTState.particles,
                systems=NPTState.systems,
                derivative_computation=cls._deriv,
                flow=euclidean_flow,
            )
        return cls._integrator

    def test_temperature_control_with_barostat(self):
        n_particles, kT_target = 5, 1.2
        state = create_npt_system(
            n_particles=n_particles, box_size=5.0, kT=kT_target, tau_t=0.1, tau_p=2.0
        )
        integrator = self._get_integrator()

        _, temps = run_simulation(
            integrator,
            state,
            jax.random.key(789),
            n_equil=150,
            n_sample=150,
            extract_fn=lambda s: compute_temperature(s, 3 * n_particles - 3),
        )
        assert_temperature(jnp.mean(temps), kT_target, 0.15, "NPT ")

    def test_volume_fluctuations(self):
        state = create_npt_system(
            n_particles=5, box_size=5.0, kT=1.0, tau_t=0.1, tau_p=1.0
        )
        integrator = self._get_integrator()

        _, volumes = run_simulation(
            integrator,
            state,
            jax.random.key(123),
            n_equil=50,
            n_sample=50,
            extract_fn=lambda s: s.systems.data.cell.volume,
        )

        mean_vol, std_vol = jnp.mean(volumes), jnp.std(volumes)
        assert std_vol > 0.01 * mean_vol, "Volume fluctuations too small"
        assert std_vol / mean_vol < 1.0, "Volume fluctuations too large"

    def test_gradients_update(self):
        state = create_npt_system(
            n_particles=5, box_size=5.0, kT=1.0, tau_t=0.1, tau_p=1.0
        )
        integrator = self._get_integrator()

        initial_grads = state.particles.data.position_gradients.copy()
        final_state, _ = run_simulation(
            integrator,
            state,
            jax.random.key(42),
            n_equil=0,
            n_sample=10,
            extract_fn=lambda s: s.particles.data.position_gradients,
        )

        assert not jnp.allclose(
            initial_grads, final_state.particles.data.position_gradients, rtol=1e-6
        )


# ============================================================================
# Tests: Utilities
# ============================================================================


def test_particle_kinetic_energy():
    """KE = p^2/(2m) for known momentum vectors."""
    # Single particle, unit mass, p = (1, 0, 0) => KE = 0.5
    ke = particle_kinetic_energy(jnp.array([[1.0, 0.0, 0.0]]), jnp.array([1.0]))
    assert jnp.isclose(ke[0], 0.5)

    # Multi-dimensional: p = (3, 4, 0), m = 2 => KE = 25/(2*2) = 6.25
    ke2 = particle_kinetic_energy(jnp.array([[3.0, 4.0, 0.0]]), jnp.array([2.0]))
    assert jnp.isclose(ke2[0], 6.25)


def test_instantaneous_pressure():
    """P = 2K/(3V) + Tr(σ)/3 for known Cauchy stress σ."""
    ke, sigma, vol = jnp.array([10.0]), jnp.eye(3)[None] * 5.0, jnp.array([125.0])
    pressure = instantaneous_pressure(ke, sigma, vol)
    expected = (2.0 * ke) / (3.0 * vol) + 15.0 / 3.0
    assert jnp.isclose(pressure, expected, rtol=1e-5).all()


def test_npt_cauchy_stress_convention():
    """Barostat pressure must use Cauchy stress σ (pressure units), not virial W.

    The StochasticCellRescalingStep computes P = 2K/(3V) + Tr(σ)/3.
    If stress_tensor were the virial W (energy), the barostat would divide
    by V twice, making the configurational pressure ~V times too small.

    This test verifies the convention on a known system: a 5-particle harmonic
    oscillator at kT=1 in a 5A box. The Cauchy stress σ = W/V, so
    Tr(σ)/3 = Tr(W)/(3V) must give the correct configurational pressure.
    """
    n_particles, box_size, k = 5, 5.0, 1.0
    positions = jnp.array(
        [
            [1.0, 0.5, -0.3],
            [-0.5, 1.2, 0.1],
            [0.3, -0.7, 0.8],
            [-0.2, 0.4, -0.6],
            [0.6, -0.1, 0.2],
        ]
    )
    forces = -k * positions
    momenta = jnp.ones_like(positions) * 0.5
    masses = jnp.ones(n_particles)

    # Virial W = Σ r⊗F (energy units)
    virial = _virial_stress(positions, forces)
    V = box_size**3

    # Cauchy stress σ = W/V (pressure units) — what stress_tensor should store
    cauchy_stress = virial / V

    # Correct pressure: P = 2K/(3V) + Tr(W)/(3V) = 2K/(3V) + Tr(σ)/3
    ke_total = 0.5 * jnp.sum(momenta**2 / masses[:, None])
    P_expected = 2 * ke_total / (3 * V) + jnp.trace(virial) / (3 * V)

    # What the fixed barostat computes: 2K/(3V) + Tr(σ)/3
    P_fixed = 2 * ke_total / (3 * V) + jnp.trace(cauchy_stress) / 3

    # What the old buggy code computed: 2K/(3V) + Tr(σ)/(3V)  ← extra /V
    P_buggy = 2 * ke_total / (3 * V) + jnp.trace(cauchy_stress) / (3 * V)

    assert jnp.isclose(P_fixed, P_expected, rtol=1e-10), (
        f"Fixed pressure {P_fixed} != expected {P_expected}"
    )
    assert not jnp.isclose(P_buggy, P_expected, rtol=0.1), (
        f"Buggy pressure should NOT match expected (off by factor V={V})"
    )


def test_stress_matches_ase():
    """Full virial stress must match ASE's stress on the CI argon system.

    Loads the 256-atom FCC argon CIF, evaluates forces with both ASE's LJ
    calculator and LJ potential, and compares the Cauchy stress tensor.
    This catches regressions in the virial computation that would break NPT.

    Verified to 5 significant figures (ratio = -1.0000, sign flip is the
    ASE convention σ_ASE = -σ_kUPS).
    """
    from pathlib import Path

    import ase.io
    import numpy as np
    from ase import Atoms
    from ase.calculators.lj import LennardJones as ASELJ

    from kups.application.md.data import MdParameters, md_state_from_ase
    from kups.core.lens import identity_lens
    from kups.core.neighborlist import (
        DenseNearestNeighborList,
        NeighborList,
        UniversalNeighborlistParameters,
    )
    from kups.observables.stress import stress_via_virial_theorem
    from kups.potential.classical.lennard_jones import (
        LennardJonesParameters,
        make_lennard_jones_from_state,
    )

    cif = (
        Path(__file__).parent.parent.parent
        / "ci"
        / "statistical"
        / "inputs"
        / "host"
        / "argon_fcc.cif"
    )
    sigma, eps = 3.405, 0.01032356174398622

    # ASE stress
    atoms = cast(Atoms, ase.io.read(str(cif)))
    atoms.calc = ASELJ(epsilon=eps, sigma=sigma, rc=10.0, smooth=False)
    ase_stress = atoms.get_stress(voigt=False)
    ase_pressure = -np.trace(ase_stress) / 3

    # kUPS: evaluate potential directly (no propagator/propagate_and_fix)
    @dataclass
    class S:
        particles: Table[ParticleId, Any]
        systems: Table[SystemId, Any]
        neighborlist_params: UniversalNeighborlistParameters
        step: jnp.ndarray
        lj_parameters: LennardJonesParameters

        @property
        def neighborlist(self) -> NeighborList[Literal[2]]:
            return DenseNearestNeighborList.from_state(self)

    lj = LennardJonesParameters.from_dict(
        cutoff=10.0, parameters={"Ar": (sigma, eps)}, mixing_rule="lorentz_berthelot"
    )
    config = MdParameters(
        temperature=100.0,
        time_step=2.0,
        friction_coefficient=1.0,
        thermostat_time_constant=100.0,
        target_pressure=1.0,
        pressure_coupling_time=1e10,
        compressibility=4.5e-5,
        minimum_scale_factor=1.0,
        integrator="baoab_langevin",
        initialize_momenta=False,
    )
    p, s = md_state_from_ase(str(cif), config)
    nl = UniversalNeighborlistParameters.estimate(p.data.system.counts, s, lj.cutoff)
    state = S(
        particles=p,
        systems=s,
        neighborlist_params=nl,
        step=jnp.array([0]),
        lj_parameters=lj,
    )
    sl = identity_lens(S)
    pot = make_lennard_jones_from_state(sl, compute_position_and_cell_gradients=True)

    # Evaluate potential and write gradients back into state
    result = pot(state)
    pos_grad = result.data.gradients.positions.data
    cell_grad = result.data.gradients.cell.data

    import dataclasses

    p_with_grad = p.set_data(dataclasses.replace(p.data, position_gradients=pos_grad))
    s_with_grad = s.set_data(dataclasses.replace(s.data, cell_gradients=cell_grad))

    kups_stress = np.asarray(
        stress_via_virial_theorem(p_with_grad, s_with_grad).data[0]
    )
    kups_pressure = np.trace(kups_stress) / 3

    np.testing.assert_allclose(
        kups_pressure,
        ase_pressure,
        rtol=1e-3,
        err_msg="kUPS virial stress diverged from ASE reference",
    )

    # NOTE: The NPT density comparison test (test_npt_density_matches_ase) lives
    # in the physical_validation PR where it uses propagate_and_fix. Keeping it
    # here would fail on CI JAX versions with the ShapedArray.vma issue.


# ============================================================================
# Tests: Gao-Fang-Wang BAOAB NPT Langevin
# ============================================================================


@dataclass
class BNPTLParams:
    """Bundled integrator-params for the BAOAB NPT Langevin tests."""

    time_step: Array
    temperature: Array
    friction_coefficient: Array
    target_pressure: Array
    pressure_coupling_time: Array
    compressibility: Array
    barostat_mass: Array
    barostat_friction: Array


@dataclass
class BNPTLSystemData:
    cell: Cell
    cell_gradients: Cell
    cell_momentum: Array
    integrator_params: BNPTLParams


@dataclass
class BNPTLState(HasLensFields):
    particles: LensField[Table[ParticleId, ParticleData]]
    systems: LensField[Table[SystemId, BNPTLSystemData]]


def create_bnptl_system(
    n_particles: int = 5,
    box_size: float = 5.0,
    kT: float = 1.0,
    target_pressure: float = 0.0,
    dt: float = 0.005,
    tau_p: float = 1.0,
    gamma: float = 1.0,
    compressibility: float = 10.0,
    key: Array | None = None,
) -> BNPTLState:
    """Harmonic NPT-Langevin system: cubic cell, isotropic harmonic forces."""
    if key is None:
        key = jax.random.key(2026)
    k1, k2 = jax.random.split(key)
    from kups.core.utils.jax import tree_zeros_like

    positions = (jax.random.uniform(k1, (n_particles, 3)) - 0.5) * box_size * 0.6
    momenta = jax.random.normal(k2, (n_particles, 3)) * jnp.sqrt(kT)
    forces = -positions
    masses = jnp.ones(n_particles)
    system_index = Index.new([SystemId(0)] * n_particles)
    particles = Table.arange(
        ParticleData(
            positions=positions,
            momenta=momenta,
            forces=forces,
            masses=masses,
            system=system_index,
            position_gradients=-forces,
        ),
        label=ParticleId,
    )
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * box_size))
    # Gao Eq. (14)/(15) in kUPS' transposed lower-triangular convention:
    # the paper row index becomes a kUPS column index.
    V0 = jnp.array([box_size**3])
    h_diag = jnp.full((1, 3), box_size)
    tau_p_arr = jnp.array([tau_p])
    kappa = jnp.array([compressibility])
    per_column = (
        3.0
        * V0[..., None]
        * (tau_p_arr[..., None] / (2.0 * jnp.pi)) ** 2
        / (kappa[..., None] * h_diag**2)
    )
    M = jnp.tril(jnp.broadcast_to(per_column[..., None, :], (1, 3, 3)))
    gamma_arr = jnp.array([gamma])
    gamma_t = jnp.tril(jnp.broadcast_to(gamma_arr[..., None, None], M.shape))
    params = BNPTLParams(
        time_step=jnp.array([dt]),
        temperature=jnp.array([kT / BOLTZMANN_CONSTANT]),
        friction_coefficient=gamma_arr,
        target_pressure=jnp.array([target_pressure]),
        pressure_coupling_time=tau_p_arr,
        compressibility=kappa,
        barostat_mass=M,
        barostat_friction=gamma_t,
    )
    systems = Table.arange(
        BNPTLSystemData(
            cell=cell,
            cell_gradients=tree_zeros_like(cell),
            cell_momentum=jnp.zeros((1, 3, 3)),
            integrator_params=params,
        ),
        label=SystemId,
    )
    return BNPTLState(particles=particles, systems=systems)


def create_bnptl_derivative_computation():
    """Harmonic forces F = -r, mirrors create_npt_derivative_computation."""

    @dataclass
    class DerivativeComputation(Propagator[BNPTLState]):
        def __call__(self, key, state: BNPTLState) -> BNPTLState:
            del key
            forces = -1.0 * state.particles.data.positions
            new_particles = Table(
                state.particles.keys,
                ParticleData(
                    positions=state.particles.data.positions,
                    momenta=state.particles.data.momenta,
                    forces=forces,
                    masses=state.particles.data.masses,
                    system=state.particles.data.system,
                    position_gradients=-forces,
                ),
                _cls=state.particles._cls,
            )
            return BNPTLState(particles=new_particles, systems=state.systems)

    return DerivativeComputation()


def test_gao_barostat_mass_uses_kups_transposed_column_index():
    """Gao's upper-triangular row index maps to a kUPS lower-triangular column."""
    from kups.application.md.data import _gao_barostat_mass

    cell_vectors = jnp.array([[[2.0, 0.0, 0.0], [0.4, 3.0, 0.0], [0.5, 0.7, 5.0]]])
    compressibility = jnp.array([7.0])
    tau_p = jnp.array([11.0])

    mass = _gao_barostat_mass(cell_vectors, compressibility, tau_p)

    volume = jnp.abs(jnp.linalg.det(cell_vectors))[0]
    diag = jnp.diagonal(cell_vectors[0])
    per_paper_row = (
        3.0 * volume * (tau_p[0] / (2.0 * jnp.pi)) ** 2 / (compressibility[0] * diag**2)
    )
    expected = jnp.tril(jnp.broadcast_to(per_paper_row[None, None, :], (1, 3, 3)))
    row_indexed = jnp.tril(jnp.broadcast_to(per_paper_row[None, :, None], (1, 3, 3)))

    assert jnp.allclose(mass, expected)
    assert not jnp.allclose(mass, row_indexed)


def _row_affine_reference(
    x0: Array, row_matrix: Array, bias: Array, dt: Array | float
) -> Array:
    """Independent row-vector solution for ``dx/dt = x A + b``."""
    dim = x0.shape[-1]
    augmented = jnp.zeros((dim + 1, dim + 1), dtype=x0.dtype)
    augmented = augmented.at[:dim, :dim].set(row_matrix)
    augmented = augmented.at[dim, :dim].set(bias)
    z0 = jnp.concatenate([x0, jnp.ones((1,), dtype=x0.dtype)])
    return (z0 @ jax.scipy.linalg.expm(augmented * dt))[:dim]


def _lower_to_symmetric(lower: Array) -> Array:
    diag = jnp.diagonal(lower, axis1=-2, axis2=-1)
    return lower + lower.T - diag[:, None] * jnp.eye(3, dtype=lower.dtype)


def _manual_pressure_tensor(
    positions: Array,
    momenta: Array,
    masses: Array,
    position_gradients: Array,
    cell_gradient_vectors: Array,
    V: Array,
) -> Array:
    volume = jnp.abs(jnp.linalg.det(V))
    kinetic = jnp.sum(
        momenta[:, :, None] * momenta[:, None, :] / masses[:, None, None], axis=0
    )
    pos_outer = jnp.sum(position_gradients[:, None] * positions[..., None], axis=0)
    sym_pos = 0.5 * (pos_outer + pos_outer.T)
    cell_lower = jnp.tril(V.T @ cell_gradient_vectors)
    sigma = -(_lower_to_symmetric(jnp.tril(sym_pos) + cell_lower)) / volume
    return kinetic / volume + sigma


def _manual_cell_velocity(cell_momentum: Array, barostat_mass: Array) -> Array:
    safe_mass = jnp.where(barostat_mass != 0, barostat_mass, 1.0)
    return jnp.tril(jnp.where(barostat_mass != 0, cell_momentum / safe_mass, 0.0))


def _manual_cell_momentum_kick(
    cell_momentum: Array,
    positions: Array,
    momenta: Array,
    masses: Array,
    position_gradients: Array,
    cell_gradient_vectors: Array,
    V: Array,
    dt_half: float,
    target_pressure: float,
    kT: float,
) -> Array:
    volume = jnp.abs(jnp.linalg.det(V))
    V_inv = jnp.linalg.inv(V)
    pressure_tensor = _manual_pressure_tensor(
        positions, momenta, masses, position_gradients, cell_gradient_vectors, V
    )
    pressure_diff = pressure_tensor - target_pressure * jnp.eye(3, dtype=V.dtype)
    kick_paper = volume * (pressure_diff @ V_inv) - kT * V_inv
    return cell_momentum + dt_half * jnp.tril(kick_paper.T)


def _manual_baoab_no_thermostat_step(
    positions: Array,
    momenta: Array,
    masses: Array,
    V: Array,
    cell_momentum: Array,
    barostat_mass: Array,
    dt: float,
    target_pressure: float,
    kT: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    dt_half = dt / 2
    cell_gradient_vectors = jnp.zeros_like(V)
    forces = -positions
    position_gradients = positions

    cell_momentum = _manual_cell_momentum_kick(
        cell_momentum,
        positions,
        momenta,
        masses,
        position_gradients,
        cell_gradient_vectors,
        V,
        dt_half,
        target_pressure,
        kT,
    )

    V_dot = _manual_cell_velocity(cell_momentum, barostat_mass)
    M_pos = jnp.linalg.inv(V) @ V_dot
    momenta = jax.vmap(lambda p, f: _row_affine_reference(p, -M_pos.T, f, dt_half))(
        momenta, forces
    )

    V = V + dt_half * V_dot
    M_pos = jnp.linalg.inv(V) @ V_dot
    positions = jax.vmap(
        lambda r, p, m: _row_affine_reference(r, M_pos, p / m, dt_half)
    )(positions, momenta, masses)

    positions = jax.vmap(
        lambda r, p, m: _row_affine_reference(r, M_pos, p / m, dt_half)
    )(positions, momenta, masses)
    V = V + dt_half * V_dot

    forces = -positions
    position_gradients = positions
    M_pos = jnp.linalg.inv(V) @ V_dot
    momenta = jax.vmap(lambda p, f: _row_affine_reference(p, -M_pos.T, f, dt_half))(
        momenta, forces
    )

    cell_momentum = _manual_cell_momentum_kick(
        cell_momentum,
        positions,
        momenta,
        masses,
        position_gradients,
        cell_gradient_vectors,
        V,
        dt_half,
        target_pressure,
        kT,
    )
    return positions, momenta, V, cell_momentum, forces, position_gradients


class TestBNPTLPropagators:
    """Per-propagator unit tests for the new NPT Langevin building blocks."""

    def test_cell_momentum_kick_increases_with_high_pressure(self):
        state = create_bnptl_system(box_size=2.0, tau_p=0.1, compressibility=10.0)
        kick = CellMomentumKick(
            particles=BNPTLState.particles, systems=BNPTLState.systems
        )
        new_state = kick(jax.random.key(0), state)
        # System is small + harmonic → high internal pressure → diagonal p^h increases.
        delta = new_state.systems.data.cell_momentum - state.systems.data.cell_momentum
        diag = jnp.diagonal(delta[0], axis1=-2, axis2=-1)
        assert jnp.all(diag > 0), f"diagonal p^h kick should be positive: {diag}"
        # Strict-upper-triangular part stays zero.
        upper = delta[0] - jnp.tril(delta[0])
        assert jnp.allclose(upper, 0.0, atol=1e-12)

    def test_cell_momentum_kick_matches_gao_transposed_formula(self):
        """B^h kick agrees with Gao Eq. 8d under kUPS' V = h.T convention."""
        dt = 0.04
        kT = 2.5
        target_pressure = 0.7
        state = create_bnptl_system(
            n_particles=2,
            box_size=5.0,
            kT=kT,
            target_pressure=target_pressure,
            dt=dt,
            gamma=0.0,
        )
        V = jnp.array([[[2.0, 0.0, 0.0], [0.4, 3.0, 0.0], [0.3, 0.8, 4.0]]])
        positions = jnp.array([[0.2, -0.3, 0.4], [-0.5, 0.1, -0.2]])
        momenta = jnp.array([[1.1, -0.7, 0.3], [0.4, 0.9, -1.2]])
        masses = jnp.array([1.5, 2.0])
        initial_cm = jnp.array([[[0.2, 0.0, 0.0], [0.1, -0.3, 0.0], [0.4, 0.2, 0.5]]])

        state = BNPTLState.systems.focus(lambda s: s.data.cell.frame).set(
            state, TriclinicFrame.from_matrix(V)
        )
        state = BNPTLState.systems.focus(lambda s: s.data.cell_momentum).set(
            state, initial_cm
        )
        state = BNPTLState.particles.focus(lambda p: p.data.positions).set(
            state, positions
        )
        state = BNPTLState.particles.focus(lambda p: p.data.momenta).set(state, momenta)
        state = BNPTLState.particles.focus(lambda p: p.data.masses).set(state, masses)
        state = BNPTLState.particles.focus(lambda p: p.data.position_gradients).set(
            state, jnp.zeros_like(positions)
        )

        kick = CellMomentumKick(
            particles=BNPTLState.particles, systems=BNPTLState.systems
        )
        out = kick(jax.random.key(0), state)

        volume = jnp.abs(jnp.linalg.det(V[0]))
        V_inv = jnp.linalg.inv(V[0])
        kinetic_tensor = jnp.sum(
            momenta[:, :, None] * momenta[:, None, :] / masses[:, None, None],
            axis=0,
        )
        pressure_tensor = kinetic_tensor / volume
        pressure_diff = pressure_tensor - target_pressure * jnp.eye(3)
        kick_paper = volume * (pressure_diff @ V_inv) - kT * V_inv
        expected = initial_cm + (dt / 2) * jnp.tril(kick_paper.T)[None]

        assert jnp.allclose(
            out.systems.data.cell_momentum,
            expected,
            rtol=2e-6,
            atol=2e-6,
        )

    def test_cell_position_step_drifts_cell_only(self):
        """Paper line 4 (F^h_K): drifts only the cell matrix.

        Particle positions are *not* modified; the convective response of r
        to cell motion (Eq. 8a's ``ḣh⁻¹·r`` term) lives in
        :class:`CoupledPositionStep`. Combining both updates in this step
        would double-count the convective drift and break the Trotter
        splitting (verified by the Hamiltonian-conservation test).
        """
        state = create_bnptl_system(box_size=5.0)
        cm = jnp.tril(jnp.ones((1, 3, 3))) * 0.5
        state = BNPTLState.systems.focus(lambda s: s.data.cell_momentum).set(state, cm)
        initial_pos = state.particles.data.positions

        step = CellPositionStep(systems=BNPTLState.systems)
        new_state = step(jax.random.key(0), state)

        # Cell vectors changed by exactly the Gao A^h half-drift.
        new_V = new_state.systems.data.cell.vectors
        old_V = state.systems.data.cell.vectors
        M = state.systems.data.integrator_params.barostat_mass
        V_dot = jnp.tril(jnp.where(M != 0, cm / jnp.where(M != 0, M, 1.0), 0.0))
        expected_V = (
            old_V
            + state.systems.data.integrator_params.time_step[..., None, None]
            / 2
            * V_dot
        )
        assert jnp.allclose(new_V, expected_V, rtol=1e-6)
        assert not jnp.allclose(new_V, old_V)
        # Cell still lower-triangular.
        upper = new_V - jnp.tril(new_V)
        assert jnp.allclose(upper, 0.0, atol=1e-12)
        # Volume cache refreshed.
        expected_vol = jnp.abs(jnp.linalg.det(new_V))
        assert jnp.allclose(new_state.systems.data.cell.volume, expected_vol, rtol=1e-6)
        # Positions are NOT modified.
        assert jnp.array_equal(new_state.particles.data.positions, initial_pos), (
            "CellPositionStep must not touch particle positions; the convective "
            "drift belongs in CoupledPositionStep."
        )

    def test_cell_stochastic_step_equipartitions(self):
        """⟨p^h_αβ²⟩ → M_αβ k_B T in the long-run limit (per-DOF equipartition)."""
        state = create_bnptl_system(
            box_size=5.0,
            kT=1.0,
            gamma=10.0,
            tau_p=1.0,
            compressibility=10.0,
            dt=0.01,
        )
        step = CellStochasticStep(systems=BNPTLState.systems)
        kT = (state.systems.data.integrator_params.temperature * BOLTZMANN_CONSTANT)[0]
        # Step many times to sample the OU stationary distribution. With γ·dt = 0.1
        # the relaxation time is ~5 steps, so 20000 samples is plenty.
        n_steps = 20000

        @jit
        def run(key, state):
            def body(carry, _):
                key, s = carry
                key, sub = jax.random.split(key)
                s = step(sub, s)
                return (key, s), s.systems.data.cell_momentum[0]

            _, traj = jax.lax.scan(body, (key, state), None, length=n_steps)
            return traj

        traj = run(jax.random.key(123), state)
        M = state.systems.data.integrator_params.barostat_mass[0]
        # Use only the lower triangle (the active DOFs)
        mask = jnp.tril(jnp.ones((3, 3))) > 0
        # ⟨p_αβ²⟩ / M_αβ should be ≈ k_B T for active DOFs.
        var = jnp.mean(traj[2000:] ** 2, axis=0)  # skip burn-in
        ratio = (var / M)[mask] / kT
        # 20k samples, autocorr ~5 steps ⇒ ~4k effective samples per DOF; expected
        # std error on variance ≈ √(2/4000) ≈ 2.2 %. Tol 15 % covers ~7σ.
        assert jnp.all(jnp.abs(ratio - 1.0) < 0.15), f"equipartition ratio: {ratio}"

    def test_coupled_momentum_step_reduces_to_momentumstep_when_p_h_zero(self):
        """When cell_momentum = 0, the coupling vanishes and CoupledMomentumStep
        must equal the plain MomentumStep at Δt/2 (the propagator's hardcoded
        BAOAB half-step)."""
        state = create_bnptl_system(box_size=5.0)
        assert jnp.allclose(state.systems.data.cell_momentum, 0.0)
        coupled = CoupledMomentumStep(
            particles=BNPTLState.particles, systems=BNPTLState.systems
        )
        out_coupled = coupled(jax.random.key(0), state)
        dt_half = state.systems.data.integrator_params.time_step[0] / 2
        expected = state.particles.data.momenta + state.particles.data.forces * dt_half
        assert jnp.allclose(out_coupled.particles.data.momenta, expected, atol=1e-8)

    def test_coupled_position_step_reduces_to_positionstep_when_p_h_zero(self):
        state = create_bnptl_system(box_size=5.0)
        coupled = CoupledPositionStep(
            particles=BNPTLState.particles, systems=BNPTLState.systems
        )
        out = coupled(jax.random.key(0), state)
        v = state.particles.data.momenta / state.particles.data.masses[..., None]
        dt_half = state.systems.data.integrator_params.time_step[0] / 2
        expected = state.particles.data.positions + v * dt_half
        assert jnp.allclose(out.particles.data.positions, expected, atol=1e-8)

    def test_coupled_steps_match_independent_row_affine_reference(self):
        """A and B coupled substeps match direct row-vector matrix exponentials."""
        dt = 0.07
        state = create_bnptl_system(
            n_particles=2,
            box_size=5.0,
            kT=1.0,
            dt=dt,
            gamma=0.0,
        )
        V = jnp.array([[[2.4, 0.0, 0.0], [0.3, 2.7, 0.0], [0.2, 0.5, 3.1]]])
        barostat_mass = jnp.array([[[3.0, 0.0, 0.0], [4.0, 5.0, 0.0], [6.0, 7.0, 8.0]]])
        cell_momentum = jnp.array(
            [[[0.6, 0.0, 0.0], [0.2, -0.5, 0.0], [0.4, 0.3, 0.7]]]
        )
        positions = jnp.array([[0.4, -0.2, 0.7], [-0.3, 0.5, -0.6]])
        momenta = jnp.array([[0.8, -1.1, 0.2], [1.3, 0.4, -0.9]])
        masses = jnp.array([1.2, 2.3])
        forces = jnp.array([[-0.4, 0.6, -0.1], [0.2, -0.3, 0.5]])

        state = BNPTLState.systems.focus(lambda s: s.data.cell.frame).set(
            state, TriclinicFrame.from_matrix(V)
        )
        state = BNPTLState.systems.focus(
            lambda s: s.data.integrator_params.barostat_mass
        ).set(state, barostat_mass)
        state = BNPTLState.systems.focus(lambda s: s.data.cell_momentum).set(
            state, cell_momentum
        )
        state = BNPTLState.particles.focus(lambda p: p.data.positions).set(
            state, positions
        )
        state = BNPTLState.particles.focus(lambda p: p.data.momenta).set(state, momenta)
        state = BNPTLState.particles.focus(lambda p: p.data.masses).set(state, masses)
        state = BNPTLState.particles.focus(lambda p: p.data.forces).set(state, forces)

        V_dot = jnp.tril(cell_momentum / barostat_mass)
        M_pos = jnp.linalg.inv(V[0]) @ V_dot[0]
        dt_half = dt / 2
        expected_positions = jax.vmap(
            lambda r, p, m: _row_affine_reference(r, M_pos, p / m, dt_half)
        )(positions, momenta, masses)
        expected_momenta = jax.vmap(
            lambda p, f: _row_affine_reference(p, -M_pos.T, f, dt_half)
        )(momenta, forces)

        out_pos = CoupledPositionStep(
            particles=BNPTLState.particles, systems=BNPTLState.systems
        )(jax.random.key(0), state)
        out_mom = CoupledMomentumStep(
            particles=BNPTLState.particles, systems=BNPTLState.systems
        )(jax.random.key(0), state)

        assert jnp.allclose(
            out_pos.particles.data.positions,
            expected_positions,
            rtol=2e-6,
            atol=2e-6,
        )
        assert jnp.allclose(
            out_mom.particles.data.momenta,
            expected_momenta,
            rtol=2e-6,
            atol=2e-6,
        )

    def test_wrap_step_wraps_positions(self):
        state = create_bnptl_system(box_size=5.0)
        # Push positions far outside the box.
        far = state.particles.data.positions + 100.0
        state = BNPTLState.particles.focus(lambda p: p.data.positions).set(state, far)
        step = WrapStep(particles=BNPTLState.particles, systems=lambda s: s.systems)
        out = step(jax.random.key(0), state)
        # All wrapped into [-box/2, box/2)
        assert jnp.all(out.particles.data.positions >= -2.5 - 1e-6)
        assert jnp.all(out.particles.data.positions < 2.5 + 1e-6)


class TestBNPTLPressureTensor:
    """instantaneous_pressure_tensor sanity."""

    def test_pressure_tensor_symmetric(self):
        state = create_bnptl_system(box_size=4.0)
        P = instantaneous_pressure_tensor(state.particles, state.systems)
        assert jnp.allclose(P, jnp.swapaxes(P, -1, -2), atol=1e-10)

    def test_pressure_tensor_trace_matches_scalar_pressure(self):
        state = create_bnptl_system(box_size=4.0)
        P_tensor = instantaneous_pressure_tensor(state.particles, state.systems)
        per_particle_ke = particle_kinetic_energy(
            state.particles.data.momenta, state.particles.data.masses
        )
        ke = jax.ops.segment_sum(
            per_particle_ke,
            state.particles.data.system.indices,
            state.particles.data.system.num_labels,
        )
        # Use stress_via_virial_theorem like the existing pressure observable does.
        from kups.observables.stress import stress_via_virial_theorem

        sigma = stress_via_virial_theorem(state.particles, state.systems).data
        V = state.systems.data.cell.volume
        P_scalar = instantaneous_pressure(ke, sigma, V)
        # Tr(P_tensor)/3 should equal scalar P.
        np_tr = jnp.trace(P_tensor[0]) / 3.0
        assert jnp.allclose(np_tr, P_scalar[0], rtol=1e-5)


class TestBNPTLIntegrator:
    _integrator = None

    @classmethod
    def _make_integrator(cls):
        if cls._integrator is None:
            deriv = create_bnptl_derivative_computation()
            cls._integrator = make_baoab_npt_langevin_step(
                particles=BNPTLState.particles,
                systems=BNPTLState.systems,
                derivative_computation=deriv,
                flow=euclidean_flow,
            )
        return cls._integrator

    def test_wraps_before_force_recompute(self):
        """Forces/stress must be evaluated at the wrapped coordinates kept in state."""

        @dataclass
        class EchoPositionDerivative(Propagator[BNPTLState]):
            def __call__(self, key, state: BNPTLState) -> BNPTLState:
                del key
                positions = state.particles.data.positions
                new_particles = Table(
                    state.particles.keys,
                    ParticleData(
                        positions=positions,
                        momenta=state.particles.data.momenta,
                        forces=-positions,
                        masses=state.particles.data.masses,
                        system=state.particles.data.system,
                        position_gradients=positions,
                    ),
                    _cls=state.particles._cls,
                )
                return BNPTLState(particles=new_particles, systems=state.systems)

        state = create_bnptl_system(
            n_particles=3,
            box_size=4.0,
            dt=0.0,
            gamma=0.0,
        )
        outside = jnp.array([[2.4, 0.0, 0.0], [-2.5, 1.9, 0.0], [0.1, 2.6, -2.7]])
        zeros = jnp.zeros_like(outside)
        state = BNPTLState.particles.focus(lambda p: p.data.positions).set(
            state, outside
        )
        state = BNPTLState.particles.focus(lambda p: p.data.momenta).set(state, zeros)
        state = BNPTLState.particles.focus(lambda p: p.data.forces).set(state, zeros)
        state = BNPTLState.particles.focus(lambda p: p.data.position_gradients).set(
            state, zeros
        )

        integrator = make_baoab_npt_langevin_step(
            particles=BNPTLState.particles,
            systems=BNPTLState.systems,
            derivative_computation=EchoPositionDerivative(),
            flow=euclidean_flow,
        )
        out = integrator(jax.random.key(0), state)

        expected = state.systems.data.cell.wrap(outside)
        assert jnp.allclose(out.particles.data.positions, expected)
        assert jnp.allclose(out.particles.data.position_gradients, expected)

    def test_dt_zero_step_is_noop_for_current_inside_state(self):
        """At dt = 0, even with nonzero frictions, the full stochastic step is inert."""
        state = create_bnptl_system(
            n_particles=4,
            box_size=5.0,
            kT=1.0,
            dt=0.0,
            gamma=3.0,
        )
        cell_momentum = jnp.array(
            [[[0.5, 0.0, 0.0], [0.1, -0.2, 0.0], [0.3, 0.4, 0.6]]]
        )
        state = BNPTLState.systems.focus(lambda s: s.data.cell_momentum).set(
            state, cell_momentum
        )
        integrator = make_baoab_npt_langevin_step(
            particles=BNPTLState.particles,
            systems=BNPTLState.systems,
            derivative_computation=create_bnptl_derivative_computation(),
            flow=euclidean_flow,
        )
        out = integrator(jax.random.key(99), state)

        assert jnp.allclose(
            out.particles.data.positions, state.particles.data.positions
        )
        assert jnp.allclose(out.particles.data.momenta, state.particles.data.momenta)
        assert jnp.allclose(out.particles.data.forces, state.particles.data.forces)
        assert jnp.allclose(
            out.particles.data.position_gradients,
            state.particles.data.position_gradients,
        )
        assert jnp.allclose(
            out.systems.data.cell.vectors, state.systems.data.cell.vectors
        )
        assert jnp.allclose(
            out.systems.data.cell_momentum,
            state.systems.data.cell_momentum,
        )

    def test_full_deterministic_step_matches_independent_reference(self):
        """Whole BAOAB step matches a hand-written deterministic oracle."""
        dt = 0.003
        kT = 1.4
        target_pressure = 0.02
        state = create_bnptl_system(
            n_particles=3,
            box_size=6.0,
            kT=kT,
            target_pressure=target_pressure,
            dt=dt,
            gamma=0.0,
        )
        V = jnp.array([[5.8, 0.0, 0.0], [0.4, 6.1, 0.0], [-0.2, 0.3, 5.6]])
        barostat_mass = jnp.array([[4.0, 0.0, 0.0], [5.0, 6.0, 0.0], [7.0, 8.0, 9.0]])
        cell_momentum = jnp.array(
            [[0.12, 0.0, 0.0], [-0.05, 0.08, 0.0], [0.03, -0.04, 0.07]]
        )
        positions = jnp.array(
            [[0.25, -0.35, 0.18], [-0.42, 0.21, -0.27], [0.16, 0.31, 0.44]]
        )
        momenta = jnp.array([[0.7, -0.2, 0.3], [-0.4, 0.5, -0.6], [0.2, 0.8, -0.1]])
        masses = jnp.array([1.1, 1.7, 2.3])
        forces = -positions

        state = BNPTLState.systems.focus(lambda s: s.data.cell.frame).set(
            state, TriclinicFrame.from_matrix(V[None])
        )
        state = BNPTLState.systems.focus(lambda s: s.data.cell_momentum).set(
            state, cell_momentum[None]
        )
        state = BNPTLState.systems.focus(
            lambda s: s.data.integrator_params.barostat_mass
        ).set(state, barostat_mass[None])
        state = BNPTLState.particles.focus(lambda p: p.data.positions).set(
            state, positions
        )
        state = BNPTLState.particles.focus(lambda p: p.data.momenta).set(state, momenta)
        state = BNPTLState.particles.focus(lambda p: p.data.masses).set(state, masses)
        state = BNPTLState.particles.focus(lambda p: p.data.forces).set(state, forces)
        state = BNPTLState.particles.focus(lambda p: p.data.position_gradients).set(
            state, positions
        )

        (
            expected_pos,
            expected_mom,
            expected_V,
            expected_cm,
            expected_forces,
            expected_grad,
        ) = _manual_baoab_no_thermostat_step(
            positions,
            momenta,
            masses,
            V,
            cell_momentum,
            barostat_mass,
            dt,
            target_pressure,
            kT,
        )
        integrator = make_baoab_npt_langevin_step(
            particles=BNPTLState.particles,
            systems=BNPTLState.systems,
            derivative_computation=create_bnptl_derivative_computation(),
            flow=euclidean_flow,
        )
        out = integrator(jax.random.key(123), state)

        assert jnp.allclose(
            out.particles.data.positions, expected_pos, rtol=3e-6, atol=3e-6
        )
        assert jnp.allclose(
            out.particles.data.momenta, expected_mom, rtol=3e-6, atol=3e-6
        )
        assert jnp.allclose(
            out.particles.data.forces, expected_forces, rtol=3e-6, atol=3e-6
        )
        assert jnp.allclose(
            out.particles.data.position_gradients, expected_grad, rtol=3e-6, atol=3e-6
        )
        assert jnp.allclose(
            out.systems.data.cell.vectors[0], expected_V, rtol=3e-6, atol=3e-6
        )
        assert jnp.allclose(
            out.systems.data.cell.volume[0],
            jnp.abs(jnp.linalg.det(expected_V)),
            rtol=3e-6,
        )
        assert jnp.allclose(
            out.systems.data.cell_momentum[0], expected_cm, rtol=3e-6, atol=3e-6
        )

    def test_temperature_control(self):
        # Larger system + stronger thermostat coupling + longer run to settle the
        # statistical noise on a small N system. The atom-side OU is the standard
        # Leimkuhler–Matthews exact solver, so convergence to ⟨p²/(2m)⟩ = (3/2)kT
        # is guaranteed in the long-time limit.
        n, kT = 16, 1.2
        state = create_bnptl_system(
            n_particles=n, box_size=5.0, kT=kT, dt=0.005, gamma=5.0, tau_p=2.0
        )
        integrator = self._make_integrator()
        _, temps = run_simulation(
            integrator,
            state,
            jax.random.key(77),
            n_equil=800,
            n_sample=1200,
            extract_fn=lambda s: compute_temperature(s, 3 * n - 3),
        )
        assert_temperature(jnp.mean(temps), kT, 0.20, "BAOAB-NPT-L ")

    def test_volume_fluctuations(self):
        state = create_bnptl_system(
            n_particles=6,
            box_size=5.0,
            kT=1.0,
            dt=0.005,
            gamma=1.0,
            tau_p=1.0,
            compressibility=10.0,
        )
        integrator = self._make_integrator()
        _, volumes = run_simulation(
            integrator,
            state,
            jax.random.key(202),
            n_equil=200,
            n_sample=200,
            extract_fn=lambda s: s.systems.data.cell.volume[0],
        )
        mean_vol, std_vol = jnp.mean(volumes), jnp.std(volumes)
        assert std_vol > 0.005 * mean_vol, "Volume must actually fluctuate"
        assert std_vol / mean_vol < 1.0, "Volume must not explode"

    def test_cell_remains_lower_triangular(self):
        state = create_bnptl_system(n_particles=5, box_size=5.0)
        integrator = self._make_integrator()
        final_state, _ = run_simulation(
            integrator,
            state,
            jax.random.key(11),
            n_equil=0,
            n_sample=50,
            extract_fn=lambda s: s.systems.data.cell.volume[0],
        )
        V = final_state.systems.data.cell.vectors[0]
        upper = V - jnp.tril(V)
        assert jnp.allclose(upper, 0.0, atol=1e-8), (
            f"cell.vectors must remain lower-triangular, got upper part:\n{upper}"
        )

    def test_triclinic_starting_cell_stays_well_conditioned(self):
        """Initialize with a non-orthogonal cell; the integrator must keep
        ``det(V) > 0`` and the perpendicular lengths well above zero over a
        moderate run. Failure mode is cell collapse / degeneracy."""
        n, kT = 8, 1.0
        k1, k2 = jax.random.split(jax.random.key(101))
        # Start with a sheared triclinic cell.
        h0 = jnp.array([[5.0, 0.0, 0.0], [1.0, 4.5, 0.0], [0.5, 0.8, 4.0]])
        positions = (jax.random.uniform(k1, (n, 3)) - 0.5) * 2.0
        momenta = jax.random.normal(k2, (n, 3)) * jnp.sqrt(kT)
        forces = -positions
        masses = jnp.ones(n)
        si = Index.new([SystemId(0)] * n)
        particles = Table.arange(
            ParticleData(
                positions=positions,
                momenta=momenta,
                forces=forces,
                masses=masses,
                system=si,
                position_gradients=-forces,
            ),
            label=ParticleId,
        )
        from kups.core.utils.jax import tree_zeros_like

        cell = PeriodicCell(TriclinicFrame.from_matrix(h0[None]))
        V0 = float(jnp.abs(jnp.linalg.det(h0)))
        h_diag = jnp.diagonal(h0)[None]  # (1, 3)
        tau_p = jnp.array([1.0])
        kappa = jnp.array([10.0])
        per_column = (
            3.0
            * V0
            * (tau_p[..., None] / (2.0 * jnp.pi)) ** 2
            / (kappa[..., None] * h_diag**2)
        )
        M = jnp.tril(jnp.broadcast_to(per_column[..., None, :], (1, 3, 3)))
        gamma = jnp.array([1.0])
        params = BNPTLParams(
            time_step=jnp.array([0.005]),
            temperature=jnp.array([kT / BOLTZMANN_CONSTANT]),
            friction_coefficient=gamma,
            target_pressure=jnp.array([0.0]),
            pressure_coupling_time=tau_p,
            compressibility=kappa,
            barostat_mass=M,
            barostat_friction=jnp.tril(
                jnp.broadcast_to(gamma[..., None, None], M.shape)
            ),
        )
        systems = Table.arange(
            BNPTLSystemData(
                cell=cell,
                cell_gradients=tree_zeros_like(cell),
                cell_momentum=jnp.zeros((1, 3, 3)),
                integrator_params=params,
            ),
            label=SystemId,
        )
        state = BNPTLState(particles=particles, systems=systems)

        deriv = create_bnptl_derivative_computation()
        integrator = make_baoab_npt_langevin_step(
            BNPTLState.particles, BNPTLState.systems, deriv, euclidean_flow
        )
        final_state, vols = run_simulation(
            integrator,
            state,
            jax.random.key(99),
            n_equil=0,
            n_sample=300,
            extract_fn=lambda s: s.systems.data.cell.volume[0],
        )
        # No collapse to zero or NaN.
        assert jnp.all(vols > 0.1 * V0), "Cell collapsed below 10% of initial volume"
        assert jnp.all(jnp.isfinite(vols)), "Volume went non-finite"
        # Off-diagonal elements of the cell remain finite and non-degenerate.
        V_final = final_state.systems.data.cell.vectors[0]
        assert jnp.all(jnp.isfinite(V_final))
        assert jnp.abs(jnp.linalg.det(V_final)) > 0.1 * V0

    def test_off_diagonal_cell_dofs_get_exercised(self):
        """Initial cubic cell + thermal noise on the cell DOFs must produce
        off-diagonal h elements over time (verifies the *anisotropic*
        flexibility, not just isotropic-by-coincidence)."""
        state = create_bnptl_system(
            n_particles=8,
            box_size=5.0,
            kT=1.0,
            dt=0.005,
            gamma=5.0,
            tau_p=1.0,
        )
        integrator = self._make_integrator()
        _, V_traj = run_simulation(
            integrator,
            state,
            jax.random.key(31),
            n_equil=100,
            n_sample=300,
            extract_fn=lambda s: s.systems.data.cell.vectors[0],
        )
        # V_traj has shape (300, 3, 3). Off-diagonal elements should
        # fluctuate, not be ~0 the whole run.
        off_diag = jnp.stack(
            [V_traj[:, 1, 0], V_traj[:, 2, 0], V_traj[:, 2, 1]], axis=-1
        )
        rms_off = jnp.sqrt(jnp.mean(off_diag**2))
        # Expect off-diagonal RMS to be at least ~1% of the diagonal.
        rms_diag = jnp.sqrt(jnp.mean(jnp.diagonal(V_traj, axis1=-2, axis2=-1) ** 2))
        assert rms_off > 0.005 * rms_diag, (
            f"Off-diagonal cell DOFs are barely active (rms_off={rms_off:.4e}, "
            f"rms_diag={rms_diag:.4e}); the anisotropic barostat is not "
            f"exercising the shear DOFs."
        )

    def test_p_h_constraint_preserved(self):
        """Strict-upper part of cell_momentum stays zero throughout integration."""
        state = create_bnptl_system(n_particles=5, box_size=5.0)
        integrator = self._make_integrator()
        final_state, _ = run_simulation(
            integrator,
            state,
            jax.random.key(13),
            n_equil=0,
            n_sample=30,
            extract_fn=lambda s: s.systems.data.cell.volume[0],
        )
        cm = final_state.systems.data.cell_momentum[0]
        upper = cm - jnp.tril(cm)
        assert jnp.allclose(upper, 0.0, atol=1e-10), (
            f"cell_momentum must remain lower-triangular, got upper part:\n{upper}"
        )


def _bnptl_hamiltonian(state: BNPTLState, k: float = 1.0) -> Array:
    r"""Extended-phase-space Hamiltonian for the harmonic NPT test system.

    $$H = \sum_{\alpha\beta} \frac{(p^h_{\alpha\beta})^2}{2 M_{\alpha\beta}}
        + \sum_i \frac{p_i^2}{2 m_i} + U + P_{\text{ext}} V + \chi k_B T \ln V$$

    with $U = \tfrac{1}{2} k \sum_i r_i^2$ for the harmonic test forces.
    """
    par = state.particles.data
    sys = state.systems.data
    params = sys.integrator_params

    # Atomic KE (uses the algorithm-side p, not p_phys — see Gao Remark 2).
    ke_atomic = jnp.sum(0.5 * par.momenta**2 / par.masses[..., None])

    # Cell KE on the lower-triangular DOFs only. Use safe-division mask.
    M = params.barostat_mass
    cm = sys.cell_momentum
    ke_cell = jnp.sum(jnp.where(M > 0, 0.5 * cm**2 / jnp.where(M > 0, M, 1.0), 0.0))

    U = 0.5 * k * jnp.sum(par.positions**2)
    V = sys.cell.volume[0]
    P_ext = params.target_pressure[0]
    kT = params.temperature[0] * BOLTZMANN_CONSTANT
    chi = 1  # _GAO_CHI for 3D lower-tri parameterisation
    return ke_atomic + ke_cell + U + P_ext * V + chi * kT * jnp.log(V)


class TestBNPTLHamiltonianConservation:
    """Deterministic (γ=γ_b=0) Hamiltonian conservation — reproduces Gao Fig. 2.

    The BAOAB-NPT-Langevin Trotter splitting is a symplectic composition of
    exact substeps for the deterministic vector field; with both frictions
    zeroed, the extended-phase-space Hamiltonian oscillates with bounded
    O(Δt²) amplitude around its initial value — no secular drift.

    Test setup notes:
        * Box size 20 with positions tightly concentrated in ``|r| ≤ 1`` keeps
          (a) positions away from the periodic boundary so the in-step
          :class:`WrapStep` never fires (the harmonic potential is *not*
          translation-invariant, so wrapping would shift U), and (b) the
          harmonic-force virial small so the Trotter error coefficient stays
          tractable.
        * ``P_ext = 0`` — the harmonic-to-origin system has no proper equation
          of state; ``P_ext ≠ 0`` would drive the cell to collapse and bury
          integration error in system-level dynamics.
    """

    def _make_no_thermostat_state(self, dt: float) -> BNPTLState:
        n, box, kT = 8, 20.0, 1.0
        k1, k2 = jax.random.split(jax.random.key(7))
        positions = (jax.random.uniform(k1, (n, 3)) - 0.5) * 2.0  # tight, ±1
        momenta = jax.random.normal(k2, (n, 3)) * jnp.sqrt(kT)
        forces = -positions
        masses = jnp.ones(n)
        si = Index.new([SystemId(0)] * n)
        particles = Table.arange(
            ParticleData(
                positions=positions,
                momenta=momenta,
                forces=forces,
                masses=masses,
                system=si,
                position_gradients=-forces,
            ),
            label=ParticleId,
        )
        from kups.core.utils.jax import tree_zeros_like

        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * box))
        V0 = box**3
        # Gao Eq. (14) per-row mass for cubic h₀ = box * I.
        M = jnp.tril(
            jnp.full(
                (1, 3, 3),
                3.0 * V0 * (1.0 / (2.0 * jnp.pi)) ** 2 / (10.0 * box**2),
            )
        )
        params = BNPTLParams(
            time_step=jnp.array([dt]),
            temperature=jnp.array([kT / BOLTZMANN_CONSTANT]),
            friction_coefficient=jnp.array([0.0]),
            target_pressure=jnp.array([0.0]),
            pressure_coupling_time=jnp.array([1.0]),
            compressibility=jnp.array([10.0]),
            barostat_mass=M,
            barostat_friction=jnp.zeros((1, 3, 3)),
        )
        systems = Table.arange(
            BNPTLSystemData(
                cell=cell,
                cell_gradients=tree_zeros_like(cell),
                cell_momentum=jnp.zeros((1, 3, 3)),
                integrator_params=params,
            ),
            label=SystemId,
        )
        return BNPTLState(particles=particles, systems=systems)

    def test_hamiltonian_bounded(self):
        """With γ=γ_b=0, max |H − H₀| over a 500-step run is tiny."""
        dt = 0.001
        state = self._make_no_thermostat_state(dt=dt)
        deriv = create_bnptl_derivative_computation()
        integrator = make_baoab_npt_langevin_step(
            particles=BNPTLState.particles,
            systems=BNPTLState.systems,
            derivative_computation=deriv,
            flow=euclidean_flow,
        )
        H0 = _bnptl_hamiltonian(state)
        _, hs = run_simulation(
            integrator,
            state,
            jax.random.key(0),
            n_equil=0,
            n_sample=500,
            extract_fn=_bnptl_hamiltonian,
        )
        rel_drift = float(jnp.max(jnp.abs(hs - H0)) / jnp.abs(H0))
        # Empirically rel_drift ~ 1e-6 at this Δt and position scale; we leave
        # generous headroom for the JAX-version drift.
        assert rel_drift < 1e-4, (
            f"H drifted by {rel_drift:.2e} over 500 steps (Δt={dt})"
        )

    def test_hamiltonian_drift_scales_as_dt_squared(self):
        """Second-order Trotter splitting: max drift ∝ Δt² at fixed wall-time."""
        deriv = create_bnptl_derivative_computation()
        wall_time = 0.5  # long enough to actually accumulate Trotter error
        dts = [0.002, 0.001, 0.0005]
        drifts = []
        for dt in dts:
            state = self._make_no_thermostat_state(dt=dt)
            integrator = make_baoab_npt_langevin_step(
                particles=BNPTLState.particles,
                systems=BNPTLState.systems,
                derivative_computation=deriv,
                flow=euclidean_flow,
            )
            n_steps = int(round(wall_time / dt))
            H0 = _bnptl_hamiltonian(state)
            _, hs = run_simulation(
                integrator,
                state,
                jax.random.key(0),
                n_equil=0,
                n_sample=n_steps,
                extract_fn=_bnptl_hamiltonian,
            )
            drifts.append(float(jnp.max(jnp.abs(hs - H0)) / jnp.abs(H0)))
        # Halving Δt should reduce drift by ~4×.
        ratios = [drifts[i] / drifts[i + 1] for i in range(len(dts) - 1)]
        for r, dt_pair in zip(ratios, list(zip(dts[:-1], dts[1:]))):
            assert r > 3.0, (
                f"Drift reduced by {r:.2f}× when Δt halved from {dt_pair[0]} to "
                f"{dt_pair[1]}; expected ~4×. Drifts={drifts}"
            )
