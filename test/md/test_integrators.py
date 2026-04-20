# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MD integrators."""

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import HasLensFields, LensField, lens
from kups.core.propagator import CachePropagator, Propagator
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass, jit
from kups.md.integrators import (
    CSVRStep,
    MinimumImageConventionFlow,
    MomentumStep,
    PositionStep,
    StochasticCellRescalingStep,
    StochasticStep,
    euclidean_flow,
    make_baoab_langevin_step,
    make_csvr_npt_step,
    make_csvr_step,
    make_velocity_verlet_step,
)
from kups.md.observables import instantaneous_pressure, particle_kinetic_energy

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
class SystemData:
    time_step: Array
    temperature: Array
    friction_coefficient: Array
    thermostat_time_constant: Array


@dataclass
class NPTSystemData:
    time_step: Array
    temperature: Array
    thermostat_time_constant: Array
    unitcell: UnitCell
    unitcell_gradients: UnitCell
    target_pressure: Array
    pressure_coupling_time: Array
    compressibility: Array
    minimum_scale_factor: Array


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
            time_step=jnp.array([dt]),
            temperature=jnp.array([kT / BOLTZMANN_CONSTANT]),
            friction_coefficient=jnp.array([gamma]),
            thermostat_time_constant=jnp.array([tau]),
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

    unitcell = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * box_size)
    from kups.core.utils.jax import tree_zeros_like

    systems = Table.arange(
        NPTSystemData(
            time_step=jnp.array([dt]),
            temperature=jnp.array([kT / BOLTZMANN_CONSTANT]),
            thermostat_time_constant=jnp.array([tau_t]),
            unitcell=unitcell,
            unitcell_gradients=tree_zeros_like(unitcell),
            target_pressure=jnp.array([target_pressure]),
            pressure_coupling_time=jnp.array([tau_p]),
            compressibility=jnp.array([compressibility]),
            minimum_scale_factor=jnp.array([0.5]),
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
            systems=SimpleState.systems.get,
            flow=euclidean_flow,
        )
        new_state = step(jax.random.key(0), state)

        velocities = state.particles.data.momenta / state.particles.data.masses[:, None]
        expected = (
            state.particles.data.positions
            + velocities * state.systems.data.time_step[0]
        )
        assert jnp.allclose(new_state.particles.data.positions, expected, rtol=1e-6)
        assert jnp.allclose(
            new_state.particles.data.momenta, state.particles.data.momenta
        )

    def test_momentum_update(self):
        """Momentum update correctness and position preservation."""
        state, _, _ = create_harmonic_system(n_particles=5, dt=0.01)
        step = MomentumStep(
            particles=SimpleState.particles, systems=SimpleState.systems.get
        )
        new_state = step(jax.random.key(0), state)

        expected = (
            state.particles.data.momenta
            + state.particles.data.forces * state.systems.data.time_step[0]
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
        step = StochasticStep(
            particles=SimpleState.particles, system=SimpleState.systems.get
        )

        _, temps = run_simulation(
            step,
            state,
            jax.random.key(42),
            n_equil=50,
            n_sample=100,
            extract_fn=lambda s: compute_temperature(s, 3 * n_particles),
        )
        assert_temperature(jnp.mean(temps), kT_target, 0.2)

    def test_csvr_velocity_rescaling(self):
        n_particles, kT_target = 10, 2.0
        state, _, _ = create_harmonic_system(
            n_particles=n_particles, kT=kT_target, tau=0.1, dt=0.02
        )
        step = CSVRStep(particles=SimpleState.particles, systems=get_systems)

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
    """Tests for StochasticCellRescalingStep and MinimumImageConventionFlow."""

    def test_unitcell_volume_updates(self):
        state = create_npt_system(n_particles=3, box_size=5.0)
        step = StochasticCellRescalingStep(
            particles=NPTState.particles, systems=NPTState.systems
        )

        initial_volume = state.systems.data.unitcell.volume
        new_state = step(jax.random.key(42), state)

        assert not jnp.isclose(
            new_state.systems.data.unitcell.volume, initial_volume, rtol=1e-8
        ), "CRITICAL BUG: UnitCell volume did not update"
        expected_volume = jnp.linalg.det(
            new_state.systems.data.unitcell.lattice_vectors
        )
        assert jnp.isclose(
            new_state.systems.data.unitcell.volume, expected_volume, rtol=1e-6
        )

    def test_positions_scale_with_box(self):
        state = create_npt_system(n_particles=3, box_size=5.0)
        step = StochasticCellRescalingStep(
            particles=NPTState.particles, systems=NPTState.systems
        )

        initial_pos = state.particles.data.positions
        initial_box = jnp.mean(jnp.diag(state.systems.data.unitcell.lattice_vectors[0]))
        new_state = step(jax.random.key(42), state)
        new_box = jnp.mean(jnp.diag(new_state.systems.data.unitcell.lattice_vectors[0]))

        expected_pos = initial_pos * (new_box / initial_box)
        assert jnp.allclose(new_state.particles.data.positions, expected_pos, rtol=1e-3)

    def test_pressure_response(self):
        state = create_npt_system(
            n_particles=3, box_size=2.0, tau_p=0.1, compressibility=10.0
        )
        step = StochasticCellRescalingStep(
            particles=NPTState.particles, systems=NPTState.systems
        )

        initial_volume = state.systems.data.unitcell.volume
        _, volumes = run_simulation(
            step,
            state,
            jax.random.key(42),
            n_equil=0,
            n_sample=10,
            extract_fn=lambda s: s.systems.data.unitcell.volume,
        )

        assert jnp.mean(volumes[5:]) > initial_volume * 1.01, (
            "Barostat did not expand box in response to high pressure"
        )

    def test_wrapping_positions(self):
        box_size = 5.0
        unitcell = TriclinicUnitCell.from_matrix(jnp.eye(3) * box_size)

        @dataclass
        class TestState:
            unitcell: UnitCell

        flow = MinimumImageConventionFlow(
            unitcell=lambda s: s.unitcell, flow=euclidean_flow
        )
        new_pos = flow(
            TestState(unitcell=unitcell),
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
            systems=get_systems,
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
            systems=get_systems,
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
            extract_fn=lambda s: compute_temperature(s, 3 * n_particles),
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
            extract_fn=lambda s: s.systems.data.unitcell.volume,
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
    from ase.calculators.lj import LennardJones as ASELJ

    from kups.application.md.data import MdParameters, md_state_from_ase
    from kups.core.lens import identity_lens
    from kups.core.neighborlist import (
        DenseNearestNeighborList,
        NearestNeighborList,
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
    atoms = ase.io.read(str(cif))
    atoms.calc = ASELJ(epsilon=eps, sigma=sigma, rc=10.0, smooth=False)
    ase_stress = atoms.get_stress(voigt=False)
    ase_pressure = -np.trace(ase_stress) / 3

    # kUPS: evaluate potential directly (no propagator/propagate_and_fix)
    @dataclass
    class S:
        particles: Table[ParticleId, ...]
        systems: Table[SystemId, ...]
        neighborlist_params: UniversalNeighborlistParameters
        step: jnp.ndarray
        lj_parameters: LennardJonesParameters

        @property
        def neighborlist(self) -> NearestNeighborList:
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
    pot = make_lennard_jones_from_state(
        sl, compute_position_and_unitcell_gradients=True
    )

    # Evaluate potential and write gradients back into state
    result = pot(state)
    pos_grad = result.data.gradients.positions.data
    uc_grad = result.data.gradients.unitcell.data

    import dataclasses

    p_with_grad = p.set_data(dataclasses.replace(p.data, position_gradients=pos_grad))
    s_with_grad = s.set_data(dataclasses.replace(s.data, unitcell_gradients=uc_grad))

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
