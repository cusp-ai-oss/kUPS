# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rigid-body MD integrator.

Covers the genuinely new pieces of `kups.md.rigid`:

- Pure NO_SQUISH (`QuaternionDriftStep` math) — symplectic, time-reversible,
  unit-norm, conserves energy and lab-frame angular momentum.
- DOF accounting for linear vs. nonlinear motifs.
- End-to-end NVE on a tiny TIP4P/2005-like water system: total energy drifts
  only by symplectic (bounded) amount.
- End-to-end NVT-CSVR: kinetic temperature equilibrates to the setpoint.

All tests use small systems and short trajectories so they finish in seconds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kups.application.mcmc.data import AdsorbateConfig
from kups.application.md.data import (
    RigidBaoabLangevinParameters,
    RigidCsvrNptParameters,
    RigidCsvrParameters,
    RigidVerletParameters,
    build_rigid_state_from_grid,
)
from kups.application.md.rigid_logging import _per_group_kinetic_energy
from kups.application.md.simulation import make_rigid_md_propagator
from kups.application.simulations.md_rigid import RigidMdState
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.lens import identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.potential import sum_potentials
from kups.core.propagator import propagate_and_fix
from kups.core.result import as_result_function
from kups.core.utils.jax import jit, key_chain
from kups.core.utils.quaternion import Quaternion
from kups.core.utils.rigid_body import inertia_tensor_diag, is_linear_motif
from kups.md.rigid import (
    _l_lab_from_l_body,
    _rotate_quaternion_l_about_axis,
)
from kups.potential.classical.ewald import EwaldParameters, make_ewald_from_state
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
    make_lennard_jones_from_state,
)


_WATER_DICT = {
    "critical_temperature": 647.16,
    "critical_pressure": 22055000,
    "acentric_factor": 0.3449,
    "positions": [
        [0.0, 0.0, 0.0],
        [0.0, 0.75695, 0.58588],
        [0.0, -0.75695, 0.58588],
        [0.0, 0.0, 0.155],
    ],
    "symbols": ["Ow", "Hw", "Hw", "Mw"],
    "charges": [0.0, 0.5564, 0.5564, -1.1128],
    "masses": [15.999, 1.008, 1.008, 0.0],
    "atomic_numbers": [8, 1, 1, 0],
}


def _make_water() -> AdsorbateConfig:
    return AdsorbateConfig(**_WATER_DICT)


def _stand_alone_no_squish(
    q: jax.Array,
    l_body: jax.Array,
    inertia: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """One Miller (2002) NO_SQUISH step on isolated (q, L_body)."""

    def phi(arr: jax.Array, axis: int, frac: float) -> jax.Array:
        return frac * dt * arr[axis] / (2.0 * inertia[axis])

    q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 2, phi(l_body, 2, 0.5))
    q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 1, phi(l_body, 1, 0.5))
    q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 0, phi(l_body, 0, 1.0))
    q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 1, phi(l_body, 1, 0.5))
    q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 2, phi(l_body, 2, 0.5))
    return q, l_body


def test_no_squish_symmetric_top_precession():
    r"""Free symmetric-top: body-frame L precesses at the analytical rate.

    For a symmetric top $I = (I_\perp, I_\perp, I_\text{axis})$ free of torque,
    Euler's equations give

    $$\dot L_1 = \alpha' L_2,\qquad \dot L_2 = -\alpha' L_1,
       \qquad \alpha' = \frac{L_3 (I_\perp - I_\text{axis})}{I_\perp I_\text{axis}}$$

    so that $L_\text{body}$ traces a circle of period $T = 2\pi/|\alpha'|$ in
    the (1, 2) plane, while $L_3$ and $L_\text{lab}$ are exactly conserved.

    This test sensitively regresses both NO_SQUISH bugs that surfaced during
    development. The q-rotation half-angle factor is locked in by the
    $|L_\text{lab}|$ conservation check (a wrong factor desynchronises q and
    L_body, and lab-frame angular momentum drifts). The L-rotation direction
    is locked in by the quarter-period check ($L_\text{body}(T/4)$ must land
    on $(0, +A, L_3)$, not $(0, -A, L_3)$).
    """
    i_perp, i_axis = 1.0, 2.0
    inertia = jnp.array([i_perp, i_perp, i_axis])
    l_body0 = jnp.array([1.0, 0.0, 4.0])
    q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
    l_lab0 = _l_lab_from_l_body(Quaternion(q0), l_body0)

    # ω_3 = L_3 / I_3 = 2; α' = ω_3 (I_⊥ - I_axis)/I_⊥ = −2; T = π.
    period = float(jnp.pi)
    dt = 0.005
    n_period = int(round(period / dt))
    n_quarter = n_period // 4

    def make_run(n_steps: int):
        @jit
        def run(q, l_body):
            def step(carry, _):
                q, l_body = carry
                q, l_body = _stand_alone_no_squish(q, l_body, inertia, dt)
                return (q, l_body), None

            (q_out, l_out), _ = jax.lax.scan(step, (q, l_body), None, length=n_steps)
            return q_out, l_out

        return run

    run_quarter = make_run(n_quarter)
    run_period = make_run(n_period)
    q_q, l_q = run_quarter(q0, l_body0)
    # Analytical L_body(T/4) for our IC: L_1=0, L_2=+1, L_3=4.
    assert abs(float(l_q[0])) < 5e-2, f"L_1(T/4) = {l_q[0]:.4f}, expected 0"
    assert float(l_q[1]) > 0.9, (
        f"L_2(T/4) = {l_q[1]:.4f}; sign indicates L-rotation direction "
        f"is reversed (regressed bug)"
    )
    assert abs(float(l_q[2]) - float(l_body0[2])) < 1e-4, (
        f"L_3 not conserved on symmetry axis: {l_q[2]:.4f} vs {l_body0[2]:.4f}"
    )

    q_p, l_p = run_period(q0, l_body0)
    # After one full period, L_body must return to start (closed orbit).
    err_lbody = float(jnp.linalg.norm(l_p - l_body0))
    assert err_lbody < 5e-3, f"L_body did not return after T_prec: |Δ| = {err_lbody:.4e}"

    # Lab-frame angular momentum must be conserved exactly under free
    # rotation; a wrong q-rotation half-angle desynchronises q and L_body
    # and breaks this invariant.
    l_lab_p = _l_lab_from_l_body(Quaternion(q_p), l_p)
    err_llab = float(jnp.linalg.norm(l_lab_p - l_lab0)) / float(jnp.linalg.norm(l_lab0))
    assert err_llab < 1e-3, (
        f"|L_lab| drift {err_llab:.4e} indicates the q-rotation half-angle "
        f"factor is wrong (regressed bug)"
    )


def test_no_squish_unit_norm_and_conservation():
    """Pure NO_SQUISH: |q| stays unit, L_lab and KE_rot are conserved.

    No external torques; running it for 10⁴ steps must keep the quaternion
    norm unit, the lab-frame angular momentum constant, and the rotational
    kinetic energy bounded (oscillation only) — all to round-off.
    """
    inertia = jnp.array([1.0, 1.5, 2.0])
    l_body0 = jnp.array([1.0, 0.5, 0.3])
    q0 = jnp.array([1.0, 0.0, 0.0, 0.0])
    dt = 0.01
    n_steps = 10_000

    @jit
    def run(q, l_body):
        def step(carry, _):
            q, l_body = carry
            q, l_body = _stand_alone_no_squish(q, l_body, inertia, dt)
            return (q, l_body), None

        (q_out, l_out), _ = jax.lax.scan(step, (q, l_body), None, length=n_steps)
        return q_out, l_out

    q, l_body = run(q0, l_body0)
    norm = float(jnp.linalg.norm(q))
    assert abs(norm - 1.0) < 1e-6, f"|q| drifted by {norm - 1.0:.2e}"

    l_lab0 = _l_lab_from_l_body(Quaternion(q0), l_body0)
    l_lab = _l_lab_from_l_body(Quaternion(q), l_body)
    drift = float(jnp.linalg.norm(l_lab - l_lab0)) / float(jnp.linalg.norm(l_lab0))
    assert drift < 1e-3, f"|L_lab| drifted relatively by {drift:.2e}"

    e0 = float(jnp.sum(l_body0**2 / (2 * inertia)))
    e = float(jnp.sum(l_body**2 / (2 * inertia)))
    rel = abs(e - e0) / abs(e0)
    assert rel < 1e-3, f"KE drift too large: {rel:.2e}"


def test_dof_water_is_nonlinear():
    """TIP4P/2005 water motif must be detected as nonlinear → 6N − 3 DOF."""
    water = _make_water()
    positions = jnp.asarray(water.positions, dtype=float)
    masses = jnp.asarray(water.masses, dtype=float)
    com = (masses[:, None] * positions).sum(axis=0) / masses.sum()
    centred = positions - com
    inertia, _ = inertia_tensor_diag(centred, masses)
    assert not is_linear_motif(inertia), (
        f"Water flagged as linear; principal moments were {inertia}"
    )


def test_dof_co2_like_motif_is_linear():
    """A linear three-site motif (CO₂-like) must be detected as linear → 5N − 3 DOF."""
    positions = jnp.array(
        [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]  # O — C — O on z-axis
    )
    masses = jnp.array([15.999, 12.011, 15.999])
    com = (masses[:, None] * positions).sum(axis=0) / masses.sum()
    centred = positions - com
    inertia, _ = inertia_tensor_diag(centred, masses)
    assert is_linear_motif(inertia), f"CO₂ flagged as nonlinear: {inertia}"


def _build_water_state(
    n_molecules: int,
    box_edge: float,
    integrator: str,
    *,
    friction: float,
    timestep_fs: float = 0.5,
    cutoff: float = 4.0,
    tau_fs: float = 100.0,
    seed: int = 42,
):
    """Helper: TIP4P/2005 water in a small box, ready for end-to-end tests."""
    chain = key_chain(jax.random.key(seed))
    water = _make_water()
    params: (
        RigidVerletParameters
        | RigidBaoabLangevinParameters
        | RigidCsvrParameters
        | RigidCsvrNptParameters
    )
    if integrator == "rigid_verlet":
        params = RigidVerletParameters(
            temperature=298.0, time_step=timestep_fs, initialize_momenta=True,
        )
    elif integrator == "rigid_baoab_langevin":
        params = RigidBaoabLangevinParameters(
            temperature=298.0, time_step=timestep_fs, initialize_momenta=True,
            friction_coefficient=friction,
        )
    elif integrator == "rigid_csvr":
        params = RigidCsvrParameters(
            temperature=298.0, time_step=timestep_fs, initialize_momenta=True,
            thermostat_time_constant=tau_fs,
        )
    elif integrator == "rigid_csvr_npt":
        params = RigidCsvrNptParameters(
            temperature=298.0, time_step=timestep_fs, initialize_momenta=True,
            thermostat_time_constant=tau_fs,
            target_pressure=101325.0,
            pressure_coupling_time=1000.0,
            compressibility=4.5e-10,
            minimum_scale_factor=0.5,
        )
    else:
        raise ValueError(f"unknown integrator: {integrator}")
    particles, groups, motifs, systems = build_rigid_state_from_grid(
        next(chain),
        (water,),
        (n_molecules,),
        (box_edge, box_edge, box_edge),
        params,
    )

    lj_params = GlobalTailCorrectedLennardJonesParameters.from_dict(
        cutoff=cutoff,
        parameters={"Ow": (3.1589, 0.008031), "Hw": (None, None), "Mw": (None, None)},
        mixing_rule="lorentz_berthelot",
        tail_correction=False,
    )
    ewald_params = EwaldParameters.make(
        particles, systems, epsilon_total=1e-4, real_cutoff=cutoff,
    )
    nl_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts,
        systems,
        lj_params.cutoff,
    )
    state = RigidMdState(
        particles=particles,
        groups=groups,
        motifs=motifs,
        systems=systems,
        neighborlist_params=nl_params,
        step=jnp.array([0]),
        lj_parameters=lj_params,
        ewald_parameters=ewald_params,
    )
    state_lens = identity_lens(RigidMdState)
    potential = sum_potentials(
        make_ewald_from_state(
            state_lens,
            compute_position_and_unitcell_gradients=True,
            include_exclusion_mask=True,
        ),
        make_lennard_jones_from_state(
            state_lens, compute_position_and_unitcell_gradients=True
        ),
    )
    propagator = make_rigid_md_propagator(state_lens, integrator, potential)  # type: ignore[arg-type]
    return state, propagator, next(chain)


def _total_energy(state: RigidMdState) -> tuple[float, float, float]:
    ke = float(jnp.sum(_per_group_kinetic_energy(state.groups)))
    pe = float(state.systems.data.potential_energy[0])
    return ke + pe, ke, pe


def test_nve_energy_drift_bounded():
    """NVE rigid Verlet on 8 waters: total energy stays bounded over 100 steps.

    Symplectic integrators allow oscillating energy but should not drift
    monotonically beyond ~1e-4 relative over a short run with cutoff=4 Å.
    """
    state, propagator, key = _build_water_state(
        n_molecules=8, box_edge=10.0, integrator="rigid_verlet", friction=0.0,
    )
    chain = key_chain(key)
    prop = jit(as_result_function(propagator), donate_argnums=(1,))

    # One step to populate forces correctly before sampling E0.
    state = propagate_and_fix(prop, next(chain), state)
    e0, _, _ = _total_energy(state)
    drifts = []
    for _ in range(100):
        state = propagate_and_fix(prop, next(chain), state)
        e, _, _ = _total_energy(state)
        drifts.append(abs(e - e0) / abs(e0))
    max_drift = max(drifts)
    assert max_drift < 1e-3, f"NVE energy drift too large: max={max_drift:.2e}"


def test_csvr_temperature_setpoint():
    """NVT-CSVR rigid: kinetic temperature equilibrates to the setpoint.

    32 waters at near-liquid density (DOF = 6·32 − 3 = 189). Strong CSVR
    coupling (τ = 10 fs) so that 500 warmup steps are enough to relax the
    initial random-orientation potential energy. Mean temperature over 500
    production steps should fall within ±15 K of 298 K (1-σ for canonical
    fluctuations is ~31 K).
    """
    state, propagator, key = _build_water_state(
        n_molecules=32,
        box_edge=9.94,
        cutoff=4.5,
        integrator="rigid_csvr",
        friction=0.0,
        timestep_fs=0.5,
        tau_fs=10.0,
    )
    chain = key_chain(key)
    prop = jit(as_result_function(propagator), donate_argnums=(1,))

    for _ in range(500):
        state = propagate_and_fix(prop, next(chain), state)

    dof = float(state.systems.data.degrees_of_freedom[0])
    temps = []
    for _ in range(500):
        state = propagate_and_fix(prop, next(chain), state)
        ke = float(jnp.sum(_per_group_kinetic_energy(state.groups)))
        temps.append(2.0 * ke / (dof * BOLTZMANN_CONSTANT))
    mean_T = float(np.mean(temps))
    assert 283.0 < mean_T < 313.0, (
        f"CSVR mean temperature {mean_T:.1f} K not within ±15 K of setpoint 298 K"
    )


def test_baoab_langevin_temperature_setpoint():
    """Rigid BAOAB Langevin drives ⟨T⟩ to the setpoint.

    Exercises :class:`StochasticStep` (translational OU on COM) **and**
    :class:`RigidRotationalStochasticStep` (body-frame OU per principal
    axis), neither of which is otherwise covered. Strong friction
    (γ = 0.1 / fs) keeps the relaxation time short.
    """
    state, propagator, key = _build_water_state(
        n_molecules=32,
        box_edge=9.94,
        cutoff=4.5,
        integrator="rigid_baoab_langevin",
        friction=0.1,
        timestep_fs=0.5,
        tau_fs=10.0,
    )
    chain = key_chain(key)
    prop = jit(as_result_function(propagator), donate_argnums=(1,))

    for _ in range(500):
        state = propagate_and_fix(prop, next(chain), state)

    dof = float(state.systems.data.degrees_of_freedom[0])
    temps = []
    for _ in range(500):
        state = propagate_and_fix(prop, next(chain), state)
        ke = float(jnp.sum(_per_group_kinetic_energy(state.groups)))
        temps.append(2.0 * ke / (dof * BOLTZMANN_CONSTANT))
    mean_T = float(np.mean(temps))
    assert 278.0 < mean_T < 318.0, (
        f"Langevin mean temperature {mean_T:.1f} K not within ±20 K of 298 K"
    )


def test_npt_volume_responds_and_density_in_window():
    """Rigid NPT-CSVR: volume rescales and ⟨ρ⟩ settles in a sensible window.

    Exercises the full NPT pipeline: ``RigidCSVRStep`` →
    ``StochasticCellRescalingStep`` with rigid overrides
    (``_rigid_translational_kinetic_energy_view``, ``_molecular_stress_view``,
    COM-position lens) → atom reconstruction → fresh force evaluation. None of
    these are touched by the NVE / NVT tests.

    With cutoff = 4.5 Å (forced by the small box) the LJ tail truncation
    pushes ⟨ρ⟩ well below the canonical 0.997 g/cm³; assert it lands in a
    loose window so the test catches gross failures (volume blowing up,
    pressure decoupling) without depending on tail-correction-quality
    physics.
    """
    state, propagator, key = _build_water_state(
        n_molecules=32,
        box_edge=9.94,
        cutoff=4.5,
        integrator="rigid_csvr_npt",
        friction=0.0,
        timestep_fs=0.5,
        tau_fs=10.0,
    )
    chain = key_chain(key)
    prop = jit(as_result_function(propagator), donate_argnums=(1,))

    v0 = float(state.systems.data.unitcell.volume[0])
    for _ in range(500):
        state = propagate_and_fix(prop, next(chain), state)
    v_after_warmup = float(state.systems.data.unitcell.volume[0])
    assert abs(v_after_warmup - v0) / v0 > 1e-3, (
        f"NPT volume did not rescale: V0={v0:.2f}, V={v_after_warmup:.2f} Å³"
    )

    mass_g = 32 * 18.015 * 1.66054e-24
    rho_samples: list[float] = []
    for _ in range(500):
        state = propagate_and_fix(prop, next(chain), state)
        v = float(state.systems.data.unitcell.volume[0])
        rho_samples.append(mass_g / (v * 1e-24))
    mean_rho = float(np.mean(rho_samples))
    assert 0.5 < mean_rho < 1.1, (
        f"NPT mean density {mean_rho:.3f} g/cm³ outside sanity window [0.5, 1.1] "
        f"(truncated-LJ TIP4P/2005 at r_c=4.5 Å expected near 0.75-0.85)"
    )
