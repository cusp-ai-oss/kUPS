# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body MD propagators.

Mirrors :mod:`kups.md.integrators` for the per-rigid-body case. The COM
direction reuses the existing :class:`MomentumStep`, :class:`PositionStep`,
:class:`StochasticStep`, and :class:`StochasticCellRescalingStep` through
a lens onto a per-group table whose translational fields are named
``positions / momenta / masses / position_gradients`` so that the existing
``HasPositions / HasMomenta / HasMasses / HasForces`` protocols are satisfied
without renaming. The rotational direction needs five new propagators
(quaternion drift, rotational kick, body-frame Langevin OU, rigid CSVR
rescaling, atom reconstruction, force/torque aggregation) because the
shape of the operations is genuinely different (per-axis principal moments,
quaternion composition, torque accumulation, atom↔group bridging).

This module is *generic* over the concrete state dataclass: protocols are
declared inline in the same style as ``integrators.py``.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Index, Table
from kups.core.lens import Lens, View, bind
from kups.core.propagator import Propagator, SequentialPropagator
from kups.core.typing import (
    GroupId,
    HasAngularMomentum,
    HasCompressibility,
    HasDegreesOfFreedom,
    HasFrictionCoefficient,
    HasGroupIndex,
    HasInertiaDiag,
    HasMasses,
    HasMinimumScaleFactor,
    HasMomenta,
    HasPositions,
    HasPressureCouplingTime,
    HasQuaternion,
    HasSystemIndex,
    HasTargetPressure,
    HasTemperature,
    HasThermostatTimeConstant,
    HasTimeStep,
    HasTorque,
    HasUnitCell,
    MotifParticleId,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import UnitCell
from kups.core.utils.functools import pipe
from kups.core.utils.jax import dataclass, field
from kups.core.utils.quaternion import Quaternion
from kups.core.utils.random import sample_like
from kups.core.utils.rigid_body import (
    aggregate_forces,
    per_group_kinetic_energy,
    reconstruct_atom_positions,
)
from kups.md.integrators import (
    Flow,
    MinimumImageConventionFlow,
    MomentumStep,
    PositionStep,
    StochasticStep,
    csvr_scale_factor,
    euclidean_flow,
    half_time,
    stochastic_cell_rescaling_factor,
)
from kups.observables.stress import molecular_stress_via_virial_theorem


type RigidIntegrator = Literal[
    "rigid_verlet", "rigid_baoab_langevin", "rigid_csvr", "rigid_csvr_npt"
]


@runtime_checkable
class _RotationalKickData(
    HasAngularMomentum, HasTorque, HasSystemIndex, Protocol
): ...


@runtime_checkable
class _RotationalDriftData(
    HasQuaternion,
    HasAngularMomentum,
    HasInertiaDiag,
    HasSystemIndex,
    Protocol,
):
    """Per-group fields needed by both NO_SQUISH drift and body-frame OU."""


@runtime_checkable
class _RigidCSVRData(
    HasMomenta,
    HasMasses,
    HasAngularMomentum,
    HasQuaternion,
    HasInertiaDiag,
    HasSystemIndex,
    Protocol,
): ...


@runtime_checkable
class _ForceAggregationAtomData(
    HasPositions,
    HasGroupIndex,
    HasSystemIndex,
    Protocol,
):
    @property
    def position_gradients(self) -> Array: ...


@runtime_checkable
class _ForceAggregationGroupData(HasPositions, HasSystemIndex, Protocol):
    @property
    def position_gradients(self) -> Array: ...
    @property
    def torque(self) -> Array: ...


@runtime_checkable
class _AtomReconstructionAtomData(HasPositions, HasGroupIndex, Protocol):
    @property
    def motif(self) -> Index[MotifParticleId]: ...


@runtime_checkable
class _AtomReconstructionGroupData(HasPositions, HasQuaternion, Protocol): ...


@runtime_checkable
class _AtomReconstructionMotifData(HasPositions, Protocol): ...


@runtime_checkable
class _RigidAtomData(
    _ForceAggregationAtomData, _AtomReconstructionAtomData, Protocol
):
    """Atom-level data shared across the rigid-body MD pipeline."""


@runtime_checkable
class _RigidGroupData(
    HasMomenta,
    HasPositions,
    HasMasses,
    HasSystemIndex,
    HasQuaternion,
    HasAngularMomentum,
    HasInertiaDiag,
    HasTorque,
    Protocol,
):
    """Per-rigid-body data shared across the rigid-body MD pipeline."""

    @property
    def forces(self) -> Array: ...
    @property
    def position_gradients(self) -> Array: ...


@runtime_checkable
class _RigidVerletSysData(HasTimeStep, HasUnitCell, Protocol): ...


@runtime_checkable
class _RigidStochasticSysData(
    HasTimeStep, HasTemperature, HasFrictionCoefficient, HasUnitCell, Protocol
): ...


@runtime_checkable
class _RigidCSVRSystemData(
    HasTimeStep,
    HasTemperature,
    HasThermostatTimeConstant,
    HasDegreesOfFreedom,
    HasUnitCell,
    Protocol,
): ...


@runtime_checkable
class _RigidNPTSystemData(
    _RigidCSVRSystemData,
    HasTargetPressure,
    HasPressureCouplingTime,
    HasCompressibility,
    HasMinimumScaleFactor,
    Protocol,
):
    """What :class:`RigidStochasticCellRescalingStep` and the rigid NPT
    factory actually need. Notably *not* :class:`HasFrictionCoefficient`:
    NPT-CSVR does not use Langevin friction.
    """

    @property
    def unitcell_gradients(self) -> UnitCell: ...


@runtime_checkable
class _RigidAnyIntegratorSysData(
    _RigidNPTSystemData,
    _RigidStochasticSysData,
    Protocol,
):
    """Union of every per-system field the rigid integrator dispatch may read.

    ``IsRigidMdState.systems`` declares this so :func:`make_rigid_md_step_from_state`
    can route to any of the four rigid integrators from a single state shape.
    Individual integrator factories take the narrower protocol they actually
    need (e.g. :class:`_RigidVerletSysData` for NVE, :class:`_RigidNPTSystemData`
    for NPT-CSVR).
    """


def _rotate_quaternion_l_about_axis(
    q: Array, l_body: Array, axis: int, dt_phi: Array
) -> tuple[Array, Array]:
    r"""Apply the body-axis sub-rotation $\hat{P}_k$ for time ``dt_phi``.

    NO_SQUISH (Miller, Eastman, Pande 2002, J. Chem. Phys. 116, 8649) splits
    the rotational Liouville operator into three commuting sub-operators
    $\hat{P}_1, \hat{P}_2, \hat{P}_3$, one per body-frame principal axis $k$.
    Each is a 4×4 rotation in $(\mathbf{q}, \mathbf{l})$ space about a
    sparse permutation matrix $P_k$ with rotation angle proportional to
    $L^{(k)}/I^{(k)}$.

    Args:
        q: Current quaternion components $(w, x, y, z)$, shape ``(..., 4)``.
        l_body: Body-frame angular momentum, shape ``(..., 3)``.
        axis: Principal axis index ``k ∈ {0, 1, 2}``.
        dt_phi: Sub-rotation angle (already includes the factor
            $L^{(k)}/(4 I^{(k)})$; see :class:`QuaternionDriftStep`),
            shape ``(...,)``.

    Returns:
        Rotated ``(q, l_body)`` pair.
    """
    # Permutation P_k acts on q as (w, x, y, z) -> body-axis-k rotation
    # P_1: (w,x,y,z) <-> (-x, w, z, -y)
    # P_2: (w,x,y,z) <-> (-y, -z, w, x)
    # P_3: (w,x,y,z) <-> (-z, y, -x, w)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    if axis == 0:
        pq = jnp.stack([-x, w, z, -y], axis=-1)
    elif axis == 1:
        pq = jnp.stack([-y, -z, w, x], axis=-1)
    else:
        pq = jnp.stack([-z, y, -x, w], axis=-1)
    cos = jnp.cos(dt_phi)[..., None]
    sin = jnp.sin(dt_phi)[..., None]
    new_q = cos * q + sin * pq
    # Under the H_k sub-Hamiltonian (free rotation about body axis k), L_lab is
    # constant. With q rotating by quaternion-space half-angle dt_phi (i.e. body
    # rotates by 2·dt_phi about +k), L_body components in the plane normal to
    # k must rotate *clockwise* (negative sense) by 2·dt_phi viewed from +k —
    # this is the body-frame consequence of dV_body/dt = -ω × V_body.
    angle_l = 2.0 * dt_phi
    cos_l = jnp.cos(angle_l)
    sin_l = jnp.sin(angle_l)
    # Clockwise rotation (cos α V_a + sin α V_b, -sin α V_a + cos α V_b) about
    # axis k. Cyclic order (a, b) follows right-handed (j, l) for axis k:
    #   k = 0  → (j, l) = (1, 2)
    #   k = 1  → (j, l) = (2, 0)
    #   k = 2  → (j, l) = (0, 1)
    if axis == 0:
        l1, l2 = l_body[..., 1], l_body[..., 2]
        new_l1 = cos_l * l1 + sin_l * l2
        new_l2 = -sin_l * l1 + cos_l * l2
        new_l_body = jnp.stack([l_body[..., 0], new_l1, new_l2], axis=-1)
    elif axis == 1:
        l0, l2 = l_body[..., 0], l_body[..., 2]
        new_l2 = cos_l * l2 + sin_l * l0
        new_l0 = -sin_l * l2 + cos_l * l0
        new_l_body = jnp.stack([new_l0, l_body[..., 1], new_l2], axis=-1)
    else:
        l0, l1 = l_body[..., 0], l_body[..., 1]
        new_l0 = cos_l * l0 + sin_l * l1
        new_l1 = -sin_l * l0 + cos_l * l1
        new_l_body = jnp.stack([new_l0, new_l1, l_body[..., 2]], axis=-1)
    return new_q, new_l_body


def _l_body_from_l_lab(quaternion: Quaternion, l_lab: Array) -> Array:
    """Rotate lab-frame angular momentum into the body frame: ``l_body = q^{-1} ⊗ l_lab``."""
    return l_lab @ quaternion.inv()


def _l_lab_from_l_body(quaternion: Quaternion, l_body: Array) -> Array:
    """Rotate body-frame angular momentum into the lab frame: ``l_lab = q ⊗ l_body``."""
    return l_body @ quaternion


@dataclass
class QuaternionDriftStep[State](Propagator[State]):
    r"""Symplectic NO_SQUISH drift of $(\mathbf{q}, \mathbf{L})$ over $\Delta t$.

    Implements the Miller/Eastman/Pande (2002) splitting:

    $$\hat{P}_3(\Delta t/2) \hat{P}_2(\Delta t/2) \hat{P}_1(\Delta t)
       \hat{P}_2(\Delta t/2) \hat{P}_3(\Delta t/2)$$

    where each $\hat{P}_k$ is a closed-form rotation of $(\mathbf{q}, \mathbf{l}_{\text{body}})$
    about body-axis $k$. The lab-frame angular momentum is rotated into the
    body frame at entry and back at exit. The scheme is symplectic and
    time-reversible; quaternions remain unit-norm to round-off without
    explicit renormalisation.

    For linear motifs, the smallest principal moment is set to ``inf``
    upstream so that the corresponding body-axis sub-rotation has zero
    angle (any finite $L$ over infinite $I$ produces no motion).

    References:
        Miller, T. F., Eastman, P., & Pande, V. S. (2002).
        Symplectic quaternion scheme for biophysical molecular dynamics.
        J. Chem. Phys., 116(20), 8649–8659. DOI: 10.1063/1.1473654
    """

    groups: Lens[State, Table[GroupId, _RotationalDriftData]] = field(static=True)
    systems: View[State, Table[SystemId, HasTimeStep]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        groups = self.groups.get(state)
        sys = self.systems(state)[groups.data.system]
        dt = sys.time_step  # (n_groups,)
        quaternion = groups.data.quaternion
        l_lab = groups.data.angular_momentum
        inertia = groups.data.inertia_diag

        l_body = _l_body_from_l_lab(quaternion, l_lab)

        def _phi(axis: int, frac: float) -> Array:
            # NO_SQUISH (Miller 2002, eq. 2.16/2.17):
            # quaternion-space rotation angle is L_k · t / (2 I_k) for substep
            # of duration t = frac · Δt. Body-frame L components co-rotate
            # by twice that angle (handled inside _rotate_quaternion_l_about_axis).
            return frac * dt * l_body[..., axis] / (2.0 * inertia[..., axis])

        q = quaternion.components
        # P_3(dt/2) P_2(dt/2) P_1(dt) P_2(dt/2) P_3(dt/2)
        q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 2, _phi(2, 0.5))
        q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 1, _phi(1, 0.5))
        q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 0, _phi(0, 1.0))
        q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 1, _phi(1, 0.5))
        q, l_body = _rotate_quaternion_l_about_axis(q, l_body, 2, _phi(2, 0.5))

        new_q = Quaternion(q)
        new_l_lab = _l_lab_from_l_body(new_q, l_body)

        groups_lens = self.groups.bind(state)
        state = groups_lens.focus(lambda g: g.data.quaternion).set(new_q)
        groups_lens = self.groups.bind(state)
        return groups_lens.focus(lambda g: g.data.angular_momentum).set(new_l_lab)


@dataclass
class RotationalMomentumStep[State](Propagator[State]):
    r"""Lab-frame angular-momentum kick: $\mathbf{L} \mathrel{+}= \boldsymbol{\tau}\,\Delta t$."""

    groups: Lens[State, Table[GroupId, _RotationalKickData]] = field(static=True)
    systems: View[State, Table[SystemId, HasTimeStep]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        groups_lens = self.groups.bind(state)
        groups = groups_lens.get()
        sys = self.systems(state)[groups.data.system]
        new_l = (
            groups.data.angular_momentum + groups.data.torque * sys.time_step[..., None]
        )
        return groups_lens.focus(lambda g: g.data.angular_momentum).set(new_l)


@dataclass
class RigidRotationalStochasticStep[State](Propagator[State]):
    r"""Body-frame Ornstein–Uhlenbeck on per-axis angular momentum.

    Mirrors :class:`StochasticStep` but with a 3-vector "mass" (per-axis
    principal moments). The lab-frame $\mathbf{L}$ is rotated into the body
    frame, OU is applied component-wise, and the result is rotated back.

    Components on a body axis with infinite inertia (linear-motif symmetry
    axis) are not stochastically refreshed (their target variance is zero).
    """

    groups: Lens[State, Table[GroupId, _RotationalDriftData]] = field(static=True)
    systems: View[State, Table[SystemId, _RigidStochasticSysData]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        groups_lens = self.groups.bind(state)
        groups = groups_lens.get()
        sys = self.systems(state)[groups.data.system]

        kT = sys.temperature * BOLTZMANN_CONSTANT
        damping = jnp.exp(-sys.friction_coefficient * sys.time_step)

        l_lab = groups.data.angular_momentum
        quaternion = groups.data.quaternion
        inertia = groups.data.inertia_diag

        l_body = _l_body_from_l_lab(quaternion, l_lab)
        # Per-axis target std: σ_a = sqrt(I_a · kT · (1 − e^{-2γΔt}))
        # An infinite I (frozen axis) yields infinite σ; we mask those to
        # leave the corresponding component unchanged.
        sigma_sq = inertia * kT[:, None] * (1.0 - damping[:, None] ** 2)
        finite = jnp.isfinite(sigma_sq)
        sigma = jnp.sqrt(jnp.where(finite, sigma_sq, 0.0))

        noise = sample_like(jax.random.normal, key, l_body)
        new_l_body = damping[:, None] * l_body + sigma * noise
        new_l_body = jnp.where(finite, new_l_body, l_body)

        new_l_lab = _l_lab_from_l_body(quaternion, new_l_body)
        return groups_lens.focus(lambda g: g.data.angular_momentum).set(new_l_lab)


@dataclass
class RigidCSVRStep[State](Propagator[State]):
    r"""Joint translational+rotational CSVR for rigid bodies.

    Computes per-system kinetic energy as the sum

    $$K = \sum_{g \in s} \frac{|\mathbf{p}_g|^2}{2 M_g}
        + \frac{1}{2} \sum_{g \in s} \sum_a \frac{(L_g^{(a)})^2}{I_g^{(a)}}$$

    (with the body-frame decomposition of $\mathbf{L}_g$), draws a single
    rescaling factor $\alpha$ per system from
    :func:`kups.md.integrators.csvr_scale_factor`, and multiplies both
    $\mathbf{p}_g$ and $\mathbf{L}_g$ by $\alpha$. The lab-frame multiplication
    of $\mathbf{L}_g$ is equivalent to scaling the body-frame components, so
    no quaternion conversion is needed.

    DOF should be supplied via :class:`HasDegreesOfFreedom` on the system
    and accounts for $6 N_{\text{nonlinear}} + 5 N_{\text{linear}} - 3$.
    """

    groups: Lens[State, Table[GroupId, _RigidCSVRData]] = field(static=True)
    systems: View[State, Table[SystemId, _RigidCSVRSystemData]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        groups_lens = self.groups.bind(state)
        groups = groups_lens.get()
        systems = self.systems(state)

        ke_per_group = per_group_kinetic_energy(
            groups.data.momenta,
            groups.data.masses,
            groups.data.angular_momentum,
            groups.data.quaternion,
            groups.data.inertia_diag,
        )
        ke_per_system = jax.ops.segment_sum(
            ke_per_group,
            groups.data.system.indices,
            groups.data.system.num_labels,
        )

        scale = csvr_scale_factor(
            key,
            kinetic_energy=ke_per_system,
            degrees_of_freedom=systems.data.degrees_of_freedom,
            target_thermal_energy=systems.data.temperature * BOLTZMANN_CONSTANT,
            timestep=systems.data.time_step,
            thermostat_timescale=systems.data.thermostat_time_constant,
        )
        scale_per_group = scale[groups.data.system.indices]

        new_p = groups.data.momenta * scale_per_group[..., None]
        new_l = groups.data.angular_momentum * scale_per_group[..., None]

        state = groups_lens.focus(lambda g: g.data.momenta).set(new_p)
        groups_lens = self.groups.bind(state)
        return groups_lens.focus(lambda g: g.data.angular_momentum).set(new_l)


@dataclass
class ForceAggregationStep[State](Propagator[State]):
    r"""Reduce per-atom forces into per-group net force and lab-frame torque.

    Reads atom-level position gradients $\nabla_{r_i} U$ and atom positions,
    plus group COM positions, and writes the per-group COM gradient and the
    per-group lab-frame torque. The COM gradient is stored on the group
    table under ``position_gradients`` (mirroring :class:`MDParticles`); the
    torque is stored under ``torque``.

    Uses minimum-image wrapping of $r_i - r_g^{\mathrm{COM}}$ so that
    molecules straddling a periodic boundary are handled correctly.
    """

    particles: Lens[State, Table[ParticleId, _ForceAggregationAtomData]] = field(
        static=True
    )
    groups: Lens[State, Table[GroupId, _ForceAggregationGroupData]] = field(static=True)
    systems: View[State, Table[SystemId, HasUnitCell]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        particles = self.particles.get(state)
        groups = self.groups.get(state)
        systems = self.systems(state)

        atom_unitcells = systems.data.unitcell[particles.data.system.indices]
        atom_forces = -particles.data.position_gradients
        com_force, torque = aggregate_forces(
            atom_forces=atom_forces,
            atom_positions=particles.data.positions,
            com_positions=groups.data.positions,
            group_idx=particles.data.group.indices,
            num_groups=particles.data.group.num_labels,
            atom_unitcells=atom_unitcells,
        )

        groups_lens = self.groups.bind(state)
        state = groups_lens.focus(lambda g: g.data.position_gradients).set(-com_force)
        groups_lens = self.groups.bind(state)
        return groups_lens.focus(lambda g: g.data.torque).set(torque)


@dataclass
class AtomReconstructionStep[State](Propagator[State]):
    r"""Place atoms in the lab frame from group COM, orientation, and motif geometry.

    For every atom $i$ belonging to group $g$ with motif site $m$:

    $$\mathbf{r}_i = \mathbf{r}_g^{\mathrm{COM}} + \mathbf{q}_g \star
       \mathbf{r}_m^{\mathrm{body}}$$

    No periodic wrapping is applied here: the COM integrator wraps the
    centre of mass each step, and atoms within a rigid molecule should
    never be wrapped independently.
    """

    particles: Lens[State, Table[ParticleId, _AtomReconstructionAtomData]] = field(
        static=True
    )
    groups: Lens[State, Table[GroupId, _AtomReconstructionGroupData]] = field(
        static=True
    )
    motifs: View[State, Table[MotifParticleId, _AtomReconstructionMotifData]] = field(
        static=True
    )

    def __call__(self, key: Array, state: State) -> State:
        del key
        particles = self.particles.get(state)
        groups = self.groups.get(state)
        motifs = self.motifs(state)

        new_positions = reconstruct_atom_positions(
            com_positions=groups.data.positions,
            quaternion=groups.data.quaternion,
            motif_positions=motifs.data.positions,
            group_idx=particles.data.group.indices,
            motif_idx=particles.data.motif.indices,
        )
        return self.particles.bind(state).focus(lambda p: p.data.positions).set(
            new_positions
        )


def make_rigid_velocity_verlet_step[State](
    particles: Lens[State, Table[ParticleId, _RigidAtomData]],
    groups: Lens[State, Table[GroupId, _RigidGroupData]],
    motifs: View[State, Table[MotifParticleId, _AtomReconstructionMotifData]],
    systems: View[State, Table[SystemId, _RigidVerletSysData]],
    derivative_computation: Propagator[State],
    flow: Flow[State, Array],
) -> SequentialPropagator[State]:
    r"""Velocity-Verlet for rigid bodies (NVE).

    Order:

    1. **B**: half-step COM-momentum kick.
    2. **B_rot**: half-step angular-momentum kick.
    3. **A**: full-step COM drift via :class:`PositionStep` (with PBC flow).
    4. **A_rot**: full-step quaternion drift via :class:`QuaternionDriftStep`.
    5. **R**: reconstruct atom positions from new COM and orientation.
    6. **F**: evaluate forces (atom-level potentials).
    7. **G**: aggregate atom forces to per-group COM force and torque.
    8. **B**: half-step COM kick.
    9. **B_rot**: half-step angular-momentum kick.
    """
    sys_half = pipe(systems, half_time)
    aggregation = ForceAggregationStep(particles, groups, systems)
    return SequentialPropagator(
        (
            MomentumStep(groups, sys_half),
            RotationalMomentumStep(groups, sys_half),
            PositionStep(groups, systems, flow),
            QuaternionDriftStep(groups, systems),
            AtomReconstructionStep(particles, groups, motifs),
            derivative_computation,
            aggregation,
            MomentumStep(groups, sys_half),
            RotationalMomentumStep(groups, sys_half),
        )
    )


def make_rigid_baoab_langevin_step[State](
    particles: Lens[State, Table[ParticleId, _RigidAtomData]],
    groups: Lens[State, Table[GroupId, _RigidGroupData]],
    motifs: View[State, Table[MotifParticleId, _AtomReconstructionMotifData]],
    systems: View[State, Table[SystemId, _RigidStochasticSysData]],
    derivative_computation: Propagator[State],
    flow: Flow[State, Array],
) -> SequentialPropagator[State]:
    r"""BAOAB Langevin for rigid bodies (NVT).

    Sequence ``B½ B_rot½ A½ A_rot½ O O_rot A½ A_rot½ R F G B½ B_rot½``.
    """
    sys_half = pipe(systems, half_time)
    aggregation = ForceAggregationStep(particles, groups, systems)
    return SequentialPropagator(
        (
            MomentumStep(groups, sys_half),
            RotationalMomentumStep(groups, sys_half),
            PositionStep(groups, sys_half, flow),
            QuaternionDriftStep(groups, sys_half),
            StochasticStep(groups, systems),
            RigidRotationalStochasticStep(groups, systems),
            PositionStep(groups, sys_half, flow),
            QuaternionDriftStep(groups, sys_half),
            AtomReconstructionStep(particles, groups, motifs),
            derivative_computation,
            aggregation,
            MomentumStep(groups, sys_half),
            RotationalMomentumStep(groups, sys_half),
        )
    )


def make_rigid_csvr_step[State](
    particles: Lens[State, Table[ParticleId, _RigidAtomData]],
    groups: Lens[State, Table[GroupId, _RigidGroupData]],
    motifs: View[State, Table[MotifParticleId, _AtomReconstructionMotifData]],
    systems: View[State, Table[SystemId, _RigidCSVRSystemData]],
    derivative_computation: Propagator[State],
    flow: Flow[State, Array],
) -> SequentialPropagator[State]:
    r"""CSVR + rigid velocity-Verlet (NVT).

    Prepends :class:`RigidCSVRStep` to the rigid NVE sequence.
    """
    sys_half = pipe(systems, half_time)
    aggregation = ForceAggregationStep(particles, groups, systems)
    return SequentialPropagator(
        (
            RigidCSVRStep(groups, systems),
            MomentumStep(groups, sys_half),
            RotationalMomentumStep(groups, sys_half),
            PositionStep(groups, systems, flow),
            QuaternionDriftStep(groups, systems),
            AtomReconstructionStep(particles, groups, motifs),
            derivative_computation,
            aggregation,
            MomentumStep(groups, sys_half),
            RotationalMomentumStep(groups, sys_half),
        )
    )


@dataclass
class RigidStochasticCellRescalingStep[State](Propagator[State]):
    r"""Bernetti–Bussi 2020 cell rescaling for rigid-body NPT.

    Differs from atomic :class:`StochasticCellRescalingStep` in two ways:

    1. **Translational KE only.** The pressure expression's kinetic term
       enters via the volume derivative of the partition function, which
       sees only translational DOF for rigid bodies; rotational rotations
       preserve volume and must not be counted. Computes $K_\mathrm{trans} =
       \tfrac{1}{2}\sum_g |\mathbf p_g^\mathrm{COM}|^2 / M_g$.
    2. **Molecular virial.** Uses :func:`molecular_stress_via_virial_theorem`
       (RASPA convention: virial taken with respect to group COMs), not the
       atomic virial.

    Scales the per-group COM positions by the same factor as the unit cell;
    atom positions are reconstructed downstream by :class:`AtomReconstructionStep`.
    """

    particles: Lens[State, Table[ParticleId, _RigidAtomData]] = field(static=True)
    groups: Lens[State, Table[GroupId, _RigidGroupData]] = field(static=True)
    systems: Lens[State, Table[SystemId, _RigidNPTSystemData]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        groups = self.groups.get(state)
        particles = self.particles.get(state)
        systems = self.systems.get(state)

        ke_trans = 0.5 * jnp.sum(groups.data.momenta**2, axis=-1) / groups.data.masses
        kinetic_energy = jax.ops.segment_sum(
            ke_trans,
            groups.data.system.indices,
            groups.data.system.num_labels,
        )
        cauchy_stress = molecular_stress_via_virial_theorem(
            particles, groups, systems
        ).data

        scaling_factor = stochastic_cell_rescaling_factor(
            key, kinetic_energy, cauchy_stress, systems.data
        )

        new_unitcell = systems.data.unitcell * scaling_factor
        state = self.systems.focus(lambda x: x.data.unitcell).set(state, new_unitcell)

        groups_lens = self.groups.bind(state)
        groups = groups_lens.get()
        scaling_per_group = scaling_factor[groups.data.system.indices]
        new_com = groups.data.positions * scaling_per_group[..., None]
        return groups_lens.focus(lambda g: g.data.positions).set(new_com)


def make_rigid_csvr_npt_step[State](
    particles: Lens[State, Table[ParticleId, _RigidAtomData]],
    groups: Lens[State, Table[GroupId, _RigidGroupData]],
    motifs: View[State, Table[MotifParticleId, _AtomReconstructionMotifData]],
    systems_lens: Lens[State, Table[SystemId, _RigidNPTSystemData]],
    derivative_computation: Propagator[State],
    flow: Flow[State, Array],
) -> SequentialPropagator[State]:
    r"""CSVR-NPT for rigid bodies.

    The NVT-CSVR rigid sequence followed by stochastic cell rescaling on
    the COM positions, atom reconstruction (since the COM moved), and a
    fresh force evaluation + aggregation (mirrors the trailing
    ``derivative_computation`` of the atomic ``make_csvr_npt_step``).
    """
    sys_view: View[State, Table[SystemId, _RigidNPTSystemData]] = systems_lens.get
    sys_half = pipe(sys_view, half_time)
    aggregation = ForceAggregationStep(particles, groups, systems_lens.get)
    cell_rescale = RigidStochasticCellRescalingStep(particles, groups, systems_lens)

    return SequentialPropagator(
        (
            RigidCSVRStep(groups, sys_view),
            MomentumStep(groups, sys_half),
            RotationalMomentumStep(groups, sys_half),
            PositionStep(groups, sys_view, flow),
            QuaternionDriftStep(groups, sys_view),
            AtomReconstructionStep(particles, groups, motifs),
            derivative_computation,
            aggregation,
            MomentumStep(groups, sys_half),
            RotationalMomentumStep(groups, sys_half),
            cell_rescale,
            AtomReconstructionStep(particles, groups, motifs),
            derivative_computation,
            aggregation,
        )
    )


class IsRigidMdState(Protocol):
    """Protocol for the state of a rigid-body MD simulation."""

    @property
    def particles(self) -> Table[ParticleId, _RigidAtomData]: ...
    @property
    def groups(self) -> Table[GroupId, _RigidGroupData]: ...
    @property
    def motifs(self) -> Table[MotifParticleId, _AtomReconstructionMotifData]: ...
    @property
    def systems(self) -> Table[SystemId, _RigidAnyIntegratorSysData]: ...


def make_rigid_md_step_from_state[State](
    state: Lens[State, IsRigidMdState],
    derivative_computation: Propagator[State],
    integrator: RigidIntegrator,
) -> Propagator[State]:
    """Build a single rigid-body MD step from a state lens.

    Mirrors :func:`kups.md.integrators.make_md_step_from_state`.

    Supported integrators:

    - ``"rigid_verlet"`` (NVE)
    - ``"rigid_baoab_langevin"`` (NVT)
    - ``"rigid_csvr"`` (NVT)
    - ``"rigid_csvr_npt"`` (NPT)
    """
    particles = state.focus(lambda x: x.particles)
    groups = state.focus(lambda x: x.groups)
    systems_lens = state.focus(lambda x: x.systems)
    motifs_view: View[
        State, Table[MotifParticleId, _AtomReconstructionMotifData]
    ] = lambda s: state.get(s).motifs

    flow = MinimumImageConventionFlow(
        lambda s: state.get(s).systems[state.get(s).groups.data.system].unitcell,
        euclidean_flow,
    )

    match integrator:
        case "rigid_verlet":
            return make_rigid_velocity_verlet_step(
                particles, groups, motifs_view, systems_lens.get,
                derivative_computation, flow,
            )
        case "rigid_baoab_langevin":
            return make_rigid_baoab_langevin_step(
                particles, groups, motifs_view, systems_lens.get,
                derivative_computation, flow,
            )
        case "rigid_csvr":
            return make_rigid_csvr_step(
                particles, groups, motifs_view, systems_lens.get,
                derivative_computation, flow,
            )
        case "rigid_csvr_npt":
            return make_rigid_csvr_npt_step(
                particles, groups, motifs_view, systems_lens,
                derivative_computation, flow,
            )
        case _:
            raise ValueError(f"Unknown rigid integrator: {integrator}")


# Re-export for callers
__all__ = [
    "RigidIntegrator",
    "QuaternionDriftStep",
    "RotationalMomentumStep",
    "RigidRotationalStochasticStep",
    "RigidCSVRStep",
    "RigidStochasticCellRescalingStep",
    "ForceAggregationStep",
    "AtomReconstructionStep",
    "IsRigidMdState",
    "make_rigid_velocity_verlet_step",
    "make_rigid_baoab_langevin_step",
    "make_rigid_csvr_step",
    "make_rigid_csvr_npt_step",
    "make_rigid_md_step_from_state",
]


# Bind/Index re-imports kept above for type clarity; suppress unused-warning.
_ = bind, Index
