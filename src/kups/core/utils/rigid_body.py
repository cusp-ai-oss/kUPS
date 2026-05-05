# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers for rigid-body kinematics.

These routines bridge atom-level potentials (which produce per-atom forces)
and per-group rigid-body integrators (which need net force, torque, and
atom positions reconstructed from the group COM and orientation).

All helpers are JIT-compatible. None of them store state — they operate on
plain arrays and quaternion components.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.unitcell import UnitCell
from kups.core.utils.quaternion import Quaternion


def aggregate_forces(
    atom_forces: Array,
    atom_positions: Array,
    com_positions: Array,
    group_idx: Array,
    num_groups: int,
    atom_unitcells: UnitCell,
) -> tuple[Array, Array]:
    r"""Reduce per-atom forces to per-group net force and lab-frame torque.

    For each group $g$, the net force is $\mathbf{F}_g = \sum_{i \in g} \mathbf{f}_i$
    and the lab-frame torque is $\boldsymbol{\tau}_g = \sum_{i \in g}
    (\mathbf{r}_i - \mathbf{r}_g^{\mathrm{COM}}) \times \mathbf{f}_i$, with the
    relative position taken under the minimum-image convention so that
    molecules straddling a periodic boundary are not split.

    Args:
        atom_forces: Per-atom forces, shape ``(n_atoms, 3)``.
        atom_positions: Per-atom positions, shape ``(n_atoms, 3)``.
        com_positions: Per-group COM positions, shape ``(n_groups, 3)``.
        group_idx: Index mapping each atom to its group, shape ``(n_atoms,)``.
        num_groups: Number of groups (segment count for ``segment_sum``).
        atom_unitcells: Unit cell per atom (typically ``unitcells[atom_system]``),
            used for MIC wrapping of $(r_i - r_g^{\mathrm{COM}})$.

    Returns:
        Tuple ``(com_force, torque)`` of shape ``(n_groups, 3)`` each.
    """
    rel = atom_unitcells.wrap(atom_positions - com_positions[group_idx])
    com_force = jax.ops.segment_sum(atom_forces, group_idx, num_groups)
    torque_per_atom = jnp.cross(rel, atom_forces)
    torque = jax.ops.segment_sum(torque_per_atom, group_idx, num_groups)
    return com_force, torque


def reconstruct_atom_positions(
    com_positions: Array,
    quaternion: Quaternion,
    motif_positions: Array,
    group_idx: Array,
    motif_idx: Array,
) -> Array:
    r"""Place atoms in the lab frame from group COM, orientation, and motif geometry.

    Each atom $i$ belonging to group $g$ with motif site $m$ is placed at

    $$\mathbf{r}_i = \mathbf{r}_g^{\mathrm{COM}} + \mathbf{q}_g \star \mathbf{r}_m^{\mathrm{body}}$$

    where ``motif_positions[m]`` is the body-frame coordinate of motif site $m$
    (with the motif's COM at the origin) and ``quaternion[g]`` is the rotation
    taking body-frame vectors into the lab frame.

    No periodic wrapping is applied: it is the responsibility of the COM
    integrator to keep the COM inside the primary cell. Atoms may stick out
    of the cell by up to the molecule's radius — distance computations use
    the minimum-image convention and handle that correctly.

    Args:
        com_positions: Per-group COM positions, shape ``(n_groups, 3)``.
        quaternion: Per-group orientation, batched shape ``(n_groups,)``.
        motif_positions: Body-frame positions of motif sites, shape
            ``(n_motif_sites, 3)``.
        group_idx: Index mapping each atom to its group, shape ``(n_atoms,)``.
        motif_idx: Index mapping each atom to its motif site, shape ``(n_atoms,)``.

    Returns:
        Lab-frame atom positions, shape ``(n_atoms, 3)``.
    """
    body = motif_positions[motif_idx]
    q_per_atom = Quaternion(quaternion.components[group_idx])
    rotated = body @ q_per_atom
    return com_positions[group_idx] + rotated


def inertia_tensor_diag(
    motif_positions: Array,
    motif_masses: Array,
) -> tuple[Array, Array]:
    r"""Diagonalise the inertia tensor of a single motif about its COM.

    Computes $I_{ab} = \sum_i m_i (|\mathbf{r}_i|^2 \delta_{ab} - r_{i,a} r_{i,b})$
    in the input frame, then performs a symmetric eigendecomposition.

    The caller is responsible for ensuring that ``motif_positions`` are
    expressed relative to the motif's COM (otherwise the result is not the
    physical inertia tensor). The returned eigenvectors form the rotation
    that takes a body-frame vector aligned with the input frame to one
    aligned with the principal axes; in practice you will rotate the motif
    positions by ``eigvecs.T`` before storing them.

    Args:
        motif_positions: Motif site positions in some Cartesian frame,
            shape ``(n_motif_sites, 3)``.
        motif_masses: Motif site masses, shape ``(n_motif_sites,)``.

    Returns:
        Tuple ``(eigvals, eigvecs)`` where ``eigvals`` are the principal
        moments (sorted ascending) and ``eigvecs`` are the corresponding
        column eigenvectors (so that ``eigvecs @ diag(eigvals) @ eigvecs.T``
        equals the original tensor).
    """
    r_sq = jnp.sum(motif_positions**2, axis=-1)
    weighted_r_sq = jnp.sum(motif_masses * r_sq)
    weighted_outer = jnp.sum(
        motif_masses[:, None, None]
        * motif_positions[:, :, None]
        * motif_positions[:, None, :],
        axis=0,
    )
    inertia = jnp.eye(3) * weighted_r_sq - weighted_outer
    return jnp.linalg.eigh(inertia)


def is_linear_motif(inertia_diag: Array, tol: float = 1e-6) -> bool:
    """Detect linear motifs from their principal moments of inertia.

    A linear molecule has one near-zero principal moment (the symmetry axis).
    The check uses the ratio of the smallest to the largest moment.

    Args:
        inertia_diag: Principal moments, shape ``(3,)``.
        tol: Tolerance on the smallest/largest ratio (default ``1e-6``).

    Returns:
        ``True`` if the motif is effectively linear.
    """
    smallest = float(jnp.min(inertia_diag))
    largest = float(jnp.max(inertia_diag))
    if largest <= 0.0:
        return True
    return smallest / largest < tol


def per_group_kinetic_energy(
    momenta: Array,
    masses: Array,
    angular_momentum: Array,
    quaternion: Quaternion,
    inertia_diag: Array,
) -> Array:
    r"""Translational + rotational kinetic energy per rigid group.

    $$K_g = \tfrac{1}{2}\,|\mathbf{p}_g|^2 / M_g
         + \tfrac{1}{2}\sum_a (L^{\mathrm{body}}_{g,a})^2 / I_{g,a}$$

    Body-frame angular momentum is recovered from $\mathbf{L}_{\mathrm{lab}}$ via
    $\mathbf{L}_{\mathrm{body}} = \mathbf{q}^{-1} \otimes \mathbf{L}_{\mathrm{lab}}$.
    Linear-motif symmetry axes (``inertia_diag = inf``) contribute zero.

    Args:
        momenta: COM momenta, shape ``(n_groups, 3)``.
        masses: Total group masses, shape ``(n_groups,)``.
        angular_momentum: Lab-frame angular momenta, shape ``(n_groups, 3)``.
        quaternion: Per-group orientation, batched ``(n_groups,)``.
        inertia_diag: Body-frame principal moments, shape ``(n_groups, 3)``.

    Returns:
        Per-group total kinetic energy, shape ``(n_groups,)``.
    """
    ke_trans = 0.5 * jnp.sum(momenta**2, axis=-1) / masses
    l_body = angular_momentum @ quaternion.inv()
    per_axis = jnp.where(
        jnp.isfinite(inertia_diag),
        l_body**2 / (2.0 * inertia_diag),
        0.0,
    )
    return ke_trans + jnp.sum(per_axis, axis=-1)


def initial_inertia_for_dynamics(
    inertia_diag: Array, *, linear_axis_tol: float = 1e-6
) -> Array:
    r"""Replace the symmetry-axis moment of a linear motif with ``inf``.

    For linear molecules, the inertia about the symmetry axis is zero, which
    makes $L^2 / I$ singular. Setting that moment to ``inf`` freezes the
    rotational DOF along that axis (any finite torque produces zero angular
    velocity), correctly accounting for the missing degree of freedom.

    Args:
        inertia_diag: Principal moments from :func:`inertia_tensor_diag`,
            shape ``(3,)``.
        linear_axis_tol: Tolerance for detecting the symmetry axis.

    Returns:
        Inertia diagonal with the symmetry-axis entry set to ``inf`` if the
        motif is linear; otherwise unchanged.
    """
    largest = jnp.max(inertia_diag)
    near_zero = inertia_diag < linear_axis_tol * jnp.maximum(largest, 1e-30)
    return jnp.where(near_zero, jnp.inf, inertia_diag)
