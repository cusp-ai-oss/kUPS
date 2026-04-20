# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Periodic unit cell representations for molecular simulations.

Defines a [UnitCell][kups.core.unitcell.UnitCell] protocol and two concrete
implementations with minimal stored data:

- [TriclinicUnitCell][kups.core.unitcell.TriclinicUnitCell]: 6 DOF (lower-triangular elements)
- [OrthorhombicUnitCell][kups.core.unitcell.OrthorhombicUnitCell]: 3 DOF (lengths)

All derived quantities (lattice vectors, volume, inverse) are computed on demand
from the stored parameters. Lattice vectors follow row convention:
`r_real = r_frac @ lattice_vectors`.
"""

from __future__ import annotations

import math
from enum import Enum
from functools import partial
from typing import Any, Protocol, Self, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.data import Sliceable
from kups.core.lens import Lens
from kups.core.utils.jax import dataclass
from kups.core.utils.math import triangular_3x3_det_and_inverse, triangular_3x3_matmul


class CoordinateSpace(Enum):
    """Enumeration for coordinate systems.

    Attributes:
        REAL: Cartesian coordinates in Angstroms
        FRACTIONAL: Scaled coordinates in [0, 1) relative to lattice vectors
    """

    REAL = "real"
    FRACTIONAL = "fractional"


class TriclinicMap(Protocol):
    """Mapping between orthogonal and triclinic coordinate frames."""

    def __call__(self, r: Array, /) -> Array: ...


@runtime_checkable
class BoundaryCondition(Protocol):
    """Trait: knows how to handle spatial boundaries.

    Any simulation domain satisfies this — from vacuum (identity wrap)
    to fully periodic unit cells. Code that only needs wrapping accepts this.
    """

    @property
    def periodic(self) -> tuple[bool, bool, bool]: ...

    def wrap(
        self,
        r: Array,
        *,
        input_space: CoordinateSpace = CoordinateSpace.REAL,
        output_space: CoordinateSpace = CoordinateSpace.REAL,
    ) -> Array: ...


@runtime_checkable
class UnitCell(BoundaryCondition, Protocol):
    """3D lattice geometry with boundary behavior. Extends BoundaryCondition.

    Adds lattice vectors, volume, and other geometric properties to the
    base boundary condition trait. Ewald, neighbor lists, and stress
    calculations require this.
    """

    @property
    def lattice_vectors(self) -> Array: ...

    @property
    def inverse_lattice_vectors(self) -> Array: ...

    @property
    def volume(self) -> Array: ...

    @property
    def perpendicular_lengths(self) -> Array: ...

    def __mul__(self, other: Array | float | int) -> Self: ...

    def __getitem__(self, index: Any) -> Self: ...


def _build_lattice_vectors(lengths: Array, angles: Array) -> Array:
    """Construct a lower-triangular 3x3 lattice matrix from crystallographic parameters.

    Uses the standard crystallographic convention where the first vector lies
    along x, the second in the xy-plane, and the third completes the cell.

    Args:
        lengths: Lattice lengths [a, b, c] in Angstroms, shape `(..., 3)`.
        angles: Lattice angles [alpha, beta, gamma] in degrees, shape `(..., 3)`.

    Returns:
        Lower-triangular lattice matrix of shape `(..., 3, 3)`.
    """
    a, b, c = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    alpha_rad, beta_rad, gamma_rad = (
        jnp.radians(angles[..., 0]),
        jnp.radians(angles[..., 1]),
        jnp.radians(angles[..., 2]),
    )

    cos_a, cos_b, cos_g = jnp.cos(alpha_rad), jnp.cos(beta_rad), jnp.cos(gamma_rad)
    sin_g = jnp.sin(gamma_rad)

    c2z = (
        c
        * jnp.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g)
        / sin_g
    )

    zero = jnp.zeros_like(a)
    return jnp.stack(
        [
            jnp.stack([a, zero, zero], axis=-1),
            jnp.stack([b * cos_g, b * sin_g, zero], axis=-1),
            jnp.stack([c * cos_b, c * (cos_a - cos_b * cos_g) / sin_g, c2z], axis=-1),
        ],
        axis=-2,
    )


def _wrap(
    lattice_vectors: Array,
    inverse_lattice_vectors: Array,
    r: Array,
    input_space: CoordinateSpace,
    output_space: CoordinateSpace,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> Array:
    """Wrap coordinates into the primary cell `[-0.5, 0.5)` in fractional space.

    Only wraps axes where ``periodic`` is True; non-periodic axes pass through.

    Args:
        lattice_vectors: Lower-triangular lattice matrix, shape `(..., 3, 3)`.
        inverse_lattice_vectors: Inverse of lattice matrix, shape `(..., 3, 3)`.
        r: Coordinates to wrap, shape `(..., 3)`.
        input_space: Coordinate system of the input.
        output_space: Coordinate system for the output.
        periodic: Per-axis periodicity mask.

    Returns:
        Wrapped coordinates, shape `(..., 3)`.
    """
    if input_space is CoordinateSpace.REAL:
        frac = triangular_3x3_matmul(inverse_lattice_vectors, r)
    else:
        frac = r
    wrapped = (frac + 0.5) % 1 - 0.5
    mask = jnp.array(periodic)
    out = jnp.where(mask, wrapped, frac)
    if output_space is CoordinateSpace.REAL:
        out = triangular_3x3_matmul(lattice_vectors, out)
    return out


def _perpendicular_lengths(lattice_vectors: Array, volume: Array) -> Array:
    """Compute perpendicular distances between opposing faces of the unit cell.

    For each axis, the perpendicular length is `V / |cross(v_j, v_k)|` where
    `v_j, v_k` are the other two lattice vectors.

    Args:
        lattice_vectors: Lower-triangular lattice matrix, shape `(..., 3, 3)`.
        volume: Cell volume, shape `(...)`.

    Returns:
        Perpendicular lengths `[Lx, Ly, Lz]`, shape `(..., 3)`.
    """
    a = lattice_vectors[..., 0, :]
    b = lattice_vectors[..., 1, :]
    c = lattice_vectors[..., 2, :]
    Lx = volume / jnp.linalg.norm(jnp.cross(b, c), axis=-1)
    Ly = volume / jnp.linalg.norm(jnp.cross(a, c), axis=-1)
    Lz = volume / jnp.linalg.norm(jnp.cross(a, b), axis=-1)
    return jnp.stack([Lx, Ly, Lz], axis=-1)


def _multiply[T, T2, Cell: UnitCell](
    cell: Cell,
    make_scaled: _MakeScaled[Cell],
    multiplicities: tuple[int, int, int] | int,
    to_replicate: T,
    to_shift: Lens[T, T2],
) -> tuple[Cell, T]:
    """Create a supercell by replicating the unit cell along each axis.

    Tiles the cell according to `multiplicities`, replicates the data, and
    shifts coordinates into the expanded cell using periodic wrapping.

    Args:
        cell: Unit cell to replicate.
        make_scaled: Factory that produces a scaled cell from multiplicities.
        multiplicities: Replication counts `(nx, ny, nz)` or a single int.
        to_replicate: Data to replicate (e.g., positions).
        to_shift: Lens focusing on coordinates to shift during replication.

    Returns:
        Tuple of (scaled_cell, replicated_data).
    """
    if isinstance(multiplicities, int):
        multiplicities = (multiplicities, multiplicities, multiplicities)
    assert len(multiplicities) == 3
    assert all(m > 0 for m in multiplicities)

    n_reps = math.prod(multiplicities)
    shifts = jnp.stack(
        jnp.meshgrid(*[jnp.arange(m) for m in multiplicities]), axis=-1
    ).reshape(-1, 3)
    real_shifts = triangular_3x3_matmul(cell.lattice_vectors, shifts)

    new_cell = make_scaled(multiplicities)

    replicated = jax.tree.map(
        lambda x: jnp.repeat(x[None], n_reps, axis=0).reshape(-1, *x.shape[1:]),
        to_replicate,
    )
    replicated = to_shift.apply(
        replicated,
        lambda y: jax.tree.map(
            lambda x: new_cell.wrap(
                x + real_shifts.repeat(x.shape[0] // n_reps, axis=0).reshape(-1, 3)
            ),
            y,
        ),
    )
    return new_cell, replicated


class _MakeScaled[Cell](Protocol):
    def __call__(self, multiplicities: tuple[int, int, int]) -> Cell: ...


@dataclass
class TriclinicUnitCell(Sliceable):
    """General triclinic unit cell with 6 degrees of freedom.

    Stores the 6 independent elements of the lower-triangular lattice matrix.
    Lattice vectors are a linear function of these parameters, making them
    suitable for gradient-based optimization.

    Attributes:
        tril: Lower-triangular elements `[L00, L10, L11, L20, L21, L22]`,
            shape `(..., 6)`. The lattice matrix is::

                [[L00,   0,   0],
                 [L10, L11,   0],
                 [L20, L21, L22]]

    Example:
        ```python
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3) * 10.0)
        cell.volume  # 1000.0
        cell.wrap(positions)  # enforce periodic boundaries
        ```
    """

    tril: Array

    @property
    def periodic(self) -> tuple[bool, bool, bool]:
        return (True, True, True)

    @classmethod
    def from_matrix(cls, vecs: Array) -> TriclinicUnitCell:
        """Construct from a lower-triangular lattice matrix.

        Extracts the 6 independent elements from the lower triangle.

        Args:
            vecs: Lower-triangular lattice vectors as rows, shape `(..., 3, 3)`.

        Returns:
            TriclinicUnitCell with the 6 independent elements.
        """
        vecs = jnp.asarray(vecs)
        return cls(vecs[..., *np.tril_indices(3)])

    @classmethod
    def from_lengths_and_angles(
        cls, lengths: Array, angles: Array
    ) -> TriclinicUnitCell:
        """Construct from crystallographic parameters.

        Builds the lower-triangular lattice matrix from lengths and angles,
        then stores the 6 independent elements.

        Args:
            lengths: Lattice lengths `[a, b, c]` in Angstroms, shape `(..., 3)`.
            angles: Lattice angles `[alpha, beta, gamma]` in degrees, shape `(..., 3)`.
                alpha = angle(b, c), beta = angle(a, c), gamma = angle(a, b).

        Returns:
            TriclinicUnitCell with tril derived from the parameters.
        """
        return cls.from_matrix(_build_lattice_vectors(lengths, angles))

    @property
    def lattice_vectors(self) -> Array:
        zero = jnp.zeros_like(self.tril[..., :1])
        return jnp.stack(
            [
                jnp.concatenate([self.tril[..., 0:1], zero, zero], axis=-1),
                jnp.concatenate([self.tril[..., 1:3], zero], axis=-1),
                self.tril[..., 3:6],
            ],
            axis=-2,
        )

    @property
    def inverse_lattice_vectors(self) -> Array:
        return triangular_3x3_det_and_inverse(self.lattice_vectors)[1]

    @property
    def volume(self) -> Array:
        return jnp.abs(self.tril[..., 0] * self.tril[..., 2] * self.tril[..., 5])

    @property
    def lengths(self) -> Array:
        return jnp.linalg.norm(self.lattice_vectors, axis=-1)

    @property
    def angles(self) -> Array:
        lv = self.lattice_vectors
        a, b, c = lv[..., 0, :], lv[..., 1, :], lv[..., 2, :]
        la, lb, lc = (
            jnp.linalg.norm(a, axis=-1),
            jnp.linalg.norm(b, axis=-1),
            jnp.linalg.norm(c, axis=-1),
        )
        cos_alpha = jnp.clip(jnp.sum(b * c, axis=-1) / (lb * lc), -1.0, 1.0)
        cos_beta = jnp.clip(jnp.sum(a * c, axis=-1) / (la * lc), -1.0, 1.0)
        cos_gamma = jnp.clip(jnp.sum(a * b, axis=-1) / (la * lb), -1.0, 1.0)
        return jnp.degrees(
            jnp.stack(
                [jnp.arccos(cos_alpha), jnp.arccos(cos_beta), jnp.arccos(cos_gamma)],
                axis=-1,
            )
        )

    @property
    def perpendicular_lengths(self) -> Array:
        return _perpendicular_lengths(self.lattice_vectors, self.volume)

    def wrap(
        self,
        r: Array,
        *,
        input_space: CoordinateSpace = CoordinateSpace.REAL,
        output_space: CoordinateSpace = CoordinateSpace.REAL,
    ) -> Array:
        return _wrap(
            self.lattice_vectors,
            self.inverse_lattice_vectors,
            r,
            input_space,
            output_space,
            self.periodic,
        )

    def __mul__(self, other: Array | float | int) -> Self:
        scaled_tril = self.tril * jnp.asarray(other)[..., None]
        return type(self)(scaled_tril)


@dataclass
class OrthorhombicUnitCell(Sliceable):
    """Orthogonal unit cell with 3 degrees of freedom.

    Exploits the diagonal structure for cheaper volume, inverse, and wrap
    operations compared to the general triclinic path.

    Attributes:
        lengths: Box side lengths `[Lx, Ly, Lz]` in Angstroms, shape `(..., 3)`.

    Example:
        ```python
        cell = OrthorhombicUnitCell(lengths=jnp.array([30., 30., 30.]))
        cell.volume  # 27000.0
        ```
    """

    lengths: Array

    @property
    def periodic(self) -> tuple[bool, bool, bool]:
        return (True, True, True)

    @property
    def lattice_vectors(self) -> Array:
        return self.lengths[..., :, None] * jnp.eye(3)

    @property
    def inverse_lattice_vectors(self) -> Array:
        return (1.0 / self.lengths)[..., :, None] * jnp.eye(3)

    @property
    def volume(self) -> Array:
        return jnp.prod(self.lengths, axis=-1)

    @property
    def perpendicular_lengths(self) -> Array:
        return self.lengths

    def wrap(
        self,
        r: Array,
        *,
        input_space: CoordinateSpace = CoordinateSpace.REAL,
        output_space: CoordinateSpace = CoordinateSpace.REAL,
    ) -> Array:
        if input_space is CoordinateSpace.REAL:
            frac = r / self.lengths
        else:
            frac = r
        wrapped = (frac + 0.5) % 1 - 0.5
        mask = jnp.array(self.periodic)
        out = jnp.where(mask, wrapped, frac)
        if output_space is CoordinateSpace.REAL:
            out = out * self.lengths
        return out

    def __mul__(self, other: Array | float | int) -> Self:
        scaled_lengths = self.lengths * jnp.asarray(other)[..., None]
        return type(self)(scaled_lengths)


@dataclass
class Vacuum:
    """Non-periodic domain with no lattice geometry.

    Satisfies BoundaryCondition but not UnitCell. Wrap is identity.
    """

    @property
    def periodic(self) -> tuple[bool, bool, bool]:
        return (False, False, False)

    def wrap(
        self,
        r: Array,
        *,
        input_space: CoordinateSpace = CoordinateSpace.REAL,
        output_space: CoordinateSpace = CoordinateSpace.REAL,
    ) -> Array:
        del input_space, output_space
        return r


def min_multiplicity(cell: UnitCell, cutoff: float | Array) -> Array:
    """Minimum supercell replication per axis for a given cutoff.

    Returns 1 for non-periodic axes (no replication needed).
    """
    computed = jnp.ceil(2 * cutoff / cell.perpendicular_lengths).astype(int)
    mask = jnp.array(cell.periodic)
    return jnp.where(mask, computed, 1)


def make_supercell[T, T2, Cell: UnitCell](
    cell: Cell,
    multiplicities: tuple[int, int, int] | int,
    to_replicate: T,
    to_shift: Lens[T, T2],
) -> tuple[Cell, T]:
    """Replicate a unit cell, clamping non-periodic axes to 1."""
    if isinstance(multiplicities, int):
        multiplicities = (multiplicities, multiplicities, multiplicities)
    clamped: tuple[int, int, int] = (
        multiplicities[0] if cell.periodic[0] else 1,
        multiplicities[1] if cell.periodic[1] else 1,
        multiplicities[2] if cell.periodic[2] else 1,
    )

    def make_scaled(multiplicities: tuple[int, int, int]) -> Cell:
        if isinstance(cell, TriclinicUnitCell):
            m = jnp.asarray(multiplicities)
            scale = jnp.array([m[0], m[1], m[1], m[2], m[2], m[2]])
            return TriclinicUnitCell(  # pyright: ignore[reportReturnType]
                cell.tril * scale
            )
        if isinstance(cell, OrthorhombicUnitCell):
            return OrthorhombicUnitCell(  # pyright: ignore[reportReturnType]
                cell.lengths * jnp.asarray(multiplicities)
            )
        msg = f"Unsupported cell type: {type(cell)}"
        raise TypeError(msg)

    return _multiply(cell, make_scaled, clamped, to_replicate, to_shift)


def to_lower_triangular(vecs: Array) -> tuple[Array, TriclinicMap]:
    """Convert arbitrary lattice vectors to lower-triangular form via QR decomposition.

    Decomposes the input into a lower-triangular matrix (the canonical cell
    representation) and an orthogonal rotation that maps coordinates from the
    original frame into the triclinic frame.

    Args:
        vecs: Lattice vectors as rows of a 3x3 matrix, shape `(3, 3)`.

    Returns:
        Tuple of (lower_triangular_vectors, coordinate_rotation_fn):
            - lower_triangular_vectors: Lower-triangular 3x3 lattice matrix.
            - coordinate_rotation_fn: Maps `(..., 3)` positions from the
              original frame to the triclinic frame.
    """
    vecs = jnp.asarray(vecs)
    Q, L = jnp.linalg.qr(vecs.T)
    Q, L = Q.T, L.T
    return L, partial(jnp.einsum, "...ij,...i->...j", Q)
