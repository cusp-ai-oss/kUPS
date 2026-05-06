# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Simulation cell representations for molecular simulations.

Splits cell description into two orthogonal axes:

- **Lattice geometry** — how the parallelepiped is parameterized.
  - [OrthogonalLattice][kups.core.cell.OrthogonalLattice]: 3 DOF (lengths)
  - [TriclinicLattice][kups.core.cell.TriclinicLattice]: 6 DOF (lower-triangular elements)

- **Boundary condition** — which axes are periodic.
  - [PeriodicCell][kups.core.cell.PeriodicCell]: all three axes periodic
    (literal ``(True, True, True)``).
  - [VacuumCell][kups.core.cell.VacuumCell]: all three axes open
    (literal ``(False, False, False)``).
  - [SlabCell][kups.core.cell.SlabCell]: runtime per-axis mask, e.g.
    ``(True, True, False)`` for a slab.

Geometry primitives (lattice vectors, volume, fractional/real conversion)
live on the [Lattice][kups.core.cell.Lattice] protocol. Boundary semantics
(wrap, supercell replication) live on the [Cell][kups.core.cell.Cell]
protocol, which is generic over both the lattice ``L`` and the periodicity
literal ``P``. This lets callers narrow on either axis independently — e.g.
an Ewald path can demand ``Cell[L, FullyPeriodic]`` regardless of geometry,
and an orthogonal-only path can demand ``Cell[OrthogonalLattice, P]``
regardless of periodicity.

Lattice vectors follow the row convention: ``r_real = r_frac @ lattice_vectors``.
"""

from __future__ import annotations

import math
from enum import Enum
from functools import partial
from typing import Any, Literal, Protocol, Self, TypeGuard, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.data import Sliceable
from kups.core.lens import Lens
from kups.core.utils.jax import dataclass, field
from kups.core.utils.math import triangular_3x3_det_and_inverse, triangular_3x3_matmul


class CoordinateSpace(Enum):
    """Enumeration for coordinate systems.

    Attributes:
        REAL: Cartesian coordinates in Angstroms.
        FRACTIONAL: Scaled coordinates in [0, 1) relative to lattice vectors.
    """

    REAL = "real"
    FRACTIONAL = "fractional"


type FullyPeriodic = tuple[Literal[True], Literal[True], Literal[True]]
type Vacuum = tuple[Literal[False], Literal[False], Literal[False]]
type AnyMask = tuple[bool, bool, bool]


class TriclinicMap(Protocol):
    """Mapping between an arbitrary frame and a triclinic frame."""

    def __call__(self, r: Array, /) -> Array: ...


@runtime_checkable
class Lattice(Protocol):
    """3D parallelepiped geometry, no periodicity attached.

    Concrete lattices are pure containers of geometric parameters: lattice
    vectors, volume, coordinate transforms. They know nothing about
    boundary conditions.
    """

    @property
    def lattice_vectors(self) -> Array: ...

    @property
    def inverse_lattice_vectors(self) -> Array: ...

    @property
    def volume(self) -> Array: ...

    @property
    def perpendicular_lengths(self) -> Array: ...

    def to_fractional(self, r: Array) -> Array: ...

    def to_real(self, r_frac: Array) -> Array: ...

    def scale(self, multiplicities: tuple[int, int, int]) -> Self: ...

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


def _perpendicular_lengths(lattice_vectors: Array, volume: Array) -> Array:
    """Compute perpendicular distances between opposing faces of the cell."""
    a = lattice_vectors[..., 0, :]
    b = lattice_vectors[..., 1, :]
    c = lattice_vectors[..., 2, :]
    Lx = volume / jnp.linalg.norm(jnp.cross(b, c), axis=-1)
    Ly = volume / jnp.linalg.norm(jnp.cross(a, c), axis=-1)
    Lz = volume / jnp.linalg.norm(jnp.cross(a, b), axis=-1)
    return jnp.stack([Lx, Ly, Lz], axis=-1)


@dataclass
class TriclinicLattice(Sliceable):
    """General triclinic lattice with 6 degrees of freedom.

    Stores the 6 independent elements of the lower-triangular lattice matrix.
    Lattice vectors are a linear function of these parameters, making them
    suitable for gradient-based optimization.

    Attributes:
        tril: Lower-triangular elements ``[L00, L10, L11, L20, L21, L22]``,
            shape ``(..., 6)``. The lattice matrix is::

                [[L00,   0,   0],
                 [L10, L11,   0],
                 [L20, L21, L22]]

    Example:
        ```python
        lat = TriclinicLattice.from_matrix(jnp.eye(3) * 10.0)
        cell = PeriodicCell(lat)
        cell.volume  # 1000.0
        ```
    """

    tril: Array

    @classmethod
    def from_matrix(cls, vecs: Array) -> TriclinicLattice:
        """Construct a triclinic lattice from a lower-triangular lattice matrix.

        Args:
            vecs: Lower-triangular lattice vectors as rows, shape ``(..., 3, 3)``.
        """
        vecs = jnp.asarray(vecs)
        return cls(vecs[..., *np.tril_indices(3)])

    @classmethod
    def from_lengths_and_angles(cls, lengths: Array, angles: Array) -> TriclinicLattice:
        """Construct a triclinic lattice from crystallographic parameters.

        Args:
            lengths: Lattice lengths ``[a, b, c]`` in Angstroms, shape ``(..., 3)``.
            angles: Lattice angles ``[alpha, beta, gamma]`` in degrees, shape ``(..., 3)``.
                alpha = angle(b, c), beta = angle(a, c), gamma = angle(a, b).
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

    def to_fractional(self, r: Array) -> Array:
        return triangular_3x3_matmul(self.inverse_lattice_vectors, r)

    def to_real(self, r_frac: Array) -> Array:
        return triangular_3x3_matmul(self.lattice_vectors, r_frac)

    def scale(self, multiplicities: tuple[int, int, int]) -> Self:
        m = jnp.asarray(multiplicities)
        scale_vec = jnp.array([m[0], m[1], m[1], m[2], m[2], m[2]])
        return type(self)(self.tril * scale_vec)

    def __mul__(self, other: Array | float | int) -> Self:
        return type(self)(self.tril * jnp.asarray(other)[..., None])


@dataclass
class OrthogonalLattice(Sliceable):
    """Orthogonal lattice with 3 degrees of freedom.

    Exploits the diagonal structure for cheaper volume, inverse, and coordinate
    transform operations compared to the general triclinic path.

    Attributes:
        lengths: Box side lengths ``[Lx, Ly, Lz]`` in Angstroms, shape ``(..., 3)``.

    Example:
        ```python
        lat = OrthogonalLattice(lengths=jnp.array([30., 30., 30.]))
        cell = PeriodicCell(lat)
        cell.volume  # 27000.0
        ```
    """

    lengths: Array

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

    def to_fractional(self, r: Array) -> Array:
        return r / self.lengths

    def to_real(self, r_frac: Array) -> Array:
        return r_frac * self.lengths

    def scale(self, multiplicities: tuple[int, int, int]) -> Self:
        return type(self)(self.lengths * jnp.asarray(multiplicities))

    def __mul__(self, other: Array | float | int) -> Self:
        return type(self)(self.lengths * jnp.asarray(other)[..., None])


def _wrap(
    lattice: Lattice,
    periodic: tuple[bool, bool, bool],
    r: Array,
    input_space: CoordinateSpace,
    output_space: CoordinateSpace,
) -> Array:
    """Fold coordinates into ``[-0.5, 0.5)`` along periodic axes."""
    frac = lattice.to_fractional(r) if input_space is CoordinateSpace.REAL else r
    wrapped = (frac + 0.5) % 1 - 0.5
    mask = jnp.array(periodic)
    out = jnp.where(mask, wrapped, frac)
    return lattice.to_real(out) if output_space is CoordinateSpace.REAL else out


@runtime_checkable
class Cell[L: Lattice, P: tuple[bool, bool, bool]](Protocol):
    """Lattice geometry with per-axis boundary semantics.

    Generic over both the lattice ``L`` (orthogonal vs triclinic) and the
    periodicity literal ``P``. The two type parameters narrow independently:
    a function may demand any combination such as
    ``Cell[OrthogonalLattice, FullyPeriodic]``,
    ``Cell[L, FullyPeriodic]``, or ``Cell[OrthogonalLattice, P]``.
    """

    @property
    def lattice(self) -> L: ...

    @property
    def periodic(self) -> P: ...

    @property
    def lattice_vectors(self) -> Array: ...

    @property
    def inverse_lattice_vectors(self) -> Array: ...

    @property
    def volume(self) -> Array: ...

    @property
    def perpendicular_lengths(self) -> Array: ...

    def wrap(
        self,
        r: Array,
        *,
        input_space: CoordinateSpace = CoordinateSpace.REAL,
        output_space: CoordinateSpace = CoordinateSpace.REAL,
    ) -> Array: ...

    def __mul__(self, other: Array | float | int) -> Self: ...

    def __getitem__(self, index: Any) -> Self: ...


@dataclass
class PeriodicCell[L: Lattice](Sliceable):
    """Cell that is periodic along all three axes.

    The ``periodic`` field is pinned to the literal type ``FullyPeriodic``,
    so a function that requires ``Cell[L, FullyPeriodic]`` accepts
    ``PeriodicCell[L]`` and rejects ``VacuumCell[L]`` or ``SlabCell[L]``
    statically.
    """

    lattice: L
    periodic: FullyPeriodic = field(default=(True, True, True), static=True)

    @property
    def lattice_vectors(self) -> Array:
        return self.lattice.lattice_vectors

    @property
    def inverse_lattice_vectors(self) -> Array:
        return self.lattice.inverse_lattice_vectors

    @property
    def volume(self) -> Array:
        return self.lattice.volume

    @property
    def perpendicular_lengths(self) -> Array:
        return self.lattice.perpendicular_lengths

    def wrap(
        self,
        r: Array,
        *,
        input_space: CoordinateSpace = CoordinateSpace.REAL,
        output_space: CoordinateSpace = CoordinateSpace.REAL,
    ) -> Array:
        return _wrap(self.lattice, self.periodic, r, input_space, output_space)

    def __mul__(self, other: Array | float | int) -> Self:
        return type(self)(self.lattice * other)


@dataclass
class VacuumCell[L: Lattice](Sliceable):
    """Cell with all three axes open (no periodicity).

    The ``periodic`` field is pinned to the literal type ``Vacuum``.
    """

    lattice: L
    periodic: Vacuum = field(default=(False, False, False), static=True)

    @property
    def lattice_vectors(self) -> Array:
        return self.lattice.lattice_vectors

    @property
    def inverse_lattice_vectors(self) -> Array:
        return self.lattice.inverse_lattice_vectors

    @property
    def volume(self) -> Array:
        return self.lattice.volume

    @property
    def perpendicular_lengths(self) -> Array:
        return self.lattice.perpendicular_lengths

    def wrap(
        self,
        r: Array,
        *,
        input_space: CoordinateSpace = CoordinateSpace.REAL,
        output_space: CoordinateSpace = CoordinateSpace.REAL,
    ) -> Array:
        return _wrap(self.lattice, self.periodic, r, input_space, output_space)

    def __mul__(self, other: Array | float | int) -> Self:
        return type(self)(self.lattice * other)


@dataclass
class SlabCell[L: Lattice](Sliceable):
    """Cell with a runtime per-axis periodicity mask.

    Use this when periodicity varies per axis at runtime (true slabs with
    one open axis, 1D wires, mixed-domain geometries). For statically known
    fully-periodic or fully-vacuum systems, prefer ``PeriodicCell`` or
    ``VacuumCell`` so the type system can discriminate.
    """

    lattice: L
    periodic: AnyMask = field(static=True)

    @property
    def lattice_vectors(self) -> Array:
        return self.lattice.lattice_vectors

    @property
    def inverse_lattice_vectors(self) -> Array:
        return self.lattice.inverse_lattice_vectors

    @property
    def volume(self) -> Array:
        return self.lattice.volume

    @property
    def perpendicular_lengths(self) -> Array:
        return self.lattice.perpendicular_lengths

    def wrap(
        self,
        r: Array,
        *,
        input_space: CoordinateSpace = CoordinateSpace.REAL,
        output_space: CoordinateSpace = CoordinateSpace.REAL,
    ) -> Array:
        return _wrap(self.lattice, self.periodic, r, input_space, output_space)

    def __mul__(self, other: Array | float | int) -> Self:
        return type(self)(self.lattice * other, self.periodic)


def min_multiplicity(cell: Cell, cutoff: float | Array) -> Array:
    """Minimum supercell replication per axis for a given cutoff.

    Returns 1 for non-periodic axes (no replication needed).
    """
    computed = jnp.ceil(2 * cutoff / cell.lattice.perpendicular_lengths).astype(int)
    mask = jnp.array(cell.periodic)
    return jnp.where(mask, computed, 1)


def make_supercell[T, T2, C: Cell](
    cell: C,
    multiplicities: tuple[int, int, int] | int,
    to_replicate: T,
    to_shift: Lens[T, T2],
) -> tuple[C, T]:
    """Replicate a cell along each periodic axis.

    Tiles the cell according to ``multiplicities`` (clamped to 1 on
    non-periodic axes), replicates the data, and shifts coordinates into
    the expanded cell using periodic wrapping. The returned cell has the
    same concrete type as the input (``PeriodicCell``/``VacuumCell``/``SlabCell``).
    """
    if isinstance(multiplicities, int):
        multiplicities = (multiplicities, multiplicities, multiplicities)
    assert len(multiplicities) == 3
    assert all(m > 0 for m in multiplicities)

    clamped: tuple[int, int, int] = (
        multiplicities[0] if cell.periodic[0] else 1,
        multiplicities[1] if cell.periodic[1] else 1,
        multiplicities[2] if cell.periodic[2] else 1,
    )

    n_reps = math.prod(clamped)
    shifts = jnp.stack(
        jnp.meshgrid(*[jnp.arange(m) for m in clamped]), axis=-1
    ).reshape(-1, 3)
    real_shifts = triangular_3x3_matmul(cell.lattice.lattice_vectors, shifts)

    new_lattice = cell.lattice.scale(clamped)
    new_cell_inst: PeriodicCell | VacuumCell | SlabCell
    if isinstance(cell, PeriodicCell):
        new_cell_inst = PeriodicCell(new_lattice)
    elif isinstance(cell, VacuumCell):
        new_cell_inst = VacuumCell(new_lattice)
    elif isinstance(cell, SlabCell):
        new_cell_inst = SlabCell(new_lattice, cell.periodic)
    else:
        msg = f"Unsupported cell type: {type(cell)}"
        raise TypeError(msg)
    new_cell: C = new_cell_inst  # pyright: ignore[reportAssignmentType]

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


def is_vacuum[L: Lattice, P: tuple[bool, bool, bool]](
    cell: Cell[L, P],
) -> TypeGuard[VacuumCell[L]]:
    """``True`` iff ``cell`` is a [VacuumCell][kups.core.cell.VacuumCell]."""
    return isinstance(cell, VacuumCell)


def is_fully_periodic[L: Lattice, P: tuple[bool, bool, bool]](
    cell: Cell[L, P],
) -> TypeGuard[PeriodicCell[L]]:
    """``True`` iff ``cell`` is a [PeriodicCell][kups.core.cell.PeriodicCell]."""
    return isinstance(cell, PeriodicCell)


def to_lower_triangular(vecs: Array) -> tuple[Array, TriclinicMap]:
    """Convert arbitrary lattice vectors to lower-triangular form via QR decomposition.

    Decomposes the input into a lower-triangular matrix (the canonical lattice
    representation) and an orthogonal rotation that maps coordinates from the
    original frame into the triclinic frame.

    Args:
        vecs: Lattice vectors as rows of a 3x3 matrix, shape ``(3, 3)``.

    Returns:
        Tuple of (lower_triangular_vectors, coordinate_rotation_fn):
            - lower_triangular_vectors: Lower-triangular 3x3 lattice matrix.
            - coordinate_rotation_fn: Maps ``(..., 3)`` positions from the
              original frame to the triclinic frame.
    """
    vecs = jnp.asarray(vecs)
    Q, L = jnp.linalg.qr(vecs.T)
    Q, L = Q.T, L.T
    return L, partial(jnp.einsum, "...ij,...i->...j", Q)
