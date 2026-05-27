# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Simulation cell representations.

This module separates two concepts that other simulation codes usually
conflate into one "cell" object:

1. **[Frame][kups.core.cell.Frame]** — pure geometry. A 3D parallelepiped
   defined by three basis vectors. No periodicity, no boundary semantics.
   What ASE calls ``cell``, OpenMM calls ``boxVectors``, LAMMPS calls the
   simulation box; in crystallography the basis vectors of a periodic
   structure are conventionally called *lattice vectors*. ``Frame``
   subsumes all of these — a Frame just describes a parallelepiped, and
   the meaning (periodic-translation vector vs bounding-box edge) is
   supplied by the cell type that wraps it.

2. **[Cell][kups.core.cell.Cell]** — frame plus per-axis boundary
   semantics. Decides whether the frame is interpreted as a periodic
   unit cell or a bounding domain on each axis.

Why split: the same parallelepiped means different things in different
contexts. A 30 Å cubic frame inside a [PeriodicCell][kups.core.cell.PeriodicCell]
is a periodic unit cell — particles wrap, neighbor searches honour minimum
image. The same frame inside a [VacuumCell][kups.core.cell.VacuumCell] is
the bounding box of a finite domain — no wrapping, neighbor searches treat
it as a spatial-partitioning hint. Naming the geometry "Lattice" or "Cell"
would prejudice the reading toward the periodic case; ``Frame`` is
boundary-condition-agnostic and reads equally honestly in both.

## Frame implementations

- [OrthogonalFrame][kups.core.cell.OrthogonalFrame] — 3 DOF, axes-aligned
  parallelepiped parameterized by side lengths. Diagonal fast paths for
  volume, inverse, and coordinate transforms.
- [TriclinicFrame][kups.core.cell.TriclinicFrame] — 6 DOF, general
  parallelepiped parameterized by the lower-triangular elements of the
  basis matrix.
- [MaterializedFrame][kups.core.cell.MaterializedFrame] — caches
  ``vectors``, ``inverse_vectors``, and ``volume`` as concrete arrays.
  Produced by [`Frame.materialize`][kups.core.cell.Frame.materialize];
  useful when the same frame is queried many times or when the inverse
  needs to be cached across a JIT boundary.

Both expose [`vectors`][kups.core.cell.Frame.vectors] (the basis matrix —
this is what crystallography calls *lattice vectors*),
[`inverse_vectors`][kups.core.cell.Frame.inverse_vectors],
[`volume`][kups.core.cell.Frame.volume],
[`perpendicular_lengths`][kups.core.cell.Frame.perpendicular_lengths],
[`to_fractional`][kups.core.cell.Frame.to_fractional] /
[`to_real`][kups.core.cell.Frame.to_real] coordinate transforms, plus
[`tile`][kups.core.cell.Frame.tile] for per-axis multiplicity tiling and
``__mul__`` for uniform scaling.

## Cell implementations

- [PeriodicCell][kups.core.cell.PeriodicCell] — all three axes periodic
  (literal ``(True, True, True)``). The frame is the unit cell of a
  periodic crystal or fluid.
- [VacuumCell][kups.core.cell.VacuumCell] — all three axes open (literal
  ``(False, False, False)``). The frame is the bounding parallelepiped
  of a finite simulation domain.

[Cell][kups.core.cell.Cell] is generic over the periodicity literal ``P``
so consumers can narrow statically on the boundary axis. An Ewald path
that requires periodic boundaries declares ``Cell[Periodic3D]`` and
pyright rejects ``VacuumCell`` at the call site:

```python
def ewald(cell: Cell[Periodic3D]) -> Energy: ...
ewald(PeriodicCell(frame))   # OK
ewald(VacuumCell(frame))     # pyright: "Vacuum is not assignable to Periodic3D"
```

Cell re-exposes the frame's geometric properties as passthrough so callers
can write ``cell.volume``, ``cell.vectors`` etc. without going through
``cell.frame``. For frame-specific fields (``OrthogonalFrame.lengths``,
``TriclinicFrame.tril`` / ``angles``), narrow with isinstance:

```python
frame = cell.frame
assert isinstance(frame, OrthogonalFrame)
side_lengths = frame.lengths
```

The same Frame instance can be wrapped in either cell type:

```python
frame = OrthogonalFrame(lengths=jnp.array([20., 20., 20.]))
PeriodicCell(frame)   # 20 Å cubic crystal — particles wrap
VacuumCell(frame)     # 20 Å bounding box of a cluster — no wrap
```

## Convention

Frame vectors follow the row convention: ``r_real = r_frac @ frame.vectors``.
"""

from __future__ import annotations

import math
from enum import Enum
from functools import partial
from typing import Any, Literal, Protocol, Self, TypeGuard, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.data import Sliceable
from kups.core.lens import Lens, bind
from kups.core.utils.jax import dataclass, field
from kups.core.utils.math import triangular_3x3_det_and_inverse, triangular_3x3_matmul


class CoordinateSpace(Enum):
    """Enumeration for coordinate systems.

    Attributes:
        REAL: Cartesian coordinates in Angstroms.
        FRACTIONAL: Scaled coordinates in [0, 1) relative to frame vectors.
    """

    REAL = "real"
    FRACTIONAL = "fractional"


type Periodic3D = tuple[Literal[True], Literal[True], Literal[True]]
type Vacuum = tuple[Literal[False], Literal[False], Literal[False]]

type SlabXY = tuple[Literal[True], Literal[True], Literal[False]]
type SlabXZ = tuple[Literal[True], Literal[False], Literal[True]]
type SlabYZ = tuple[Literal[False], Literal[True], Literal[True]]
type Slab2D = SlabXY | SlabXZ | SlabYZ

type WireX = tuple[Literal[True], Literal[False], Literal[False]]
type WireY = tuple[Literal[False], Literal[True], Literal[False]]
type WireZ = tuple[Literal[False], Literal[False], Literal[True]]
type Wire1D = WireX | WireY | WireZ

type AnyPeriodicity = tuple[bool, bool, bool]


class TriclinicMap(Protocol):
    """Mapping that takes positions in an arbitrary frame and returns them
    in the lower-triangular triclinic frame produced by
    [to_lower_triangular][kups.core.cell.to_lower_triangular]."""

    def __call__(self, r: Array, /) -> Array: ...


class Frame(Protocol):
    """3D parallelepiped geometry, no periodicity attached.

    A Frame is a pure geometric container — three basis vectors that span
    a parallelepiped in 3D space. It does not commit to whether those
    vectors represent periodic translations or just bounding-box edges;
    that distinction is supplied by the [Cell][kups.core.cell.Cell] type
    that wraps a Frame.

    In crystallography these basis vectors are conventionally called
    "lattice vectors". They are exposed here under the name
    [`vectors`][kups.core.cell.Frame.vectors] because the crystallographic
    label implies a periodic interpretation that does not apply to all
    Frame uses (e.g. the bounding box of a vacuum simulation).

    Concrete implementations:

    - [OrthogonalFrame][kups.core.cell.OrthogonalFrame]: 3 DOF (lengths).
    - [TriclinicFrame][kups.core.cell.TriclinicFrame]: 6 DOF (lower-triangular).
    """

    @property
    def vectors(self) -> Array:
        """Basis vectors of the parallelepiped, shape ``(..., 3, 3)``.

        Rows are the basis vectors. Lower-triangular by convention so that
        ``v[0]`` lies along x, ``v[1]`` in the xy-plane, ``v[2]`` general.
        Crystallography calls this matrix the *lattice vectors*.
        """
        ...

    @property
    def inverse_vectors(self) -> Array:
        """Matrix inverse of [`vectors`][kups.core.cell.Frame.vectors],
        used to convert real-space coordinates to fractional, shape
        ``(..., 3, 3)``."""
        ...

    @property
    def volume(self) -> Array:
        """Volume of the parallelepiped, shape ``(...)``."""
        ...

    @property
    def perpendicular_lengths(self) -> Array:
        """Perpendicular distance between opposing faces, per axis,
        shape ``(..., 3)``. Used by neighbor-list cutoff checks and by
        [min_multiplicity][kups.core.cell.min_multiplicity]."""
        ...

    def to_fractional(self, r: Array) -> Array:
        """Convert real-space coordinates to fractional, shape ``(..., 3)``."""
        ...

    def to_real(self, r_frac: Array) -> Array:
        """Convert fractional coordinates to real-space, shape ``(..., 3)``."""
        ...

    def __mul__(self, other: Array | float | int) -> Self:
        """Uniformly scale all basis vectors by ``other``."""
        ...

    def tile(self, multiplicities: tuple[int, int, int]) -> Self:
        """Per-axis integer scaling. Used by
        [make_supercell][kups.core.cell.make_supercell] to build supercells."""
        ...

    @classmethod
    def from_matrix(cls, vecs: Array) -> Self:
        """Construct from a ``(..., 3, 3)`` basis matrix.

        Projects the input onto the frame's parameter space — entries
        not represented by the parameterisation are discarded
        (``OrthogonalFrame`` keeps the diagonal; ``TriclinicFrame``
        keeps the lower-triangular block). Used to wrap a generic
        ``∂E/∂h`` matrix back into the same frame type as an input cell.
        """
        ...

    def materialize(self) -> MaterializedFrame:
        """Return a [MaterializedFrame][kups.core.cell.MaterializedFrame]
        with ``vectors``, ``inverse_vectors`` and ``volume`` evaluated
        and stored as concrete arrays.

        Use this to avoid recomputing the inverse and determinant when
        the same frame is queried many times, or to lift these arrays
        across a JIT boundary so downstream callers don't need to know
        the frame's parametrisation.

        Requires at least one leading batch dim so that ``vectors``
        ``(B, ..., 3, 3)``, ``inverse_vectors`` ``(B, ..., 3, 3)`` and
        ``volume`` ``(B, ...)`` share a leading axis (see
        [MaterializedFrame][kups.core.cell.MaterializedFrame]).
        """
        ...


def _build_vectors(lengths: Array, angles: Array) -> Array:
    """Construct a lower-triangular 3x3 matrix from crystallographic parameters.

    Uses the standard convention where the first vector lies along x, the
    second in the xy-plane, and the third completes the cell.
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


@dataclass
class TriclinicFrame(Sliceable):
    """General triclinic [Frame][kups.core.cell.Frame] with 6 degrees of freedom.

    Stores the 6 independent elements of the lower-triangular basis matrix.
    Vectors are a linear function of these parameters, making them suitable
    for gradient-based optimization (e.g. NPT cell-vector relaxation).

    Use this when the simulation domain has non-orthogonal axes
    (monoclinic / triclinic crystals, sheared MD boxes). For axis-aligned
    domains, [OrthogonalFrame][kups.core.cell.OrthogonalFrame] is cheaper.

    Attributes:
        tril: Lower-triangular elements ``[L00, L10, L11, L20, L21, L22]``,
            shape ``(..., 6)``. The basis matrix is::

                [[L00,   0,   0],
                 [L10, L11,   0],
                 [L20, L21, L22]]
    """

    tril: Array

    @classmethod
    def from_matrix(cls, vecs: Array) -> TriclinicFrame:
        """Construct from a lower-triangular basis matrix, shape ``(..., 3, 3)``."""
        vecs = jnp.asarray(vecs)
        return cls(vecs[..., *np.tril_indices(3)])

    @classmethod
    def from_lengths_and_angles(cls, lengths: Array, angles: Array) -> TriclinicFrame:
        """Construct from crystallographic parameters.

        Args:
            lengths: Lattice lengths ``[a, b, c]`` in Angstroms, shape ``(..., 3)``.
            angles: Lattice angles ``[alpha, beta, gamma]`` in degrees, shape ``(..., 3)``.
                alpha = angle(b, c), beta = angle(a, c), gamma = angle(a, b).
        """
        return cls.from_matrix(_build_vectors(lengths, angles))

    @property
    def vectors(self) -> Array:
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
    def inverse_vectors(self) -> Array:
        return triangular_3x3_det_and_inverse(self.vectors)[1]

    @property
    def volume(self) -> Array:
        return jnp.abs(self.tril[..., 0] * self.tril[..., 2] * self.tril[..., 5])

    @property
    def lengths(self) -> Array:
        return jnp.linalg.norm(self.vectors, axis=-1)

    @property
    def angles(self) -> Array:
        v = self.vectors
        a, b, c = v[..., 0, :], v[..., 1, :], v[..., 2, :]
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
        v = self.vectors
        a, b, c = v[..., 0, :], v[..., 1, :], v[..., 2, :]
        Lx = self.volume / jnp.linalg.norm(jnp.cross(b, c), axis=-1)
        Ly = self.volume / jnp.linalg.norm(jnp.cross(a, c), axis=-1)
        Lz = self.volume / jnp.linalg.norm(jnp.cross(a, b), axis=-1)
        return jnp.stack([Lx, Ly, Lz], axis=-1)

    def to_fractional(self, r: Array) -> Array:
        return triangular_3x3_matmul(self.inverse_vectors, r)

    def to_real(self, r_frac: Array) -> Array:
        return triangular_3x3_matmul(self.vectors, r_frac)

    def tile(self, multiplicities: tuple[int, int, int]) -> Self:
        m = jnp.asarray(multiplicities)
        scale = jnp.array([m[0], m[1], m[1], m[2], m[2], m[2]])
        return type(self)(self.tril * scale)

    def __mul__(self, other: Array | float | int) -> Self:
        return type(self)(self.tril * jnp.asarray(other)[..., None])

    def materialize(self) -> MaterializedFrame:
        vecs = self.vectors
        det, inv = triangular_3x3_det_and_inverse(vecs)
        return MaterializedFrame(vectors=vecs, inverse_vectors=inv, volume=jnp.abs(det))


@dataclass
class OrthogonalFrame(Sliceable):
    """Axis-aligned [Frame][kups.core.cell.Frame] with 3 degrees of freedom.

    Parameterized by the three side lengths. Exploits the diagonal
    structure for cheaper volume, inverse, and coordinate-transform
    operations than the general triclinic path. Use this when the
    simulation domain has perpendicular axes (cubic, tetragonal, or
    orthorhombic crystals; standard rectangular MD boxes).

    Attributes:
        lengths: Box side lengths ``[Lx, Ly, Lz]`` in Angstroms,
            shape ``(..., 3)``.
    """

    lengths: Array

    @classmethod
    def from_matrix(cls, vecs: Array) -> Self:
        """Construct from a diagonal basis matrix, shape ``(..., 3, 3)``.

        Projects the input matrix onto the orthogonal subspace by taking
        its diagonal — off-diagonal entries are discarded, matching the
        3-parameter ``(Lx, Ly, Lz)`` representation.
        """
        vecs = jnp.asarray(vecs)
        return cls(jnp.diagonal(vecs, axis1=-2, axis2=-1))

    @property
    def vectors(self) -> Array:
        return self.lengths[..., :, None] * jnp.eye(3)

    @property
    def inverse_vectors(self) -> Array:
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

    def tile(self, multiplicities: tuple[int, int, int]) -> Self:
        return type(self)(self.lengths * jnp.asarray(multiplicities))

    def __mul__(self, other: Array | float | int) -> Self:
        return type(self)(self.lengths * jnp.asarray(other)[..., None])

    def materialize(self) -> MaterializedFrame:
        return MaterializedFrame(
            vectors=self.vectors,
            inverse_vectors=self.inverse_vectors,
            volume=self.volume,
        )


@dataclass
class MaterializedFrame(Sliceable):
    """[Frame][kups.core.cell.Frame] that stores its basis matrix, inverse,
    and volume as concrete arrays.

    Produced by [`Frame.materialize`][kups.core.cell.Frame.materialize].
    Useful when the same frame is queried repeatedly (no recomputation
    of the inverse or determinant) or when the arrays need to cross a
    JIT boundary independently of the frame's original parametrisation.

    The stored matrices are assumed to follow the lower-triangular
    convention shared by every other Frame implementation; coordinate
    transforms use [triangular_3x3_matmul][kups.core.utils.math.triangular_3x3_matmul]
    accordingly. Manually constructing this frame with non-triangular
    vectors violates the convention.

    All three fields are pytree leaves and must share a leading batch
    dim ``B`` to satisfy the [Sliceable][kups.core.data.Sliceable]
    contract — unbatched single-frame inputs must be wrapped with an
    explicit ``[None]`` axis before being passed in.

    Attributes:
        vectors: Basis matrix, shape ``(B, 3, 3)``.
        inverse_vectors: Matrix inverse of ``vectors``, shape ``(B, 3, 3)``.
        volume: Absolute determinant of ``vectors``, shape ``(B,)``.
    """

    vectors: Array
    inverse_vectors: Array
    volume: Array

    @classmethod
    def from_matrix(cls, vecs: Array) -> Self:
        vecs = jnp.asarray(vecs)
        det, inv = triangular_3x3_det_and_inverse(vecs)
        return cls(vectors=vecs, inverse_vectors=inv, volume=jnp.abs(det))

    @property
    def perpendicular_lengths(self) -> Array:
        v = self.vectors
        a, b, c = v[..., 0, :], v[..., 1, :], v[..., 2, :]
        Lx = self.volume / jnp.linalg.norm(jnp.cross(b, c), axis=-1)
        Ly = self.volume / jnp.linalg.norm(jnp.cross(a, c), axis=-1)
        Lz = self.volume / jnp.linalg.norm(jnp.cross(a, b), axis=-1)
        return jnp.stack([Lx, Ly, Lz], axis=-1)

    def to_fractional(self, r: Array) -> Array:
        return triangular_3x3_matmul(self.inverse_vectors, r)

    def to_real(self, r_frac: Array) -> Array:
        return triangular_3x3_matmul(self.vectors, r_frac)

    def tile(self, multiplicities: tuple[int, int, int]) -> Self:
        m = jnp.asarray(multiplicities)
        return type(self)(
            vectors=self.vectors * m[:, None],
            inverse_vectors=self.inverse_vectors / m[None, :],
            volume=self.volume * jnp.prod(m),
        )

    def __mul__(self, other: Array | float | int) -> Self:
        scale = jnp.asarray(other)
        return type(self)(
            vectors=self.vectors * scale[..., None, None],
            inverse_vectors=self.inverse_vectors / scale[..., None, None],
            volume=self.volume * scale**3,
        )

    def materialize(self) -> Self:
        return self


def _wrap(
    frame: Frame,
    periodic: tuple[bool, bool, bool],
    r: Array,
    input_space: CoordinateSpace,
    output_space: CoordinateSpace,
) -> Array:
    """Fold coordinates into ``[-0.5, 0.5)`` along periodic axes."""
    frac = frame.to_fractional(r) if input_space is CoordinateSpace.REAL else r
    wrapped = (frac + 0.5) % 1 - 0.5
    mask = jnp.array(periodic)
    out = jnp.where(mask, wrapped, frac)
    return frame.to_real(out) if output_space is CoordinateSpace.REAL else out


@dataclass
class Cell[P: tuple[bool, bool, bool]](Sliceable):
    """A [Frame][kups.core.cell.Frame] plus per-axis boundary semantics.

    Generic over the periodicity literal ``P`` (a length-3 tuple of
    booleans). [PeriodicCell][kups.core.cell.PeriodicCell] and
    [VacuumCell][kups.core.cell.VacuumCell] are subclasses that pin ``P``
    to a literal-typed default; for slab and wire geometries, construct
    ``Cell(frame, periodic=mask)`` directly — the literal tuple narrows
    ``P`` to the corresponding [Slab2D][kups.core.cell.Slab2D] or
    [Wire1D][kups.core.cell.Wire1D] alias.

    The cell delegates geometry queries (``volume``, ``vectors``, etc.) to
    its frame. Periodic-mask-aware operations
    ([`wrap`][kups.core.cell.Cell.wrap], [`fold`][kups.core.cell.Cell.fold],
    [`minimum_image_shifts`][kups.core.cell.Cell.minimum_image_shifts])
    live on the cell so callers don't need to access ``cell.periodic``
    directly.
    """

    frame: Frame
    periodic: P = field(static=True)

    @property
    def vectors(self) -> Array:
        return self.frame.vectors

    @property
    def inverse_vectors(self) -> Array:
        return self.frame.inverse_vectors

    @property
    def volume(self) -> Array:
        return self.frame.volume

    @property
    def perpendicular_lengths(self) -> Array:
        return self.frame.perpendicular_lengths

    def wrap(
        self,
        r: Array,
        *,
        input_space: CoordinateSpace = CoordinateSpace.REAL,
        output_space: CoordinateSpace = CoordinateSpace.REAL,
    ) -> Array:
        return _wrap(self.frame, self.periodic, r, input_space, output_space)

    def fold(self, r_frac: Array) -> tuple[Array, Array]:
        """Fold fractional coords into ``[0, 1)`` on periodic axes; non-periodic
        axes pass through unchanged.

        Returns ``(folded, in_cell)`` where ``in_cell`` is a per-particle
        mask, ``True`` where the folded coords lie in ``[0, 1)`` on every
        axis. For fully-periodic cells, ``in_cell`` is trivially ``True``
        after folding; for cells with non-periodic axes, particles that
        leaked out of the box on those axes are flagged ``False``.

        Used by neighbor-list spatial hashing; complementary to
        [`wrap`][kups.core.cell.Cell.wrap] which uses the ``[-0.5, 0.5)``
        convention.
        """
        folded = jnp.where(jnp.array(self.periodic), r_frac % 1, r_frac)
        in_cell = jnp.all((folded >= 0) & (folded < 1), axis=-1)
        return folded, in_cell

    def minimum_image_shifts(self, deltas: Array) -> Array:
        """Per-axis minimum-image shifts for fractional separations.

        Returns ``round(deltas)`` on periodic axes (the closest integer
        cell offset that wraps the separation to its minimum image) and
        ``0`` on non-periodic axes.
        """
        return jnp.where(jnp.array(self.periodic), jnp.round(deltas), 0.0)

    def __mul__(self, other: Array | float | int) -> Self:
        return bind(self, lambda c: c.frame).set(self.frame * other)

    @overload
    @staticmethod
    def from_pbc(frame: Frame, pbc: Periodic3D) -> PeriodicCell: ...
    @overload
    @staticmethod
    def from_pbc(frame: Frame, pbc: Vacuum) -> VacuumCell: ...
    @overload
    @staticmethod
    def from_pbc[Q: tuple[bool, bool, bool]](frame: Frame, pbc: Q) -> Cell[Q]: ...
    @staticmethod
    def from_pbc(frame: Frame, pbc: tuple[bool, bool, bool]) -> Cell[Any]:
        """Construct the cell flavor matching ``pbc``.

        Returns [`PeriodicCell`][kups.core.cell.PeriodicCell] for
        ``(True, True, True)``, [`VacuumCell`][kups.core.cell.VacuumCell]
        for ``(False, False, False)``, and a generic ``Cell[P]`` carrying
        the runtime mask for slab and wire geometries (``P`` is inferred
        from the literal tuple).
        """
        match pbc:
            case (True, True, True):
                return PeriodicCell(frame)
            case (False, False, False):
                return VacuumCell(frame)
            case _:
                return Cell(frame, periodic=pbc)


@dataclass
class PeriodicCell(Cell[Periodic3D]):
    """Cell that is periodic along all three axes.

    The frame is interpreted as a *unit cell* — a tile of a periodic
    crystal or fluid. ``wrap`` folds coordinates into the primary unit
    cell; the neighbor list applies the minimum-image convention; the
    Ewald summation requires this cell type.
    """

    periodic: Periodic3D = field(default=(True, True, True), init=False, static=True)


@dataclass
class VacuumCell(Cell[Vacuum]):
    """Cell with all three axes open.

    The frame is interpreted as the *bounding parallelepiped* of a finite
    simulation domain — a cluster, an isolated molecule, a gas-phase
    sample. ``wrap`` is a no-op; long-range electrostatics use direct
    pairwise sums (no Ewald). The frame is still required because the
    neighbor-list machinery needs a spatial-partitioning hint.
    """

    periodic: Vacuum = field(default=(False, False, False), init=False, static=True)


def min_multiplicity(cell: Cell, cutoff: float | Array) -> Array:
    """Minimum supercell replication per axis for a given cutoff.

    Returns 1 for non-periodic axes (no replication needed).
    """
    computed = jnp.ceil(2 * cutoff / cell.perpendicular_lengths).astype(int)
    mask = jnp.array(cell.periodic)
    return jnp.where(mask, computed, 1)


def make_supercell[T, T2, C: Cell[Any]](
    cell: C,
    multiplicities: tuple[int, int, int] | int,
    to_replicate: T,
    to_shift: Lens[T, T2],
) -> tuple[C, T]:
    """Replicate a cell along each periodic axis.

    Tiles the cell according to ``multiplicities`` (clamped to 1 on
    non-periodic axes), replicates the data, and shifts coordinates into
    the expanded cell using periodic wrapping. The returned cell has the
    same concrete type as the input.
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
    real_shifts = triangular_3x3_matmul(cell.vectors, shifts)

    new_cell: C = bind(cell, lambda c: c.frame).set(cell.frame.tile(clamped))

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


def is_vacuum[P: tuple[bool, bool, bool]](
    cell: Cell[P],
) -> TypeGuard[VacuumCell]:
    """``True`` iff ``cell`` is a [VacuumCell][kups.core.cell.VacuumCell]."""
    return isinstance(cell, VacuumCell)


def is_3d_periodic[P: tuple[bool, bool, bool]](
    cell: Cell[P],
) -> TypeGuard[Cell[Periodic3D]]:
    """``True`` iff ``cell`` is periodic on all three axes."""
    return all(cell.periodic)


def require_periodic_3d(cell: Cell) -> None:
    """Raise ``TypeError`` unless ``cell`` is fully 3D-periodic.

    Equivalent to asserting ``isinstance(cell, PeriodicCell)`` but with a
    helpful message.
    """
    if not isinstance(cell, PeriodicCell):
        raise TypeError(
            f"Expected a PeriodicCell (3D-periodic boundaries); got "
            f"{type(cell).__name__} with periodic={cell.periodic}."
        )


def require_triclinic_frame(cell: Cell) -> None:
    """Raise ``TypeError`` unless ``cell.frame`` is a
    [TriclinicFrame][kups.core.cell.TriclinicFrame].

    Some integrators (e.g. fully-flexible-cell NPT Langevin) drift the cell
    matrix to a general lower-triangular state at every step, which an
    [OrthogonalFrame][kups.core.cell.OrthogonalFrame] (3-DOF) cannot represent.
    Use [TriclinicFrame.from_matrix][kups.core.cell.TriclinicFrame.from_matrix]
    to auto-promote ``cell.frame`` before constructing such an integrator.
    """
    if not isinstance(cell.frame, TriclinicFrame):
        raise TypeError(
            f"Expected cell.frame to be TriclinicFrame (6 DOF, lower-triangular); "
            f"got {type(cell.frame).__name__}. Use "
            f"TriclinicFrame.from_matrix(cell.vectors) to promote an "
            f"orthogonal cell."
        )


def require_periodic_3d_triclinic(cell: Cell) -> None:
    """Shorthand for ``require_periodic_3d(cell); require_triclinic_frame(cell)``."""
    require_periodic_3d(cell)
    require_triclinic_frame(cell)


def to_lower_triangular(vecs: Array) -> tuple[Array, TriclinicMap]:
    """Convert arbitrary basis vectors to lower-triangular form via QR.

    The returned basis has a positive diagonal. The coordinate mapper applies
    the same rigid rotation to positions, preserving fractional coordinates.

    Args:
        vecs: Basis vectors as rows of a 3x3 matrix, shape ``(3, 3)``.

    Returns:
        Tuple of (lower_triangular_vectors, coordinate_rotation_fn).
    """
    vecs = jnp.asarray(vecs)
    Q, R = jnp.linalg.qr(vecs.T)
    signs = jnp.sign(jnp.diagonal(R))
    signs = jnp.where(signs == 0, 1.0, signs)
    R = R * signs[:, None]
    Q = Q * signs[None, :]
    L = R.T
    return L, partial(jnp.einsum, "...ij,...i->...j", Q)
