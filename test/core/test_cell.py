# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Frame and Cell types."""

from typing import Callable, Literal, cast, override

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array
from jax.scipy.linalg import expm

from kups.core import units
from kups.core.cell import (
    BaseFrame,
    Cell,
    CoordinateSpace,
    DeformedFrame,
    Frame,
    LinearFrame,
    LogTriclinicFrame,
    MaterializedFrame,
    MatrixLogFrame,
    OrthogonalFrame,
    PeriodicCell,
    TriclinicFrame,
    VacuumCell,
    is_3d_periodic,
    is_vacuum,
    make_supercell,
    min_multiplicity,
    to_lower_triangular,
    zero_gradient_frame,
)
from kups.core.data import Sliceable
from kups.core.lens import lens
from kups.core.units import catalog as u
from kups.core.units import testing
from kups.core.utils.jax import dataclass
from kups.core.utils.math import GeneralSquareMatrix, LowerTriangularSquareMatrix
from test.units import EV, FORCE, A


class TestTriclinicFrame:
    def test_from_matrix(self):
        # cubic
        vecs = jnp.eye(3)
        frame = TriclinicFrame.from_matrix(vecs * A)
        testing.assert_allclose(frame.vectors, vecs, A)
        testing.assert_allclose(frame.inverse_vectors, jnp.eye(3), 1 / A)
        testing.assert_allclose(frame.volume, 1.0, A**3)

        # orthorhombic
        vecs = jnp.diag(jnp.array([2.0, 3.0, 4.0]))
        frame = TriclinicFrame.from_matrix(vecs * A)
        testing.assert_allclose(frame.vectors, vecs, A)
        testing.assert_allclose(
            frame.inverse_vectors,
            jnp.diag(jnp.array([0.5, 1.0 / 3.0, 0.25])),
            1 / A,
        )
        testing.assert_allclose(frame.volume, 24.0, A**3)

        # triclinic
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]])
        frame = TriclinicFrame.from_matrix(vecs * A)
        testing.assert_allclose(frame.vectors, vecs, A)
        testing.assert_allclose(frame.volume, 1.0, A**3)
        npt.assert_allclose(
            frame.vectors @ frame.inverse_vectors,
            jnp.eye(3),
            atol=1e-10,
        )

    def test_plain_magnitudes_are_rejected(self):
        with pytest.raises(TypeError, match="tril"):
            TriclinicFrame.from_matrix(jnp.eye(3) * 10.0)

    def test_from_matrix_negative_volume(self):
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        L, _ = to_lower_triangular(vecs * A)
        frame = TriclinicFrame.from_matrix(L)
        assert frame.volume.to(A**3) > 0
        testing.assert_allclose(frame.volume, 1.0, A**3)

    def test_wrap_real_to_real(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * A))
        r = jnp.array([1.5, -0.7, 2.3])
        testing.assert_allclose(
            cell.wrap(r * A), jnp.array([-0.5, 0.3, 0.3]), A, atol=1e-10
        )

        cell = PeriodicCell(TriclinicFrame.from_matrix((jnp.eye(3) * 2.0) * A))
        r = jnp.array([[3.0, 1.0, -1.0], [-1.0, 4.0, 0.5], [0.0, 0.0, 0.0]])
        expected = jnp.array([[-1.0, -1.0, -1.0], [-1.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
        testing.assert_allclose(cell.wrap(r * A), expected, A, atol=1e-10)

        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * A))
        r = jnp.array(
            [
                [0.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, -0.5, 0.0],
            ]
        )
        expected = jnp.array(
            [
                [-0.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [0.0, -0.5, 0.0],
                [0.0, -0.5, 0.0],
            ]
        )
        testing.assert_allclose(cell.wrap(r * A), expected, A, atol=1e-10)

    def test_wrap_cross_space(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix((jnp.eye(3) * 2.0) * A))

        r = jnp.array([1.0, 3.0, -1.0])
        wrapped = cell.wrap(
            r * A,
            input_space=CoordinateSpace.REAL,
            output_space=CoordinateSpace.FRACTIONAL,
        )
        npt.assert_allclose(wrapped, jnp.array([-0.5, -0.5, -0.5]), atol=1e-10)

        r = jnp.array([1.2, -0.8, 0.3])
        wrapped = cell.wrap(
            r,
            input_space=CoordinateSpace.FRACTIONAL,
            output_space=CoordinateSpace.REAL,
        )
        testing.assert_allclose(wrapped, jnp.array([0.4, 0.4, 0.6]), A, atol=1e-6)

        cell1 = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * A))
        r = jnp.array([1.7, -1.3, 0.8])
        wrapped = cell1.wrap(
            r,
            input_space=CoordinateSpace.FRACTIONAL,
            output_space=CoordinateSpace.FRACTIONAL,
        )
        npt.assert_allclose(wrapped, jnp.array([-0.3, -0.3, -0.2]), atol=1e-6)

    def test_wrap_non_orthogonal_and_shape(self):
        vecs = jnp.array(
            [[1.0, 0.0, 0.0], [0.5, jnp.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]]
        )
        cell = PeriodicCell(TriclinicFrame.from_matrix(vecs * A))
        r = jnp.array([1.5, 1.0, 0.5])
        wrapped = cell.wrap(r * A)
        assert not jnp.allclose(wrapped.to(A), r)
        testing.assert_allclose(wrapped, cell.wrap(wrapped), A, atol=1e-10)

        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * A))
        for shape in [(3,), (5, 3), (2, 4, 3), (10, 1, 3)]:
            r = jnp.ones(shape) * 1.5
            assert cell.wrap(r * A).shape == shape

    def test_wrap_jit_and_gradient(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * A))

        jit_wrap = jax.jit(cell.wrap)
        r = jnp.array([1.5, -0.7, 2.3])
        testing.assert_allclose(cell.wrap(r * A), jit_wrap(r * A), A, atol=1e-10)

        r = jnp.array([0.3, -0.2, 0.1])
        grad = jax.grad(lambda r: units.sum(cell.wrap(r * A)).to(A))(r)
        npt.assert_allclose(grad, jnp.array([1.0, 1.0, 1.0]), atol=1e-6)

    def test_min_multiplicity(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix((jnp.eye(3) * 10.0) * A))
        npt.assert_array_equal(min_multiplicity(cell, 4.0 * A), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 5.0 * A), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 8.0 * A), [2, 2, 2])

        cell = PeriodicCell(
            TriclinicFrame.from_matrix(
                jnp.array([[8.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 6.0]]) * A,
            )
        )
        npt.assert_array_equal(min_multiplicity(cell, 2.5 * A), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 5.0 * A), [2, 1, 2])

        vecs = jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 20.0])
        cell = PeriodicCell(TriclinicFrame.from_matrix(vecs * A))
        result = min_multiplicity(cell, 8.0 * A)
        assert result.shape == (2, 3)
        npt.assert_array_equal(result[0], [2, 2, 2])
        npt.assert_array_equal(result[1], [1, 1, 1])


class TestOrthogonalFrame:
    def test_vectors_diagonal(self):
        frame = OrthogonalFrame(jnp.array([2.0, 3.0, 4.0]) * A)
        testing.assert_allclose(frame.vectors, jnp.diag(jnp.array([2.0, 3.0, 4.0])), A)

    def test_from_matrix_extracts_diagonal(self):
        vecs = jnp.array([[2.0, 0.7, 0.0], [0.0, 3.0, 0.0], [0.4, 0.0, 4.0]])
        frame = OrthogonalFrame.from_matrix(vecs * A)
        testing.assert_allclose(frame.lengths, jnp.array([2.0, 3.0, 4.0]), A)

    def test_from_matrix_batched(self):
        vecs = jnp.stack([jnp.diag(jnp.array([2.0, 3.0, 4.0]))] * 2)
        frame = OrthogonalFrame.from_matrix(vecs * A)
        assert frame.lengths.shape == (2, 3)
        testing.assert_allclose(frame.lengths[0], jnp.array([2.0, 3.0, 4.0]), A)

    def test_wrap_real_to_real(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        r = jnp.array([12.0, -3.0, 25.0])
        wrapped = cell.wrap(r * A)
        testing.assert_allclose(wrapped, jnp.array([2.0, -3.0, -5.0]), A, atol=1e-10)

    def test_wrap_matches_triclinic(self):
        """Orthogonal wrap must agree with equivalent TriclinicFrame."""
        lengths = jnp.array([2.0, 3.0, 4.0])
        ortho = PeriodicCell(OrthogonalFrame(lengths * A))
        tri = PeriodicCell(TriclinicFrame.from_matrix(jnp.diag(lengths) * A))
        r = jnp.array([3.5, -1.2, 7.8])
        testing.assert_allclose(ortho.wrap(r * A), tri.wrap(r * A), A, atol=1e-10)

    def test_wrap_cross_space(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([4.0, 4.0, 4.0]) * A))
        r = jnp.array([3.0, -3.0, 5.0])
        frac = cell.wrap(
            r * A,
            input_space=CoordinateSpace.REAL,
            output_space=CoordinateSpace.FRACTIONAL,
        )
        npt.assert_allclose(frac, jnp.array([-0.25, 0.25, 0.25]), atol=1e-10)

    def test_batched(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        frame = OrthogonalFrame(lengths * A)
        assert frame.volume.shape == (2,)
        testing.assert_allclose(frame.volume, jnp.array([24.0, 210.0]), A**3)

    def test_slicing(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        frame = OrthogonalFrame(lengths * A)
        sub = frame[0]
        testing.assert_allclose(sub.lengths, jnp.array([2.0, 3.0, 4.0]), A)

    def test_cell_isinstance(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0]) * A))
        assert isinstance(cell, Cell)

    def test_is_jax_pytree(self):
        frame = OrthogonalFrame(jnp.array([2.0, 3.0, 4.0]) * A)
        leaves = jax.tree.leaves(frame)
        assert len(leaves) == 1
        scaled = jax.tree.map(lambda x: x * 2, frame)
        testing.assert_allclose(scaled.lengths, jnp.array([4.0, 6.0, 8.0]), A)

    def test_plain_magnitudes_are_rejected(self):
        with pytest.raises(TypeError, match="lengths"):
            OrthogonalFrame(jnp.array([2.0, 3.0, 4.0]))

    def test_other_dimensions_are_rejected(self):
        with pytest.raises(
            units.DimensionError, match="OrthogonalFrame.lengths: .* got one in eV"
        ):
            OrthogonalFrame(jnp.ones(3) * u.eV)
        with pytest.raises(units.DimensionError, match="TriclinicFrame.tril"):
            TriclinicFrame.from_matrix(jnp.eye(3) * u.eV)

    def test_pytree_rebuild_keeps_magnitudes(self):
        """Transforms may map a frame to the magnitudes an optimizer state holds."""
        frame = OrthogonalFrame(jnp.array([2.0, 3.0, 4.0]) * A)
        lowered = units.tree_map(lambda q: q.value, frame)
        npt.assert_array_equal(lowered.lengths, jnp.array([2.0, 3.0, 4.0]))


class TestCellIsinstance:
    def test_periodic_triclinic(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * A))
        assert isinstance(cell, Cell)

    def test_periodic_orthogonal(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0]) * A))
        assert isinstance(cell, Cell)

    def test_vacuum(self):
        cell = VacuumCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0]) * A))
        assert isinstance(cell, Cell)


class TestCellConstructors:
    def test_periodic_default_orthogonal(self):
        c = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        assert c.periodic == (True, True, True)
        assert isinstance(c.frame, OrthogonalFrame)
        testing.assert_allclose(c.frame.lengths, jnp.array([10.0, 10.0, 10.0]), A)

    def test_vacuum_orthogonal(self):
        c = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        assert c.periodic == (False, False, False)
        assert isinstance(c.frame, OrthogonalFrame)
        testing.assert_allclose(c.frame.lengths, jnp.array([10.0, 10.0, 10.0]), A)

    def test_vacuum_triclinic(self):
        c = VacuumCell(TriclinicFrame.from_matrix((jnp.eye(3) * 2.0) * A))
        assert c.periodic == (False, False, False)
        testing.assert_allclose(c.vectors, jnp.eye(3) * 2.0, A)

    @pytest.mark.parametrize("cls", [PeriodicCell, VacuumCell])
    def test_periodic_arg_is_rejected(self, cls):
        frame = OrthogonalFrame(jnp.array([1.0, 1.0, 1.0]) * A)
        with pytest.raises(TypeError, match="periodic"):
            cls(frame, periodic=(True, False, True))  # type: ignore[call-arg]


class TestTypeGuards:
    def test_is_vacuum_positive(self):
        c = VacuumCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0]) * A))
        assert is_vacuum(c)
        assert not is_3d_periodic(c)

    def test_is_3d_periodic_positive(self):
        c = PeriodicCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0]) * A))
        assert is_3d_periodic(c)
        assert not is_vacuum(c)


class TestPeriodicPreservedUnderScaling:
    """Scaling a cell must preserve its concrete type."""

    def test_orthogonal_mul_preserves_vacuum(self):
        c = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        scaled = c * 2.0
        assert isinstance(scaled, VacuumCell)
        assert scaled.periodic == (False, False, False)
        assert isinstance(scaled.frame, OrthogonalFrame)
        testing.assert_allclose(scaled.frame.lengths, jnp.array([20.0, 20.0, 20.0]), A)

    def test_triclinic_mul_preserves_periodic(self):
        c = PeriodicCell(TriclinicFrame.from_matrix((jnp.eye(3) * 5.0) * A))
        scaled = c * 2.0
        assert isinstance(scaled, PeriodicCell)
        assert scaled.periodic == (True, True, True)


class TestLensReplacePreservesConcreteType:
    def test_lens_replace_periodic(self):
        from kups.core.lens import bind

        c = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        new_frame = OrthogonalFrame(jnp.array([5.0, 5.0, 5.0]) * A)
        c2 = bind(c, lambda x: x.frame).set(new_frame)
        assert isinstance(c2, PeriodicCell)
        assert c2.periodic == (True, True, True)

    def test_lens_replace_vacuum(self):
        from kups.core.lens import bind

        c = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 30.0]) * A))
        new_frame = OrthogonalFrame(jnp.array([5.0, 5.0, 10.0]) * A)
        c2 = bind(c, lambda x: x.frame).set(new_frame)
        assert isinstance(c2, VacuumCell)
        assert c2.periodic == (False, False, False)


class TestVacuumWrap:
    def test_vacuum_wrap_is_identity(self):
        """VacuumCell has no periodic axes — wrap must leave coordinates alone."""
        c = VacuumCell(TriclinicFrame.from_matrix((jnp.eye(3) * 5.0) * A))
        r = jnp.array([[15.0, -3.0, 25.0], [-50.0, 100.0, 0.5]])
        testing.assert_allclose(c.wrap(r * A), r, A, atol=1e-6)


class TestCellFold:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize(
        "periodic", [(True, True, True), (True, False, True), (False, False, False)]
    )
    def test_roundoff_at_periodic_boundaries(
        self, dtype: Literal["float32", "float64"], periodic: tuple[bool, bool, bool]
    ):
        cell = Cell(OrthogonalFrame(jnp.ones(3, dtype=dtype) * A), periodic=periodic)
        tiny = jnp.finfo(dtype).eps / 4
        positions = jnp.array(
            [[-tiny, 0.5, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, -tiny]], dtype=dtype
        )
        expected = jnp.where(
            jnp.array(periodic),
            jnp.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]),
            positions,
        )
        folded, in_cell = jax.jit(cell.fold)(positions)
        npt.assert_array_equal(folded, expected)
        npt.assert_array_equal(in_cell, periodic)
        gradient = jax.jit(jax.grad(lambda p: cell.fold(p)[0].sum()))(positions)
        npt.assert_array_equal(gradient, jnp.ones_like(positions))


class TestCellSlicing:
    def test_slice_periodic_orthogonal(self):
        cell = PeriodicCell(
            OrthogonalFrame(jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]) * A)
        )
        sub = cell[0]
        assert isinstance(sub, PeriodicCell)
        assert isinstance(sub.frame, OrthogonalFrame)
        testing.assert_allclose(sub.frame.lengths, jnp.array([2.0, 3.0, 4.0]), A)
        assert sub.periodic == (True, True, True)

    def test_slice_vacuum_triclinic(self):
        cells = jnp.stack([jnp.eye(3) * 5.0, jnp.eye(3) * 10.0])
        cell = VacuumCell(TriclinicFrame.from_matrix(cells * A))
        sub = cell[1]
        assert isinstance(sub, VacuumCell)
        testing.assert_allclose(sub.vectors, jnp.eye(3) * 10.0, A, atol=1e-6)
        assert sub.periodic == (False, False, False)

    def test_add_batch_dim(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([2.0, 3.0, 4.0]) * A))
        batched = cell[None]
        assert isinstance(batched, PeriodicCell)
        frame = batched.frame
        assert isinstance(frame, OrthogonalFrame)
        assert frame.lengths.shape == (1, 3)


class TestMaterializedFrame:
    """All inputs carry a leading batch dim so that vectors ``(B, 3, 3)``,
    inverse ``(B, 3, 3)`` and volume ``(B,)`` share the same leading axis
    and satisfy the ``Batched`` contract."""

    @pytest.fixture
    def vecs(self):
        return jnp.stack(
            [
                jnp.array([[1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.3, 0.4, 3.0]]),
                jnp.array([[2.0, 0.0, 0.0], [0.1, 1.5, 0.0], [0.2, 0.3, 4.0]]),
            ]
        )

    def test_from_matrix_roundtrip(self, vecs):
        mat = MaterializedFrame.from_matrix(vecs * A)
        testing.assert_allclose(mat.vectors, vecs, A, atol=1e-6)
        npt.assert_allclose(
            jax.vmap(lambda a, b: a @ b)(mat.vectors, mat.inverse_vectors),
            jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
            atol=1e-6,
        )
        testing.assert_allclose(
            mat.volume, jnp.abs(jnp.linalg.det(vecs)), A**3, atol=1e-6
        )

    def test_materialize_is_idempotent(self, vecs):
        mat = TriclinicFrame.from_matrix(vecs * A).materialize()
        assert mat.materialize() is mat


class TestMakeSupercell:
    def test_general_frame_shifts_by_lattice_vectors(self):
        vecs = jnp.array([[4.0, 1.0, 0.5], [0.3, 5.0, 1.2], [0.2, 0.7, 6.0]])
        cell = PeriodicCell(MaterializedFrame.from_matrix(vecs[None] * A))
        new_cell, new_positions = make_supercell(
            cell, (2, 1, 1), jnp.zeros((1, 3)) * A, lens(lambda x: x)
        )
        frac = new_cell.frame.to_fractional(new_positions)
        npt.assert_allclose(
            jnp.abs(frac), [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], atol=1e-12
        )

    def test_periodic_orthogonal_replicates_frame_and_data(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        positions = jnp.array([[0.0, 0.0, 0.0]]) * A
        new_cell, new_positions = make_supercell(
            cell, (2, 1, 1), positions, lens(lambda x: x)
        )
        assert isinstance(new_cell, PeriodicCell)
        assert isinstance(new_cell.frame, OrthogonalFrame)
        testing.assert_allclose(
            new_cell.frame.lengths, jnp.array([20.0, 10.0, 10.0]), A
        )
        # 1 particle x 2 replicas = 2 particles
        assert new_positions.shape == (2, 3)

    def test_periodic_int_multiplicity_expanded(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        positions = jnp.array([[0.0, 0.0, 0.0]]) * A
        new_cell, new_positions = make_supercell(cell, 3, positions, lens(lambda x: x))
        new_frame = new_cell.frame
        assert isinstance(new_frame, OrthogonalFrame)
        testing.assert_allclose(new_frame.lengths, jnp.array([30.0, 30.0, 30.0]), A)
        assert new_positions.shape == (27, 3)  # 3^3

    def test_vacuum_clamps_to_one(self):
        """Vacuum cells have no periodic axes — multiplicities must clamp to 1."""
        cell = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * A
        new_cell, new_positions = make_supercell(
            cell, (2, 2, 2), positions, lens(lambda x: x)
        )
        assert isinstance(new_cell, VacuumCell)
        # frame unchanged (clamped to 1 on every axis)
        new_frame = new_cell.frame
        assert isinstance(new_frame, OrthogonalFrame)
        testing.assert_allclose(new_frame.lengths, jnp.array([10.0, 10.0, 10.0]), A)
        # positions unchanged (n_reps = 1)
        assert new_positions.shape == positions.shape

    def test_triclinic_supercell(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix((jnp.eye(3) * 5.0) * A))
        positions = jnp.array([[0.0, 0.0, 0.0]]) * A
        new_cell, _ = make_supercell(cell, (2, 2, 2), positions, lens(lambda x: x))
        assert isinstance(new_cell, PeriodicCell)
        testing.assert_allclose(new_cell.vectors, jnp.eye(3) * 10.0, A, atol=1e-6)

    def test_concrete_subclass_preserved(self):
        """The returned cell type matches the input cell type."""
        positions = jnp.array([[0.0, 0.0, 0.0]]) * A
        for cell_cls in (PeriodicCell, VacuumCell):
            cell = cell_cls(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
            new_cell, _ = make_supercell(cell, 2, positions, lens(lambda x: x))
            assert type(new_cell) is cell_cls

    def test_tuple_data_lens_picks_positions(self):
        """Lens can focus on a specific element of structured replicate-data."""
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        positions = jnp.array([[0.0, 0.0, 0.0]]) * A
        charges = jnp.array([1.0])
        _, (new_positions, new_charges) = make_supercell(
            cell, (2, 1, 1), (positions, charges), lens(lambda x: x[0])
        )
        assert new_positions.shape == (2, 3)
        assert new_charges.shape == (2,)

    def test_zero_multiplicity_rejected(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        positions = jnp.array([[0.0, 0.0, 0.0]]) * A
        with pytest.raises(AssertionError):
            make_supercell(cell, (0, 1, 1), positions, lens(lambda x: x))


class TestMinMultiplicityVacuum:
    def test_vacuum_returns_ones(self):
        """Non-periodic axes don't need replication regardless of cutoff."""
        cell = VacuumCell(OrthogonalFrame(jnp.array([5.0, 5.0, 5.0]) * A))
        # cutoff > box size would imply replication for periodic, but vacuum clamps to 1
        npt.assert_array_equal(min_multiplicity(cell, 100.0 * A), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 1.0 * A), [1, 1, 1])


class TestToLowerTriangular:
    def test_already_lower_triangular_passes_through(self):
        L = jnp.array([[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.1, 0.2, 4.0]])
        L_out, mapper = to_lower_triangular(L * A)
        testing.assert_allclose(L_out, L, A, atol=1e-6)
        # On lower-triangular input the mapper is the identity rotation
        r = jnp.array([1.0, 2.0, 3.0]) * A
        testing.assert_allclose(mapper(r), r, A, atol=1e-6)

    def test_swap_yz_yields_negative_volume_input(self):
        """Input with negative determinant: output volume is positive."""
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        L, _ = to_lower_triangular(vecs * A)
        frame = TriclinicFrame.from_matrix(L)
        assert float(frame.volume) > 0
        testing.assert_allclose(frame.volume, 1.0, A**3, atol=1e-6)

    def test_mapper_preserves_distances(self):
        """The orthogonal rotation preserves vector norms."""
        vecs = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        _, mapper = to_lower_triangular(vecs * A)
        r = jnp.array([1.0, 2.0, 3.0]) * A
        testing.assert_allclose(units.norm(mapper(r)), units.norm(r), A, atol=1e-6)

    @pytest.mark.parametrize(
        "vecs",
        [
            # FCC primitive vectors (a=3.61), historically produced a negative diagonal.
            1.805 * jnp.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
            # Swap-yz input (negative determinant).
            jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            # Random rotations of a positive-diagonal lower-triangular cell.
            *[
                jnp.array([[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.1, 0.2, 4.0]])
                @ jnp.linalg.qr(jax.random.normal(jax.random.key(seed), (3, 3)))[0]
                for seed in (0, 1, 2)
            ],
        ],
    )
    def test_diagonal_is_positive_for_various_inputs(self, vecs):
        L, _ = to_lower_triangular(vecs * A)
        assert jnp.all(jnp.diagonal(L.to(A)) > 0)

    def test_round_trip_recovers_input(self):
        """``vecs == L @ Q.T``, where Q is the rotation reconstructed from
        the mapper. The mapper acts as ``r @ Q`` in row form (= ``Q.T @ r``
        on a column vector), matching the same rotation that takes
        ``vecs`` row-by-row into the rows of ``L``."""
        vecs = jnp.array([[1.0, 2.0, 0.3], [0.2, 3.0, 0.1], [0.5, 0.4, 4.0]])
        L, mapper = to_lower_triangular(vecs * A)
        Q = jnp.stack([mapper(row * A).to(A) for row in jnp.eye(3)])
        testing.assert_allclose(L @ Q.T, vecs, A, atol=1e-6)

    def test_mapper_preserves_fractional_coordinates(self):
        """The mapper re-expresses positions without changing fractional coordinates."""
        vecs = jnp.array([[1.0, 2.0, 0.3], [0.2, 3.0, 0.1], [0.5, 0.4, 4.0]])
        L, mapper = to_lower_triangular(vecs * A)
        # Several arbitrary fractional positions; all must survive.
        frac_coords = jnp.array(
            [[0.7, 0.3, 0.5], [0.0, 0.0, 0.0], [0.123, 0.456, 0.789]]
        )
        for f in frac_coords:
            r_orig = f @ vecs * A
            r_new = mapper(r_orig)
            f_new = r_new @ units.inv(L)
            npt.assert_allclose(f_new, f, atol=1e-6)

    def test_mapper_preserves_fractional_coordinates_fcc_primitive(self):
        """End-to-end regression for the FCC primitive ASE case: the
        upper-triangular ASE convention must round-trip through the rotation
        without changing fractional coordinates."""
        vecs = 1.805 * jnp.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        # The 8 atoms of an FCC 2×2×2 primitive supercell — fractional coords
        # are at half-integer combinations of the basis vectors.
        frac = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
            ]
        )
        L, mapper = to_lower_triangular(vecs * A)
        r_orig = frac @ vecs * A
        r_new = units.stack([mapper(r) for r in r_orig])
        frac_new = r_new @ units.inv(L)
        npt.assert_allclose(frac_new, frac, atol=1e-5)

    def test_fcc_primitive_gives_positive_diagonal(self):
        """Regression: ase FCC Cu (a=3.61) previously produced a negative diagonal."""
        import ase.build

        vecs = jnp.asarray(ase.build.bulk("Cu", "fcc", a=3.61).cell.array)
        L, _ = to_lower_triangular(vecs * A)
        assert jnp.all(jnp.diagonal(L.to(A)) > 0)


_FRAMES = pytest.mark.parametrize(
    "frame",
    [
        OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A),
        TriclinicFrame.from_matrix((jnp.eye(3) * 10.0) * A),
    ],
    ids=["orthogonal", "triclinic"],
)


class TestPeriodicityAcrossFrames:
    """The cell-type × frame-type matrix for periodicity-aware operations.

    `wrap`, `min_multiplicity`, and `make_supercell` all consult the cell's
    `periodic` mask. Each operation must behave correctly for both
    PeriodicCell and VacuumCell, with both frame parameterizations.
    """

    @_FRAMES
    def test_periodic_wrap_folds(self, frame):
        cell = PeriodicCell(frame)
        # 12 along x in a 10 Å cubic frame folds to 2
        wrapped = cell.wrap(jnp.array([12.0, 0.0, 0.0]) * A)
        testing.assert_allclose(wrapped, jnp.array([2.0, 0.0, 0.0]), A, atol=1e-6)

    @_FRAMES
    def test_vacuum_wrap_is_identity(self, frame):
        cell = VacuumCell(frame)
        r = jnp.array([12.0, -3.0, 25.0])
        testing.assert_allclose(cell.wrap(r * A), r, A, atol=1e-6)

    @_FRAMES
    def test_periodic_min_multiplicity_grows_with_cutoff(self, frame):
        cell = PeriodicCell(frame)
        npt.assert_array_equal(min_multiplicity(cell, 4.0 * A), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 8.0 * A), [2, 2, 2])

    @_FRAMES
    def test_vacuum_min_multiplicity_always_one(self, frame):
        cell = VacuumCell(frame)
        # cutoff bigger than the box would force replication if periodic
        npt.assert_array_equal(min_multiplicity(cell, 100.0 * A), [1, 1, 1])

    @_FRAMES
    def test_periodic_supercell_replicates(self, frame):
        cell = PeriodicCell(frame)
        positions = jnp.array([[0.0, 0.0, 0.0]]) * A
        new_cell, new_positions = make_supercell(
            cell, (2, 2, 2), positions, lens(lambda x: x)
        )
        testing.assert_allclose(new_cell.volume, 8 * cell.volume, A**3, rtol=1e-6)
        assert new_positions.shape == (8, 3)

    @_FRAMES
    def test_vacuum_supercell_clamps(self, frame):
        cell = VacuumCell(frame)
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * A
        new_cell, new_positions = make_supercell(
            cell, (2, 3, 4), positions, lens(lambda x: x)
        )
        # frame untouched; data not replicated
        testing.assert_allclose(new_cell.volume, cell.volume, A**3, rtol=1e-6)
        assert new_positions.shape == positions.shape


class TestPytreeAndJIT:
    def test_periodic_is_static_aux_not_a_leaf(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        assert len(jax.tree.leaves(cell)) == 1

    def test_jit_traces_through_wrap(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        r = jnp.array([12.0, -3.0, 25.0])
        testing.assert_allclose(
            jax.jit(cell.wrap)(r * A), cell.wrap(r * A), A, atol=1e-6
        )

    def test_tree_map_preserves_concrete_subclass(self):
        cell = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0]) * A))
        scaled = jax.tree.map(lambda x: x * 2, cell)
        assert isinstance(scaled, VacuumCell)


class TestFromLengthsAndAngles:
    def test_roundtrip(self):
        """Construct from (lengths, angles); read them back."""
        lengths = jnp.array([5.0, 6.0, 7.0])
        angles = jnp.array([85.0, 95.0, 100.0])
        frame = TriclinicFrame.from_lengths_and_angles(lengths * A, angles)
        testing.assert_allclose(frame.lengths, lengths, A, atol=1e-6)
        npt.assert_allclose(frame.angles, angles, atol=1e-6)


@dataclass
class _ExpFrame(BaseFrame, Sliceable):
    """Toy nonlinear frame for tests: ``vectors = diag(exp(log_lengths))``.

    The exponential makes ``vectors`` a nonlinear function of the parameters,
    so its Jacobian is not orthonormal — exercising the general inverse-Jacobian
    path in [BaseFrame.vectors_gradient][kups.core.cell.BaseFrame.vectors_gradient]
    that the linear frames override.
    """

    log_lengths: jax.Array

    @classmethod
    @override
    def from_matrix(cls, vecs: units.Length) -> "_ExpFrame":
        diagonal = vecs.diagonal(axis1=-2, axis2=-1)
        return cls(jnp.log(diagonal.to(A)))

    @property
    @override
    def vectors(self) -> units.Length:
        return jnp.exp(self.log_lengths)[..., :, None] * jnp.eye(3) * A

    @override
    def tile(self, multiplicities: tuple[int, int, int]) -> "_ExpFrame":
        return type(self)(self.log_lengths + jnp.log(jnp.asarray(multiplicities)))

    @override
    def __mul__(self, other: jax.Array | float | int) -> "_ExpFrame":
        return type(self)(self.log_lengths + jnp.log(jnp.asarray(other)))


# A batched (1, 3, 3) general lower-triangular matrix; each frame keeps the part
# it can represent.
_GRAD_PRIMAL = jnp.array([[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.1, 0.2, 4.0]])[None] * A
_CELL_GRAD = jnp.arange(9.0).reshape(1, 3, 3) * FORCE

# (frame class, projection of a cartesian grad onto the frame's representable DOF)
# for the linear parameterisations whose Jacobian is orthonormal.
_FrameClass = type[BaseFrame]
_Projector = Callable[[units.Force], units.Force]
_LINEAR_FRAMES = [
    pytest.param(TriclinicFrame, lambda m: units.apply(jnp.tril, m), id="triclinic"),
    pytest.param(OrthogonalFrame, lambda m: m * jnp.eye(3), id="orthogonal"),
]
_LINEAR_FRAME_CLASSES = [
    pytest.param(TriclinicFrame, id="triclinic"),
    pytest.param(OrthogonalFrame, id="orthogonal"),
]

# Frame classes whose ``from_matrix`` accepts an unbatched matrix, including the
# nonlinear toy frame that exercises the general inverse-Jacobian path.
_GRAD_FRAME_CLASSES = [
    *_LINEAR_FRAME_CLASSES,
    pytest.param(LogTriclinicFrame, id="log_triclinic"),
    pytest.param(DeformedFrame, id="deformed"),
    pytest.param(_ExpFrame, id="exp_nonlinear"),
]


class TestFrameGradients:
    """``∂E/∂h`` ↔ ``∂E/∂θ`` maps on the Frame protocol."""

    def test_baseframe_not_instantiable(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseFrame()  # type: ignore[abstract]

    @pytest.mark.parametrize("frame_cls, project", _LINEAR_FRAMES)
    def test_linear_roundtrip_projects_onto_dofs(
        self, frame_cls: _FrameClass, project: _Projector
    ):
        """``parameter_gradient`` keeps the frame type; ``vectors_gradient``
        recovers the cartesian grad projected onto the representable DOF."""
        frame = frame_cls.from_matrix(_GRAD_PRIMAL)
        param_grad = frame.parameter_gradient(_CELL_GRAD)
        assert isinstance(param_grad, frame_cls)
        testing.assert_allclose(
            frame.vectors_gradient(param_grad), project(_CELL_GRAD), FORCE, atol=1e-6
        )

    @pytest.mark.parametrize("frame_cls", _LINEAR_FRAME_CLASSES)
    def test_general_path_matches_linear_override(self, frame_cls: _FrameClass):
        """``BaseFrame``'s general (Jᵀ)⁺ path agrees with the cheap override."""
        frame = frame_cls.from_matrix(_GRAD_PRIMAL)
        param_grad = frame.parameter_gradient(_CELL_GRAD)
        testing.assert_allclose(
            BaseFrame.vectors_gradient(frame, param_grad),
            frame.vectors_gradient(param_grad),
            FORCE,
            atol=1e-6,
        )

    @pytest.mark.parametrize("frame_cls, project", _LINEAR_FRAMES)
    def test_vectors_gradient_batched(
        self, frame_cls: _FrameClass, project: _Projector
    ):
        frames = frame_cls.from_matrix(
            units.concatenate([_GRAD_PRIMAL, _GRAD_PRIMAL * 1.5])
        )
        cell_grad = units.apply(jnp.broadcast_to, _CELL_GRAD, (2, 3, 3))
        param_grad = frames.parameter_gradient(cell_grad)
        out = frames.vectors_gradient(param_grad)
        assert out.shape == (2, 3, 3)
        testing.assert_allclose(out, project(cell_grad), FORCE, atol=1e-6)

    @pytest.mark.parametrize(
        "base", [units.ATOMISTIC, units.SI], ids=["atomistic", "si"]
    )
    @pytest.mark.parametrize("frame_cls", _GRAD_FRAME_CLASSES)
    def test_roundtrip_identity_on_parameters(
        self, frame_cls: _FrameClass, base: units.BaseSystem
    ):
        """``parameter_gradient ∘ vectors_gradient`` is the identity on parameter
        space (holds for the nonlinear frame too)."""
        frame = frame_cls.from_matrix(_GRAD_PRIMAL.rebase(base))
        param_grad = frame.parameter_gradient(_CELL_GRAD)
        recovered = frame.parameter_gradient(frame.vectors_gradient(param_grad))
        for got, want in zip(jax.tree.leaves(recovered), jax.tree.leaves(param_grad)):
            npt.assert_allclose(got, want, atol=1e-6)

    def test_nonlinear_inverse_jacobian_closed_form(self):
        """For ``vectors = diag(exp(θ))``: ``∂E/∂h_ii = ∂E/∂θ_i / exp(θ_i)``."""
        theta = jnp.log(jnp.array([2.0, 3.0, 4.0]))
        g_h = _ExpFrame(theta).vectors_gradient(_ExpFrame(jnp.array([1.0, 2.0, 5.0])))
        testing.assert_allclose(
            units.apply(jnp.diag, g_h),
            jnp.array([1.0, 2.0, 5.0]) / jnp.exp(theta),
            FORCE,
            atol=1e-6,
        )
        diag = units.apply(lambda x: jnp.diag(jnp.diag(x)), g_h)
        testing.assert_allclose(g_h - diag, 0.0, FORCE, atol=1e-6)

    @pytest.mark.parametrize(
        "base", [units.ATOMISTIC, units.SI], ids=["atomistic", "si"]
    )
    def test_materialized_frame_gradient_identity(self, base: units.BaseSystem):
        frame = MaterializedFrame.from_matrix(_GRAD_PRIMAL.rebase(base))
        cell_grad = units.apply(jnp.tril, _CELL_GRAD)
        param_grad = frame.parameter_gradient(cell_grad)
        testing.assert_allclose(
            frame.vectors_gradient(param_grad), cell_grad, FORCE, atol=1e-6
        )


# One frame per implementation, all batched so ``vectors`` is ``(B, 3, 3)``; each
# describes its own lower-triangular parallelepiped. The triclinic case carries a
# second batch entry to exercise batched geometry.
_M_ORTHO = jnp.diag(jnp.array([2.0, 3.0, 4.0]))[None]
_M_TRICLINIC = jnp.array([[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.1, 0.2, 4.0]])[None]
_M_SI = (_M_TRICLINIC * A).rebase(units.SI)
_GEOMETRY_FRAMES = [
    pytest.param(
        TriclinicFrame.from_matrix(jnp.concatenate([_M_TRICLINIC, _M_ORTHO]) * A),
        id="triclinic",
    ),
    pytest.param(OrthogonalFrame.from_matrix(_M_ORTHO * A), id="orthogonal"),
    pytest.param(MaterializedFrame.from_matrix(_M_TRICLINIC * A), id="materialized"),
    pytest.param(LogTriclinicFrame.from_matrix(_M_TRICLINIC * A), id="log_triclinic"),
    pytest.param(DeformedFrame.from_matrix(_M_TRICLINIC * A), id="deformed"),
    # The same frames built from SI magnitudes.
    pytest.param(MaterializedFrame.from_matrix(_M_SI), id="materialized_si"),
    pytest.param(LogTriclinicFrame.from_matrix(_M_SI), id="log_triclinic_si"),
    pytest.param(DeformedFrame.from_matrix(_M_SI), id="deformed_si"),
]
_frame_case = pytest.mark.parametrize("frame", _GEOMETRY_FRAMES)
# Frame classes that a ``(B,)`` factor scales system by system.
_SCALED_FRAME_CLASSES = [
    *_LINEAR_FRAME_CLASSES,
    pytest.param(LogTriclinicFrame, id="log_triclinic"),
    pytest.param(MatrixLogFrame, id="matrix_log"),
    pytest.param(DeformedFrame, id="deformed"),
]


class TestFrameGeometry:
    """``vectors``-derived geometry, checked uniformly across implementations.

    Consolidates the inverse, volume, perpendicular-length, coordinate-transform,
    materialize, scaling and tile invariants that each frame type shares, and
    locks every implementation (the two ``BaseFrame`` subclasses and the
    structural ``MaterializedFrame``) to the linalg ground truth.
    """

    @_frame_case
    def test_inverse_vectors_inverts(self, frame: Frame):
        eye = jnp.broadcast_to(jnp.eye(3), frame.vectors.shape)
        npt.assert_allclose(frame.vectors @ frame.inverse_vectors, eye, atol=1e-6)

    @_frame_case
    def test_volume_matches_determinant(self, frame: Frame):
        testing.assert_allclose(
            frame.volume,
            units.absolute(units.det(frame.vectors)),
            A**3,
            atol=1e-6,
        )

    @_frame_case
    def test_perpendicular_lengths(self, frame: Frame):
        v = frame.vectors
        a, b, c = v[..., 0, :], v[..., 1, :], v[..., 2, :]
        reference = frame.volume[..., None] / units.stack(
            [
                units.norm(units.cross(b, c), axis=-1),
                units.norm(units.cross(a, c), axis=-1),
                units.norm(units.cross(a, b), axis=-1),
            ],
            axis=-1,
        )
        pl = frame.perpendicular_lengths
        assert pl.shape == frame.vectors.shape[:-1]
        testing.assert_allclose(pl, reference, A, atol=1e-6)
        assert bool(jnp.all(pl.to(A) > 0))

    @_frame_case
    def test_coordinate_roundtrip(self, frame: Frame):
        r = jnp.broadcast_to(jnp.array([0.1, 0.2, 0.3]), (*frame.vectors.shape[:-2], 3))
        testing.assert_allclose(
            frame.to_real(frame.to_fractional(r * A)), r, A, atol=1e-6
        )

    @_frame_case
    def test_materialize_preserves_geometry(self, frame: Frame):
        mat = frame.materialize()
        assert isinstance(mat, MaterializedFrame)
        testing.assert_allclose(mat.vectors, frame.vectors, A, atol=1e-6)
        testing.assert_allclose(
            mat.inverse_vectors, frame.inverse_vectors, 1 / A, atol=1e-6
        )
        testing.assert_allclose(mat.volume, frame.volume, A**3, atol=1e-6)
        testing.assert_allclose(
            mat.perpendicular_lengths, frame.perpendicular_lengths, A, atol=1e-6
        )

    @_frame_case
    def test_scaling(self, frame: Frame):
        scaled = frame * 2.0
        testing.assert_allclose(scaled.vectors, frame.vectors * 2.0, A, atol=1e-6)
        testing.assert_allclose(
            scaled.inverse_vectors, frame.inverse_vectors / 2.0, 1 / A, atol=1e-6
        )
        testing.assert_allclose(scaled.volume, frame.volume * 8.0, A**3, atol=1e-6)

    @pytest.mark.parametrize("frame_cls", _SCALED_FRAME_CLASSES)
    def test_per_system_scaling(self, frame_cls: _FrameClass):
        """A ``(B,)`` factor scales each system by its own entry."""
        frame = frame_cls.from_matrix(
            jnp.concatenate([_M_TRICLINIC, _M_ORTHO, 1.5 * _M_TRICLINIC]) * A
        )
        scale = jnp.array([1.5, 2.0, 3.0])
        scaled = frame * scale
        testing.assert_allclose(
            scaled.vectors, frame.vectors * scale[:, None, None], A, atol=1e-6
        )
        testing.assert_allclose(scaled.volume, frame.volume * scale**3, A**3, atol=1e-6)

    @_frame_case
    def test_tile_scales_basis_rows(self, frame: Frame):
        multiples = jnp.array([2, 3, 4])
        testing.assert_allclose(
            frame.tile((2, 3, 4)).vectors,
            frame.vectors * multiples[:, None],
            A,
            atol=1e-6,
        )


class TestLogTriclinicFrame:
    """Matrix-log parameterisation: ``vectors == expm(A)`` for lower-triangular ``A``."""

    def test_vectors_is_expm_of_lower_triangular(self):
        a = jnp.array([[0.7, 0.0, 0.0], [0.5, 1.1, 0.0], [0.1, 0.2, 1.4]])
        tril = jnp.array([[0.7, 0.5, 1.1, 0.1, 0.2, 1.4]])
        frame = LogTriclinicFrame(tril, jnp.ones(1))
        testing.assert_allclose(frame.vectors, expm(a)[None], A, atol=1e-6)
        # expm of a lower-triangular matrix stays lower-triangular.
        testing.assert_allclose(
            units.apply(jnp.triu, frame.vectors, 1), 0.0, A, atol=1e-6
        )

    def test_volume_is_exp_trace(self):
        tril = jnp.array([[0.7, 0.5, 1.1, 0.1, 0.2, 1.4]])
        frame = LogTriclinicFrame(tril, cell_factor=jnp.ones(1))
        trace = tril[0, 0] + tril[0, 2] + tril[0, 5]
        testing.assert_allclose(frame.volume, jnp.exp(trace), A**3, atol=1e-6)
        # The exponential map always yields a positive-volume cell.
        assert float(frame.volume[0]) > 0.0

    def test_from_matrix_roundtrips_through_expm(self):
        """``from_matrix`` (triangular logm) then ``vectors`` (expm) recovers the
        input lattice. Degenerate-diagonal robustness is covered by
        ``test_math.TestTriangular3x3Logm``."""
        lattice = expm(jnp.array([[0.7, 0, 0], [0.5, 1.1, 0], [0.3, 0.2, 1.4]]))
        frame = LogTriclinicFrame.from_matrix(lattice[None] * A)
        testing.assert_allclose(frame.vectors, lattice[None], A, atol=1e-5)

    def test_uses_general_nonlinear_gradient_path(self):
        """``vectors`` is nonlinear in the parameters, so the frame is a plain
        ``BaseFrame`` (general inverse-Jacobian), not a ``LinearFrame``."""
        frame = LogTriclinicFrame.from_matrix((jnp.eye(3)[None] * 2.0) * A)
        assert isinstance(frame, BaseFrame)
        assert not isinstance(frame, LinearFrame)

    def test_cell_factor_scales_exponential_map(self):
        """``cell_factor`` divides ``A`` inside the exponential map."""
        a = jnp.array([[0.7, 0.0, 0.0], [0.5, 1.1, 0.0], [0.1, 0.2, 1.4]])
        tril = jnp.array([[0.7, 0.5, 1.1, 0.1, 0.2, 1.4]])
        frame = LogTriclinicFrame(tril, jnp.array([4.0]))
        testing.assert_allclose(frame.vectors, expm(a / 4.0)[None], A, atol=1e-6)

    def test_from_matrix_cell_factor_roundtrips(self):
        """``from_matrix(L, cell_factor=k).vectors == L`` regardless of ``k``."""
        lattice = expm(jnp.array([[0.7, 0, 0], [0.5, 1.1, 0], [0.3, 0.2, 1.4]]))
        frame = LogTriclinicFrame.from_matrix(lattice[None] * A, cell_factor=4.0)
        testing.assert_allclose(frame.vectors, lattice[None], A, atol=1e-5)


class TestDeformedFrame:
    """Deformation parameterisation: ``vectors == base @ deformation`` with the
    fixed reference ``base`` held stop-gradient, so only the deformation is
    optimised. With a ``LogTriclinicFrame`` deformation this is ASE's
    ``FrechetCellFilter``: ``vectors == base @ expm(A / cell_factor)``."""

    def test_vectors_is_base_times_deformation(self):
        a = jnp.array([[0.7, 0.0, 0.0], [0.5, 1.1, 0.0], [0.1, 0.2, 1.4]])
        tril = jnp.array([[0.7, 0.5, 1.1, 0.1, 0.2, 1.4]])
        base = TriclinicFrame.from_matrix(_M_TRICLINIC * A)
        frame = DeformedFrame(base, LogTriclinicFrame(tril, jnp.ones(1)))
        # Right-multiplication (Cartesian deformation), matching ASE's
        # FrechetCellFilter; left-multiplication would recombine lattice vectors.
        testing.assert_allclose(
            frame.vectors, base.vectors @ jnp.asarray(expm(a)), A, atol=1e-6
        )
        testing.assert_allclose(
            frame.vectors, base.vectors @ frame.deformation.vectors.to(A), A, atol=1e-6
        )
        # base @ expm(A) of two lower-triangular factors stays lower-triangular.
        testing.assert_allclose(
            units.apply(jnp.triu, frame.vectors, 1), 0.0, A, atol=1e-6
        )

    def test_from_frame_anchors_at_base(self):
        """The identity deformation (``A = 0``) reproduces the reference frame."""
        base = TriclinicFrame.from_matrix(_M_TRICLINIC * A)
        frame = DeformedFrame.from_frame(base)
        deformation = cast(LogTriclinicFrame, frame.deformation)
        npt.assert_allclose(deformation.tril, 0.0, atol=1e-6)
        testing.assert_allclose(frame.vectors, base.vectors, A, atol=1e-6)
        testing.assert_allclose(frame.reference_vectors, base.vectors, A, atol=1e-6)

    def test_base_held_fixed_under_gradient(self):
        """The cartesian→parameter pullback flows entirely into the deformation;
        the stop-gradient base receives no gradient."""
        frame = DeformedFrame.from_matrix(_M_TRICLINIC * A)
        grad = frame.parameter_gradient(_CELL_GRAD)
        assert isinstance(grad, DeformedFrame)
        testing.assert_allclose(grad.base.vectors, 0.0, FORCE, atol=1e-6)
        deformation_grad = cast(
            units.Energy, cast(LogTriclinicFrame, grad.deformation).tril
        )
        assert bool(jnp.any(units.absolute(deformation_grad).to(u.eV) > 0.0))

    def test_cell_factor_stiffens_deformation_and_gradient(self):
        """``cell_factor`` (ASE's exp_cell_factor) scales the deformation to
        ``expm(A / cell_factor)`` and divides the parameter gradient by it."""
        a = jnp.array([[0.3, 0, 0], [0.2, 0.1, 0], [0.05, 0.1, 0.2]])
        tril = jnp.array([[0.3, 0.2, 0.1, 0.05, 0.1, 0.2]])
        base = TriclinicFrame.from_matrix(_M_TRICLINIC * A)
        defm4 = cast(
            LogTriclinicFrame,
            DeformedFrame.from_frame(base, cell_factor=4.0).deformation,
        )
        cf4 = defm4.cell_factor
        testing.assert_allclose(
            DeformedFrame(base, LogTriclinicFrame(tril, cf4)).vectors,
            base.vectors @ expm(a / 4.0),
            A,
            atol=1e-6,
        )
        # At A = 0 the deformation Jacobian is the base Jacobian times
        # 1 / cell_factor, so the parameter gradient scales exactly by it.
        z = jnp.zeros((1, 6))
        cf1 = cast(
            LogTriclinicFrame,
            DeformedFrame.from_frame(base, cell_factor=1.0).deformation,
        ).cell_factor
        g1 = cast(
            LogTriclinicFrame,
            DeformedFrame(base, LogTriclinicFrame(z, cf1))
            .parameter_gradient(_CELL_GRAD)
            .deformation,
        ).tril
        g4 = cast(
            LogTriclinicFrame,
            DeformedFrame(base, LogTriclinicFrame(z, cf4))
            .parameter_gradient(_CELL_GRAD)
            .deformation,
        ).tril
        testing.assert_allclose(g4, g1 / 4.0, EV, atol=1e-6)

    def test_per_system_cell_factor(self):
        """A per-system ``cell_factor`` array threads through ``from_frame`` and
        ``vectors``, deforming each system by its own factor."""
        a = jnp.array([[0.3, 0, 0], [0.2, 0.1, 0], [0.05, 0.1, 0.2]])
        tril = jnp.array([0.3, 0.2, 0.1, 0.05, 0.1, 0.2])
        base = TriclinicFrame.from_matrix(jnp.broadcast_to(_M_TRICLINIC, (3, 3, 3)) * A)
        cell_factor = jnp.array([1.0, 2.0, 4.0])
        frame = DeformedFrame.from_frame(base, cell_factor=cell_factor)
        deformation = cast(LogTriclinicFrame, frame.deformation)
        # One value per system, sharing ``tril``'s leading dim.
        assert deformation.cell_factor.shape == (3, 1)
        npt.assert_allclose(deformation.tril, 0.0)
        batched = DeformedFrame(
            base,
            LogTriclinicFrame(jnp.broadcast_to(tril, (3, 6)), deformation.cell_factor),
        )
        for i, cf in enumerate([1.0, 2.0, 4.0]):
            testing.assert_allclose(
                batched.vectors[i], _M_TRICLINIC[0] @ expm(a / cf), A, atol=1e-6
            )

    def test_per_system_cell_factor_zero_gradient(self):
        """``from_frame`` stop-gradients ``cell_factor`` even as a per-system
        array, so it receives no gradient."""
        base = TriclinicFrame.from_matrix(jnp.broadcast_to(_M_TRICLINIC, (3, 3, 3)) * A)
        cell_factor = jnp.array([1.0, 2.0, 4.0])

        def loss(cf: Array) -> Array:
            built = cast(
                LogTriclinicFrame,
                DeformedFrame.from_frame(base, cell_factor=cf).deformation,
            )
            frame = DeformedFrame(
                base, LogTriclinicFrame(jnp.ones((3, 6)) * 0.1, built.cell_factor)
            )
            return units.sum(frame.vectors).to(A)

        grad = jax.grad(loss)(cell_factor)
        npt.assert_allclose(grad, 0.0)

    @pytest.mark.parametrize("deformation", [LogTriclinicFrame, MatrixLogFrame])
    def test_system_slices_and_gathers(
        self, deformation: type[LogTriclinicFrame | MatrixLogFrame]
    ):
        """A one-row slice or a gather by system index gives each system its own
        base, deformation and ``cell_factor``."""
        scale = jnp.array([1.0, 1.5, 2.0])[:, None, None]
        base = TriclinicFrame.from_matrix(_M_TRICLINIC * scale * A)
        # Distinct diagonal entries keep MatrixLogFrame's eig-based logm accurate.
        strain = jnp.eye(3) + scale * jnp.array(
            [[0.1, 0.0, 0.0], [0.05, -0.1, 0.0], [0.02, 0.03, 0.2]]
        )
        cell_factor = jnp.array([1.0, 2.0, 4.0])
        frame = DeformedFrame(
            base, deformation.from_matrix(strain * A, cell_factor=cell_factor)
        )
        expected = base.vectors @ strain
        for i in range(3):
            testing.assert_allclose(
                frame[i : i + 1].vectors[0], expected[i], A, atol=1e-6
            )
        ids = jnp.array([2, 0, 1, 1])
        testing.assert_allclose(frame[ids].vectors, expected[ids], A, atol=1e-6)


_M_ROTATED = jnp.array([[2.0, 0.3, -0.1], [0.4, 3.0, 0.2], [-0.2, 0.5, 4.0]])[
    None
]  # batched (1, 3, 3) full (non-triangular) basis


def _frechet(base: units.Length, deformation: MatrixLogFrame) -> DeformedFrame:
    """Full-3×3 Frechet basis: ``DeformedFrame(MatrixLog base, MatrixLog deformation)``."""
    return DeformedFrame(MatrixLogFrame.from_lower_triangular(base), deformation)


class TestMatrixLogFrame:
    """Full-3×3 matrix-log frame and its ``DeformedFrame(MatrixLog, MatrixLog)`` use."""

    @pytest.fixture(autouse=True)
    def _x64(self):
        prev = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        yield
        jax.config.update("jax_enable_x64", prev)

    def test_structure_is_general(self):
        matrix = MatrixLogFrame.from_matrix(_M_ROTATED * A).matrix
        assert type(matrix) is GeneralSquareMatrix

    def test_deformed_matrix_log_structure_is_general(self):
        base = TriclinicFrame.from_matrix(_M_TRICLINIC * A)
        frame = DeformedFrame.from_frame(base, deformation=MatrixLogFrame)
        assert type(frame.matrix) is GeneralSquareMatrix

    def test_deformed_log_triclinic_structure_unchanged(self):
        base = TriclinicFrame.from_matrix(_M_TRICLINIC * A)
        frame = DeformedFrame.from_frame(base)  # default LogTriclinicFrame
        assert type(frame.matrix) is LowerTriangularSquareMatrix

    def test_vectors_is_expm_of_log(self):
        from scipy.linalg import expm as scipy_expm

        log_m = jnp.asarray(
            [[0.05, 0.10, -0.03], [0.02, 0.07, 0.04], [-0.06, 0.01, 0.08]]
        )
        cf = 2.0
        frame = MatrixLogFrame(log_m[None], jnp.asarray([cf]))
        ref = jnp.asarray(scipy_expm(np.asarray(log_m) / cf))
        testing.assert_allclose(frame.vectors, ref[None], A, atol=1e-10)

    def test_parameter_gradient_matches_expm_frechet_adjoint(self):
        """The deformation's ``log_matrix`` gradient is the ``expm_frechet`` adjoint."""
        from itertools import product

        from scipy.linalg import expm_frechet

        log_m = np.array([[0.05, 0.10, -0.03], [0.02, 0.07, 0.04], [-0.06, 0.01, 0.08]])
        cell_grad = np.array(
            [[0.20, -0.07, 0.11], [-0.07, 0.05, 0.15], [0.11, 0.15, -0.20]]
        )
        frame = MatrixLogFrame(jnp.asarray(log_m)[None], jnp.ones(1))
        grad = frame.parameter_gradient(jnp.asarray(cell_grad)[None] * FORCE)
        ours = np.asarray(cast(units.Energy, grad.log_matrix[0]).to(u.eV))

        expected = np.zeros((3, 3))
        for mu, nu in product(range(3), repeat=2):
            d = np.zeros((3, 3))
            d[mu, nu] = 1.0
            expected[mu, nu] = np.sum(
                expm_frechet(log_m, d, compute_expm=False) * cell_grad
            )
        npt.assert_allclose(ours, expected, atol=1e-9)

    def test_base_roundtrip_exactness(self):
        """``DeformedFrame.from_frame(.., MatrixLogFrame)`` at identity deformation
        reproduces the triclinic reference, which it keeps as the base."""
        ref = TriclinicFrame.from_matrix(_M_TRICLINIC * A)
        frame = DeformedFrame.from_frame(
            ref, cell_factor=8.0, deformation=MatrixLogFrame
        )
        testing.assert_allclose(frame.vectors, ref.vectors, A, atol=1e-12)
        testing.assert_allclose(frame.reference_vectors, ref.vectors, A, atol=1e-12)

    def test_batched_safety_table_slice(self):
        defm = MatrixLogFrame(jnp.zeros((2, 3, 3)), jnp.ones(2))
        frame = _frechet(jnp.broadcast_to(_M_TRICLINIC, (2, 3, 3)) * A, defm)
        sliced = frame[0:1]
        testing.assert_allclose(sliced.vectors, _M_TRICLINIC, A, atol=1e-10)

    def test_rotated_cell_is_not_lower_triangular(self):
        log_m = jnp.asarray([[0.1, 0.2, 0.0], [0.0, 0.1, 0.0], [0.3, 0.0, 0.1]])
        frame = _frechet(jnp.eye(3)[None] * A, MatrixLogFrame(log_m[None], jnp.ones(1)))
        vecs = frame.vectors[0]
        upper = (vecs[0, 1] ** 2 + vecs[0, 2] ** 2 + vecs[1, 2] ** 2).to(A**2)
        assert upper > 1e-6  # genuinely non-triangular

    def test_rotated_cell_inverse_volume_roundtrip(self):
        log_m = jnp.asarray([[0.1, 0.2, 0.0], [0.0, 0.1, 0.0], [0.3, 0.0, 0.1]])
        frame = _frechet(_M_TRICLINIC * A, MatrixLogFrame(log_m[None], jnp.ones(1)))
        vecs = frame.vectors
        npt.assert_allclose(frame.inverse_vectors @ vecs, jnp.eye(3)[None], atol=1e-10)
        testing.assert_allclose(
            frame.volume, units.absolute(units.det(vecs)), A**3, atol=1e-10
        )
        r = jnp.array([[0.7, -0.3, 0.4]])
        testing.assert_allclose(
            frame.to_real(frame.to_fractional(r * A)), r, A, atol=1e-10
        )


_TRIL = jnp.array([[0.7, 0.5, 1.1, 0.1, 0.2, 1.4]])
_LOG_M = jnp.asarray([[[0.05, 0.10, -0.03], [0.02, 0.07, 0.04], [-0.06, 0.01, 0.08]]])


@pytest.mark.parametrize(
    "frame",
    [
        LogTriclinicFrame(_TRIL, jnp.ones(1)),
        MatrixLogFrame(_LOG_M, jnp.ones(1)),
        DeformedFrame(
            TriclinicFrame.from_matrix(_M_TRICLINIC * A),
            LogTriclinicFrame(_TRIL, jnp.ones(1)),
        ),
    ],
    ids=["log_triclinic", "matrix_log", "deformed"],
)
def test_zero_gradient_frame_labels(frame: Frame):
    """Each parameter of the zero gradient is labelled as parameter_gradient does."""
    force = units.relabel(units.ones_like(frame.vectors), units.dim_of(EV / A))
    probe = units.tree_leaves(frame.parameter_gradient(force))
    zero = units.tree_leaves(zero_gradient_frame(frame))
    assert [units.dim_of(x) for x in zero] == [units.dim_of(x) for x in probe]
    for leaf in zero:
        npt.assert_array_equal(units.magnitudes(leaf), 0.0)
