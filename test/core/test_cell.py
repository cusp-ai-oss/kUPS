# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Frame and Cell types."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.cell import (
    Cell,
    CoordinateSpace,
    OrthogonalFrame,
    PeriodicCell,
    SlabCell,
    TriclinicFrame,
    VacuumCell,
    is_fully_periodic,
    is_vacuum,
    min_multiplicity,
    to_lower_triangular,
)


class TestTriclinicFrame:
    def test_from_matrix(self):
        # cubic
        vecs = jnp.eye(3)
        frame = TriclinicFrame.from_matrix(vecs)
        npt.assert_allclose(frame.vectors, vecs)
        npt.assert_allclose(frame.inverse_vectors, jnp.eye(3))
        npt.assert_allclose(frame.volume, 1.0)

        # orthorhombic
        vecs = jnp.diag(jnp.array([2.0, 3.0, 4.0]))
        frame = TriclinicFrame.from_matrix(vecs)
        npt.assert_allclose(frame.vectors, vecs)
        npt.assert_allclose(
            frame.inverse_vectors,
            jnp.diag(jnp.array([0.5, 1.0 / 3.0, 0.25])),
        )
        npt.assert_allclose(frame.volume, 24.0)

        # triclinic
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]])
        frame = TriclinicFrame.from_matrix(vecs)
        npt.assert_allclose(frame.vectors, vecs)
        npt.assert_allclose(frame.volume, 1.0)
        npt.assert_allclose(
            frame.vectors @ frame.inverse_vectors,
            jnp.eye(3),
            atol=1e-10,
        )

    def test_from_matrix_negative_volume(self):
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        L, _ = to_lower_triangular(vecs)
        frame = TriclinicFrame.from_matrix(L)
        assert frame.volume > 0
        npt.assert_allclose(frame.volume, 1.0)

    def test_wrap_real_to_real(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)))
        r = jnp.array([1.5, -0.7, 2.3])
        npt.assert_allclose(cell.wrap(r), jnp.array([-0.5, 0.3, 0.3]), atol=1e-10)

        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * 2.0))
        r = jnp.array([[3.0, 1.0, -1.0], [-1.0, 4.0, 0.5], [0.0, 0.0, 0.0]])
        expected = jnp.array([[-1.0, -1.0, -1.0], [-1.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
        npt.assert_allclose(cell.wrap(r), expected, atol=1e-10)

        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)))
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
        npt.assert_allclose(cell.wrap(r), expected, atol=1e-10)

    def test_wrap_cross_space(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * 2.0))

        r = jnp.array([1.0, 3.0, -1.0])
        wrapped = cell.wrap(
            r,
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
        npt.assert_allclose(wrapped, jnp.array([0.4, 0.4, 0.6]), atol=1e-6)

        cell1 = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)))
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
        cell = PeriodicCell(TriclinicFrame.from_matrix(vecs))
        r = jnp.array([1.5, 1.0, 0.5])
        wrapped = cell.wrap(r)
        assert not jnp.allclose(wrapped, r)
        npt.assert_allclose(wrapped, cell.wrap(wrapped), atol=1e-10)

        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)))
        for shape in [(3,), (5, 3), (2, 4, 3), (10, 1, 3)]:
            r = jnp.ones(shape) * 1.5
            assert cell.wrap(r).shape == shape

    def test_wrap_jit_and_gradient(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)))

        jit_wrap = jax.jit(cell.wrap)
        r = jnp.array([1.5, -0.7, 2.3])
        npt.assert_allclose(cell.wrap(r), jit_wrap(r), atol=1e-10)

        r = jnp.array([0.3, -0.2, 0.1])
        grad = jax.grad(lambda r: jnp.sum(cell.wrap(r)))(r)
        npt.assert_allclose(grad, jnp.array([1.0, 1.0, 1.0]), atol=1e-6)

    def test_orthogonality_and_volume(self):
        lattices = [
            (jnp.eye(3), 1.0),
            (jnp.diag(jnp.array([2.0, 3.0, 4.0])), 24.0),
            (jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, 0.3, 1.0]]), None),
            (jnp.diag(jnp.array([1.0, 2.0, 3.0])), 6.0),
        ]
        for vecs, expected_vol in lattices:
            frame = TriclinicFrame.from_matrix(vecs)
            npt.assert_allclose(
                frame.vectors @ frame.inverse_vectors,
                jnp.eye(3),
                atol=1e-10,
            )
            if expected_vol is not None:
                npt.assert_allclose(frame.volume, expected_vol, rtol=1e-10)

    def test_perpendicular_lengths(self):
        frame = TriclinicFrame.from_matrix(jnp.eye(3) * 5.0)
        npt.assert_allclose(
            frame.perpendicular_lengths, jnp.array([5.0, 5.0, 5.0]), rtol=1e-10
        )

        frame = TriclinicFrame.from_matrix(jnp.diag(jnp.array([2.0, 3.0, 4.0])))
        npt.assert_allclose(
            frame.perpendicular_lengths, jnp.array([2.0, 3.0, 4.0]), rtol=1e-10
        )

        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
        frame = TriclinicFrame.from_matrix(vecs)
        a, b, c = vecs
        V = frame.volume
        expected = jnp.array(
            [
                V / jnp.linalg.norm(jnp.cross(b, c)),
                V / jnp.linalg.norm(jnp.cross(a, c)),
                V / jnp.linalg.norm(jnp.cross(a, b)),
            ]
        )
        npt.assert_allclose(frame.perpendicular_lengths, expected, rtol=1e-10)

        frame = TriclinicFrame.from_matrix(jnp.eye(3))
        assert frame.perpendicular_lengths.shape == (3,)

        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, 0.3, 1.0]])
        frame = TriclinicFrame.from_matrix(vecs)
        assert jnp.all(frame.perpendicular_lengths > 0)

        vecs = jnp.stack(
            [
                jnp.diag(jnp.array([2.0, 3.0, 4.0])),
                jnp.eye(3) * 5.0,
            ]
        )
        frame = TriclinicFrame.from_matrix(vecs)
        lengths = frame.perpendicular_lengths
        assert lengths.shape == (2, 3)
        npt.assert_allclose(lengths[0], [2.0, 3.0, 4.0], rtol=1e-10)
        npt.assert_allclose(lengths[1], [5.0, 5.0, 5.0], rtol=1e-10)

    def test_min_multiplicity(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * 10.0))
        npt.assert_array_equal(min_multiplicity(cell, 4.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 5.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 8.0), [2, 2, 2])

        cell = PeriodicCell(
            TriclinicFrame.from_matrix(
                jnp.array([[8.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 6.0]]),
            )
        )
        npt.assert_array_equal(min_multiplicity(cell, 2.5), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 5.0), [2, 1, 2])

        vecs = jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 20.0])
        cell = PeriodicCell(TriclinicFrame.from_matrix(vecs))
        result = min_multiplicity(cell, 8.0)
        assert result.shape == (2, 3)
        npt.assert_array_equal(result[0], [2, 2, 2])
        npt.assert_array_equal(result[1], [1, 1, 1])


class TestOrthogonalFrame:
    def test_vectors_diagonal(self):
        frame = OrthogonalFrame(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(frame.vectors, jnp.diag(jnp.array([2.0, 3.0, 4.0])))

    def test_inverse_vectors(self):
        frame = OrthogonalFrame(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(
            frame.vectors @ frame.inverse_vectors,
            jnp.eye(3),
            atol=1e-10,
        )

    def test_volume(self):
        frame = OrthogonalFrame(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(frame.volume, 24.0)

    def test_perpendicular_lengths_equal_lengths(self):
        lengths = jnp.array([2.0, 3.0, 4.0])
        frame = OrthogonalFrame(lengths)
        npt.assert_allclose(frame.perpendicular_lengths, lengths)

    def test_wrap_real_to_real(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        r = jnp.array([12.0, -3.0, 25.0])
        wrapped = cell.wrap(r)
        npt.assert_allclose(wrapped, jnp.array([2.0, -3.0, -5.0]), atol=1e-10)

    def test_wrap_matches_triclinic(self):
        """Orthogonal wrap must agree with equivalent TriclinicFrame."""
        lengths = jnp.array([2.0, 3.0, 4.0])
        ortho = PeriodicCell(OrthogonalFrame(lengths))
        tri = PeriodicCell(TriclinicFrame.from_matrix(jnp.diag(lengths)))
        r = jnp.array([3.5, -1.2, 7.8])
        npt.assert_allclose(ortho.wrap(r), tri.wrap(r), atol=1e-10)

    def test_wrap_cross_space(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([4.0, 4.0, 4.0])))
        r = jnp.array([3.0, -3.0, 5.0])
        frac = cell.wrap(
            r, input_space=CoordinateSpace.REAL, output_space=CoordinateSpace.FRACTIONAL
        )
        npt.assert_allclose(frac, jnp.array([-0.25, 0.25, 0.25]), atol=1e-10)

    def test_batched(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        frame = OrthogonalFrame(lengths)
        assert frame.volume.shape == (2,)
        npt.assert_allclose(frame.volume, jnp.array([24.0, 210.0]))

    def test_slicing(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        frame = OrthogonalFrame(lengths)
        sub = frame[0]
        npt.assert_allclose(sub.lengths, jnp.array([2.0, 3.0, 4.0]))

    def test_cell_isinstance(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0])))
        assert isinstance(cell, Cell)

    def test_is_jax_pytree(self):
        frame = OrthogonalFrame(jnp.array([2.0, 3.0, 4.0]))
        leaves = jax.tree.leaves(frame)
        assert len(leaves) == 1
        scaled = jax.tree.map(lambda x: x * 2, frame)
        npt.assert_allclose(scaled.lengths, jnp.array([4.0, 6.0, 8.0]))


class TestCellIsinstance:
    def test_periodic_triclinic(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)))
        assert isinstance(cell, Cell)

    def test_periodic_orthogonal(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0])))
        assert isinstance(cell, Cell)

    def test_vacuum(self):
        cell = VacuumCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0])))
        assert isinstance(cell, Cell)

    def test_slab(self):
        cell = SlabCell(
            OrthogonalFrame(jnp.array([1.0, 1.0, 1.0])),
            periodic=(True, True, False),
        )
        assert isinstance(cell, Cell)


class TestCellConstructors:
    """Runtime behavior of PeriodicCell / VacuumCell / SlabCell.

    Type-level discrimination — that ``PeriodicCell`` is rejected where
    ``VacuumCell`` is expected and vice versa — is enforced by pyright at
    static-analysis time. Here we just lock in the runtime semantics.
    """

    def test_periodic_default_orthogonal(self):
        c = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        assert c.periodic == (True, True, True)
        assert isinstance(c.frame, OrthogonalFrame)
        npt.assert_allclose(c.frame.lengths, jnp.array([10.0, 10.0, 10.0]))

    def test_vacuum_orthogonal(self):
        c = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        assert c.periodic == (False, False, False)
        assert isinstance(c.frame, OrthogonalFrame)
        npt.assert_allclose(c.frame.lengths, jnp.array([10.0, 10.0, 10.0]))

    def test_vacuum_triclinic(self):
        c = VacuumCell(TriclinicFrame.from_matrix(jnp.eye(3) * 2.0))
        assert c.periodic == (False, False, False)
        npt.assert_allclose(c.vectors, jnp.eye(3) * 2.0)

    def test_slab_orthogonal(self):
        c = SlabCell(
            OrthogonalFrame(jnp.array([10.0, 10.0, 30.0])),
            periodic=(True, True, False),
        )
        assert c.periodic == (True, True, False)

    def test_slab_triclinic(self):
        c = SlabCell(
            TriclinicFrame.from_matrix(jnp.eye(3) * 5.0),
            periodic=(True, True, False),
        )
        assert c.periodic == (True, True, False)


class TestTypeGuards:
    def test_is_vacuum_positive(self):
        c = VacuumCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0])))
        assert is_vacuum(c)
        assert not is_fully_periodic(c)

    def test_is_fully_periodic_positive(self):
        c = PeriodicCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0])))
        assert is_fully_periodic(c)
        assert not is_vacuum(c)

    def test_slab_is_neither(self):
        c = SlabCell(
            OrthogonalFrame(jnp.array([10.0, 10.0, 30.0])),
            periodic=(True, True, False),
        )
        assert not is_vacuum(c)
        assert not is_fully_periodic(c)


class TestPeriodicPreservedUnderScaling:
    """Scaling a cell must preserve its concrete type and periodicity."""

    def test_orthogonal_mul_preserves_vacuum(self):
        c = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        scaled = c * 2.0
        assert isinstance(scaled, VacuumCell)
        assert scaled.periodic == (False, False, False)
        assert isinstance(scaled.frame, OrthogonalFrame)
        npt.assert_allclose(scaled.frame.lengths, jnp.array([20.0, 20.0, 20.0]))

    def test_triclinic_mul_preserves_slab(self):
        c = SlabCell(
            TriclinicFrame.from_matrix(jnp.eye(3) * 5.0),
            periodic=(True, True, False),
        )
        scaled = c * 2.0
        assert isinstance(scaled, SlabCell)
        assert scaled.periodic == (True, True, False)


class TestDataclassReplacePreservesConcreteType:
    def test_replace_periodic(self):
        c = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        new_frame = OrthogonalFrame(jnp.array([5.0, 5.0, 5.0]))
        c2 = dataclasses.replace(c, frame=new_frame)
        assert isinstance(c2, PeriodicCell)
        assert c2.periodic == (True, True, True)

    def test_replace_slab_keeps_runtime_periodic(self):
        c = SlabCell(
            OrthogonalFrame(jnp.array([10.0, 10.0, 30.0])),
            periodic=(True, True, False),
        )
        new_frame = OrthogonalFrame(jnp.array([5.0, 5.0, 10.0]))
        c2 = dataclasses.replace(c, frame=new_frame)
        assert isinstance(c2, SlabCell)
        assert c2.periodic == (True, True, False)
