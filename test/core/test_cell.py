# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Frame and Cell types."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.cell import (
    Cell,
    CoordinateSpace,
    OrthogonalFrame,
    PeriodicCell,
    TriclinicFrame,
    VacuumCell,
    is_3d_periodic,
    is_vacuum,
    make_supercell,
    min_multiplicity,
    to_lower_triangular,
)
from kups.core.lens import lens


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

    def test_from_matrix_extracts_diagonal(self):
        vecs = jnp.array([[2.0, 0.7, 0.0], [0.0, 3.0, 0.0], [0.4, 0.0, 4.0]])
        frame = OrthogonalFrame.from_matrix(vecs)
        npt.assert_allclose(frame.lengths, jnp.array([2.0, 3.0, 4.0]))

    def test_from_matrix_batched(self):
        vecs = jnp.stack([jnp.diag(jnp.array([2.0, 3.0, 4.0]))] * 2)
        frame = OrthogonalFrame.from_matrix(vecs)
        assert frame.lengths.shape == (2, 3)
        npt.assert_allclose(frame.lengths[0], jnp.array([2.0, 3.0, 4.0]))

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


class TestCellConstructors:
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

    @pytest.mark.parametrize("cls", [PeriodicCell, VacuumCell])
    def test_periodic_arg_is_rejected(self, cls):
        frame = OrthogonalFrame(jnp.array([1.0, 1.0, 1.0]))
        with pytest.raises(TypeError, match="periodic"):
            cls(frame, periodic=(True, False, True))  # type: ignore[call-arg]


class TestTypeGuards:
    def test_is_vacuum_positive(self):
        c = VacuumCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0])))
        assert is_vacuum(c)
        assert not is_3d_periodic(c)

    def test_is_3d_periodic_positive(self):
        c = PeriodicCell(OrthogonalFrame(jnp.array([1.0, 1.0, 1.0])))
        assert is_3d_periodic(c)
        assert not is_vacuum(c)


class TestPeriodicPreservedUnderScaling:
    """Scaling a cell must preserve its concrete type."""

    def test_orthogonal_mul_preserves_vacuum(self):
        c = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        scaled = c * 2.0
        assert isinstance(scaled, VacuumCell)
        assert scaled.periodic == (False, False, False)
        assert isinstance(scaled.frame, OrthogonalFrame)
        npt.assert_allclose(scaled.frame.lengths, jnp.array([20.0, 20.0, 20.0]))

    def test_triclinic_mul_preserves_periodic(self):
        c = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * 5.0))
        scaled = c * 2.0
        assert isinstance(scaled, PeriodicCell)
        assert scaled.periodic == (True, True, True)


class TestLensReplacePreservesConcreteType:
    def test_lens_replace_periodic(self):
        from kups.core.lens import bind

        c = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        new_frame = OrthogonalFrame(jnp.array([5.0, 5.0, 5.0]))
        c2 = bind(c, lambda x: x.frame).set(new_frame)
        assert isinstance(c2, PeriodicCell)
        assert c2.periodic == (True, True, True)

    def test_lens_replace_vacuum(self):
        from kups.core.lens import bind

        c = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 30.0])))
        new_frame = OrthogonalFrame(jnp.array([5.0, 5.0, 10.0]))
        c2 = bind(c, lambda x: x.frame).set(new_frame)
        assert isinstance(c2, VacuumCell)
        assert c2.periodic == (False, False, False)


class TestVacuumWrap:
    def test_vacuum_wrap_is_identity(self):
        """VacuumCell has no periodic axes — wrap must leave coordinates alone."""
        c = VacuumCell(TriclinicFrame.from_matrix(jnp.eye(3) * 5.0))
        r = jnp.array([[15.0, -3.0, 25.0], [-50.0, 100.0, 0.5]])
        npt.assert_allclose(c.wrap(r), r, atol=1e-6)


class TestCellSlicing:
    def test_slice_periodic_orthogonal(self):
        cell = PeriodicCell(
            OrthogonalFrame(jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
        )
        sub = cell[0]
        assert isinstance(sub, PeriodicCell)
        assert isinstance(sub.frame, OrthogonalFrame)
        npt.assert_allclose(sub.frame.lengths, jnp.array([2.0, 3.0, 4.0]))
        assert sub.periodic == (True, True, True)

    def test_slice_vacuum_triclinic(self):
        cells = jnp.stack([jnp.eye(3) * 5.0, jnp.eye(3) * 10.0])
        cell = VacuumCell(TriclinicFrame.from_matrix(cells))
        sub = cell[1]
        assert isinstance(sub, VacuumCell)
        npt.assert_allclose(sub.vectors, jnp.eye(3) * 10.0, atol=1e-6)
        assert sub.periodic == (False, False, False)

    def test_add_batch_dim(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([2.0, 3.0, 4.0])))
        batched = cell[None]
        assert isinstance(batched, PeriodicCell)
        frame = batched.frame
        assert isinstance(frame, OrthogonalFrame)
        assert frame.lengths.shape == (1, 3)


class TestFrameTile:
    def test_triclinic_tile_expands_per_axis(self):
        """The triclinic [m0, m1, m1, m2, m2, m2] tril expansion is non-obvious;
        verify a non-cubic, non-uniform multiplicity gives the right scaled
        basis matrix."""
        frame = TriclinicFrame.from_matrix(
            jnp.array([[1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.3, 0.4, 3.0]])
        )
        tiled = frame.tile((2, 3, 4))
        expected = jnp.array([[2.0, 0.0, 0.0], [1.5, 6.0, 0.0], [1.2, 1.6, 12.0]])
        npt.assert_allclose(tiled.vectors, expected, atol=1e-6)


class TestMakeSupercell:
    def test_periodic_orthogonal_replicates_frame_and_data(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        positions = jnp.array([[0.0, 0.0, 0.0]])
        new_cell, new_positions = make_supercell(
            cell, (2, 1, 1), positions, lens(lambda x: x)
        )
        assert isinstance(new_cell, PeriodicCell)
        assert isinstance(new_cell.frame, OrthogonalFrame)
        npt.assert_allclose(new_cell.frame.lengths, jnp.array([20.0, 10.0, 10.0]))
        # 1 particle x 2 replicas = 2 particles
        assert new_positions.shape == (2, 3)

    def test_periodic_int_multiplicity_expanded(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        positions = jnp.array([[0.0, 0.0, 0.0]])
        new_cell, new_positions = make_supercell(cell, 3, positions, lens(lambda x: x))
        new_frame = new_cell.frame
        assert isinstance(new_frame, OrthogonalFrame)
        npt.assert_allclose(new_frame.lengths, jnp.array([30.0, 30.0, 30.0]))
        assert new_positions.shape == (27, 3)  # 3^3

    def test_vacuum_clamps_to_one(self):
        """Vacuum cells have no periodic axes — multiplicities must clamp to 1."""
        cell = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        new_cell, new_positions = make_supercell(
            cell, (2, 2, 2), positions, lens(lambda x: x)
        )
        assert isinstance(new_cell, VacuumCell)
        # frame unchanged (clamped to 1 on every axis)
        new_frame = new_cell.frame
        assert isinstance(new_frame, OrthogonalFrame)
        npt.assert_allclose(new_frame.lengths, jnp.array([10.0, 10.0, 10.0]))
        # positions unchanged (n_reps = 1)
        assert new_positions.shape == positions.shape

    def test_triclinic_supercell(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * 5.0))
        positions = jnp.array([[0.0, 0.0, 0.0]])
        new_cell, _ = make_supercell(cell, (2, 2, 2), positions, lens(lambda x: x))
        assert isinstance(new_cell, PeriodicCell)
        npt.assert_allclose(new_cell.vectors, jnp.eye(3) * 10.0, atol=1e-6)

    def test_concrete_subclass_preserved(self):
        """The returned cell type matches the input cell type."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        for cell_cls in (PeriodicCell, VacuumCell):
            cell = cell_cls(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
            new_cell, _ = make_supercell(cell, 2, positions, lens(lambda x: x))
            assert type(new_cell) is cell_cls

    def test_tuple_data_lens_picks_positions(self):
        """Lens can focus on a specific element of structured replicate-data."""
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        positions = jnp.array([[0.0, 0.0, 0.0]])
        charges = jnp.array([1.0])
        _, (new_positions, new_charges) = make_supercell(
            cell, (2, 1, 1), (positions, charges), lens(lambda x: x[0])
        )
        assert new_positions.shape == (2, 3)
        assert new_charges.shape == (2,)

    def test_zero_multiplicity_rejected(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        positions = jnp.array([[0.0, 0.0, 0.0]])
        with pytest.raises(AssertionError):
            make_supercell(cell, (0, 1, 1), positions, lens(lambda x: x))


class TestMinMultiplicityVacuum:
    def test_vacuum_returns_ones(self):
        """Non-periodic axes don't need replication regardless of cutoff."""
        cell = VacuumCell(OrthogonalFrame(jnp.array([5.0, 5.0, 5.0])))
        # cutoff > box size would imply replication for periodic, but vacuum clamps to 1
        npt.assert_array_equal(min_multiplicity(cell, 100.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 1.0), [1, 1, 1])


class TestToLowerTriangular:
    def test_already_lower_triangular_passes_through(self):
        L = jnp.array([[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.1, 0.2, 4.0]])
        L_out, mapper = to_lower_triangular(L)
        npt.assert_allclose(L_out, L, atol=1e-6)
        # On lower-triangular input the mapper is the identity rotation
        r = jnp.array([1.0, 2.0, 3.0])
        npt.assert_allclose(mapper(r), r, atol=1e-6)

    def test_swap_yz_yields_negative_volume_input(self):
        """Input with negative determinant: output volume is positive."""
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        L, _ = to_lower_triangular(vecs)
        frame = TriclinicFrame.from_matrix(L)
        assert float(frame.volume) > 0
        npt.assert_allclose(frame.volume, 1.0, atol=1e-6)

    def test_mapper_preserves_distances(self):
        """The orthogonal rotation preserves vector norms."""
        vecs = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        _, mapper = to_lower_triangular(vecs)
        r = jnp.array([1.0, 2.0, 3.0])
        npt.assert_allclose(jnp.linalg.norm(mapper(r)), jnp.linalg.norm(r), atol=1e-6)

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
        L, _ = to_lower_triangular(vecs)
        assert jnp.all(jnp.diagonal(L) > 0)

    def test_round_trip_recovers_input(self):
        """``vecs == L @ Q.T``, where Q is the rotation reconstructed from
        the mapper. The mapper acts as ``r @ Q`` in row form (= ``Q.T @ r``
        on a column vector), matching the same rotation that takes
        ``vecs`` row-by-row into the rows of ``L``."""
        vecs = jnp.array([[1.0, 2.0, 0.3], [0.2, 3.0, 0.1], [0.5, 0.4, 4.0]])
        L, mapper = to_lower_triangular(vecs)
        Q = jnp.stack([mapper(row) for row in jnp.eye(3)])
        npt.assert_allclose(L @ Q.T, vecs, atol=1e-6)

    def test_mapper_preserves_fractional_coordinates(self):
        """The mapper re-expresses positions without changing fractional coordinates."""
        vecs = jnp.array([[1.0, 2.0, 0.3], [0.2, 3.0, 0.1], [0.5, 0.4, 4.0]])
        L, mapper = to_lower_triangular(vecs)
        # Several arbitrary fractional positions; all must survive.
        frac_coords = jnp.array(
            [[0.7, 0.3, 0.5], [0.0, 0.0, 0.0], [0.123, 0.456, 0.789]]
        )
        for f in frac_coords:
            r_orig = f @ vecs
            r_new = mapper(r_orig)
            f_new = r_new @ jnp.linalg.inv(L)
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
        L, mapper = to_lower_triangular(vecs)
        r_orig = frac @ vecs
        r_new = jnp.stack([mapper(r) for r in r_orig])
        frac_new = r_new @ jnp.linalg.inv(L)
        npt.assert_allclose(frac_new, frac, atol=1e-5)

    def test_fcc_primitive_gives_positive_diagonal(self):
        """Regression: ase FCC Cu (a=3.61) previously produced a negative diagonal."""
        import ase.build

        vecs = jnp.asarray(ase.build.bulk("Cu", "fcc", a=3.61).cell.array)
        L, _ = to_lower_triangular(vecs)
        assert jnp.all(jnp.diagonal(L) > 0)


_FRAMES = pytest.mark.parametrize(
    "frame",
    [
        OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])),
        TriclinicFrame.from_matrix(jnp.eye(3) * 10.0),
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
        wrapped = cell.wrap(jnp.array([12.0, 0.0, 0.0]))
        npt.assert_allclose(wrapped, jnp.array([2.0, 0.0, 0.0]), atol=1e-6)

    @_FRAMES
    def test_vacuum_wrap_is_identity(self, frame):
        cell = VacuumCell(frame)
        r = jnp.array([12.0, -3.0, 25.0])
        npt.assert_allclose(cell.wrap(r), r, atol=1e-6)

    @_FRAMES
    def test_periodic_min_multiplicity_grows_with_cutoff(self, frame):
        cell = PeriodicCell(frame)
        npt.assert_array_equal(min_multiplicity(cell, 4.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 8.0), [2, 2, 2])

    @_FRAMES
    def test_vacuum_min_multiplicity_always_one(self, frame):
        cell = VacuumCell(frame)
        # cutoff bigger than the box would force replication if periodic
        npt.assert_array_equal(min_multiplicity(cell, 100.0), [1, 1, 1])

    @_FRAMES
    def test_periodic_supercell_replicates(self, frame):
        cell = PeriodicCell(frame)
        positions = jnp.array([[0.0, 0.0, 0.0]])
        new_cell, new_positions = make_supercell(
            cell, (2, 2, 2), positions, lens(lambda x: x)
        )
        npt.assert_allclose(new_cell.volume, 8 * cell.volume, rtol=1e-6)
        assert new_positions.shape == (8, 3)

    @_FRAMES
    def test_vacuum_supercell_clamps(self, frame):
        cell = VacuumCell(frame)
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        new_cell, new_positions = make_supercell(
            cell, (2, 3, 4), positions, lens(lambda x: x)
        )
        # frame untouched; data not replicated
        npt.assert_allclose(new_cell.volume, cell.volume, rtol=1e-6)
        assert new_positions.shape == positions.shape


class TestPytreeAndJIT:
    def test_periodic_is_static_aux_not_a_leaf(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        assert len(jax.tree.leaves(cell)) == 1

    def test_jit_traces_through_wrap(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        r = jnp.array([12.0, -3.0, 25.0])
        npt.assert_allclose(jax.jit(cell.wrap)(r), cell.wrap(r), atol=1e-6)

    def test_tree_map_preserves_concrete_subclass(self):
        cell = VacuumCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        scaled = jax.tree.map(lambda x: x * 2, cell)
        assert isinstance(scaled, VacuumCell)


class TestFromLengthsAndAngles:
    def test_roundtrip(self):
        """Construct from (lengths, angles); read them back."""
        lengths = jnp.array([5.0, 6.0, 7.0])
        angles = jnp.array([85.0, 95.0, 100.0])
        frame = TriclinicFrame.from_lengths_and_angles(lengths, angles)
        npt.assert_allclose(frame.lengths, lengths, atol=1e-6)
        npt.assert_allclose(frame.angles, angles, atol=1e-6)
