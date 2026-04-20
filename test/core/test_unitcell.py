# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for UnitCell types: TriclinicUnitCell, OrthorhombicUnitCell, Vacuum."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.unitcell import (
    BoundaryCondition,
    CoordinateSpace,
    OrthorhombicUnitCell,
    TriclinicUnitCell,
    UnitCell,
    Vacuum,
    min_multiplicity,
    to_lower_triangular,
)


class TestTriclinicUnitCell:
    def test_from_lattice_vectors(self):
        """Merged: cubic + orthorhombic + triclinic."""
        # cubic
        vecs = jnp.eye(3)
        cell = TriclinicUnitCell.from_matrix(vecs)
        npt.assert_allclose(cell.lattice_vectors, vecs)
        npt.assert_allclose(cell.inverse_lattice_vectors, jnp.eye(3))
        npt.assert_allclose(cell.volume, 1.0)

        # orthorhombic
        vecs = jnp.diag(jnp.array([2.0, 3.0, 4.0]))
        cell = TriclinicUnitCell.from_matrix(vecs)
        npt.assert_allclose(cell.lattice_vectors, vecs)
        npt.assert_allclose(
            cell.inverse_lattice_vectors,
            jnp.diag(jnp.array([0.5, 1.0 / 3.0, 0.25])),
        )
        npt.assert_allclose(cell.volume, 24.0)

        # triclinic
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]])
        cell = TriclinicUnitCell.from_matrix(vecs)
        npt.assert_allclose(cell.lattice_vectors, vecs)
        npt.assert_allclose(cell.volume, 1.0)
        npt.assert_allclose(
            cell.lattice_vectors @ cell.inverse_lattice_vectors,
            jnp.eye(3),
            atol=1e-10,
        )

    def test_from_lattice_vectors_negative_volume(self):
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        L, _ = to_lower_triangular(vecs)
        cell = TriclinicUnitCell.from_matrix(L)
        assert cell.volume > 0
        npt.assert_allclose(cell.volume, 1.0)

    def test_wrap_real_to_real(self):
        """Merged: single_point + multiple_points + edge_cases."""
        # single point
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3))
        r = jnp.array([1.5, -0.7, 2.3])
        npt.assert_allclose(cell.wrap(r), jnp.array([-0.5, 0.3, 0.3]), atol=1e-10)

        # multiple points
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3) * 2.0)
        r = jnp.array([[3.0, 1.0, -1.0], [-1.0, 4.0, 0.5], [0.0, 0.0, 0.0]])
        expected = jnp.array([[-1.0, -1.0, -1.0], [-1.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
        npt.assert_allclose(cell.wrap(r), expected, atol=1e-10)

        # edge cases (boundary)
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3))
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
        """Merged: real_to_reciprocal + reciprocal_to_real + reciprocal_to_reciprocal."""
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3) * 2.0)

        # real -> reciprocal
        r = jnp.array([1.0, 3.0, -1.0])
        wrapped = cell.wrap(
            r,
            input_space=CoordinateSpace.REAL,
            output_space=CoordinateSpace.FRACTIONAL,
        )
        npt.assert_allclose(wrapped, jnp.array([-0.5, -0.5, -0.5]), atol=1e-10)

        # reciprocal -> real
        r = jnp.array([1.2, -0.8, 0.3])
        wrapped = cell.wrap(
            r,
            input_space=CoordinateSpace.FRACTIONAL,
            output_space=CoordinateSpace.REAL,
        )
        npt.assert_allclose(wrapped, jnp.array([0.4, 0.4, 0.6]), atol=1e-6)

        # reciprocal -> reciprocal (use cubic cell)
        cell1 = TriclinicUnitCell.from_matrix(jnp.eye(3))
        r = jnp.array([1.7, -1.3, 0.8])
        wrapped = cell1.wrap(
            r,
            input_space=CoordinateSpace.FRACTIONAL,
            output_space=CoordinateSpace.FRACTIONAL,
        )
        npt.assert_allclose(wrapped, jnp.array([-0.3, -0.3, -0.2]), atol=1e-6)

    def test_wrap_non_orthogonal_and_shape(self):
        """Test wrapping with non-orthogonal cell and shape preservation."""
        # non-orthogonal
        vecs = jnp.array(
            [[1.0, 0.0, 0.0], [0.5, jnp.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]]
        )
        cell = TriclinicUnitCell.from_matrix(vecs)
        r = jnp.array([1.5, 1.0, 0.5])
        wrapped = cell.wrap(r)
        assert not jnp.allclose(wrapped, r)
        npt.assert_allclose(wrapped, cell.wrap(wrapped), atol=1e-10)

        # shape preservation
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3))
        for shape in [(3,), (5, 3), (2, 4, 3), (10, 1, 3)]:
            r = jnp.ones(shape) * 1.5
            assert cell.wrap(r).shape == shape

    def test_wrap_jit_and_gradient(self):
        """Merged: JIT compilation + gradient."""
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3))

        # JIT
        jit_wrap = jax.jit(cell.wrap)
        r = jnp.array([1.5, -0.7, 2.3])
        npt.assert_allclose(cell.wrap(r), jit_wrap(r), atol=1e-10)

        # gradient
        r = jnp.array([0.3, -0.2, 0.1])
        grad = jax.grad(lambda r: jnp.sum(cell.wrap(r)))(r)
        npt.assert_allclose(grad, jnp.array([1.0, 1.0, 1.0]), atol=1e-6)

    def test_orthogonality_and_volume(self):
        """Merged: reciprocal_lattice_orthogonality + volume_computation."""
        lattices = [
            (jnp.eye(3), 1.0),
            (jnp.diag(jnp.array([2.0, 3.0, 4.0])), 24.0),
            (jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, 0.3, 1.0]]), None),
            (jnp.diag(jnp.array([1.0, 2.0, 3.0])), 6.0),
        ]
        for vecs, expected_vol in lattices:
            cell = TriclinicUnitCell.from_matrix(vecs)
            npt.assert_allclose(
                cell.lattice_vectors @ cell.inverse_lattice_vectors,
                jnp.eye(3),
                atol=1e-10,
            )
            if expected_vol is not None:
                npt.assert_allclose(cell.volume, expected_vol, rtol=1e-10)

    def test_perpendicular_lengths(self):
        """Merged: cubic, ortho, triclinic, shape, positive, batched."""
        # cubic
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3) * 5.0)
        npt.assert_allclose(
            cell.perpendicular_lengths, jnp.array([5.0, 5.0, 5.0]), rtol=1e-10
        )

        # orthorhombic
        cell = TriclinicUnitCell.from_matrix(jnp.diag(jnp.array([2.0, 3.0, 4.0])))
        npt.assert_allclose(
            cell.perpendicular_lengths, jnp.array([2.0, 3.0, 4.0]), rtol=1e-10
        )

        # triclinic
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cell = TriclinicUnitCell.from_matrix(vecs)
        a, b, c = vecs
        V = cell.volume
        expected = jnp.array(
            [
                V / jnp.linalg.norm(jnp.cross(b, c)),
                V / jnp.linalg.norm(jnp.cross(a, c)),
                V / jnp.linalg.norm(jnp.cross(a, b)),
            ]
        )
        npt.assert_allclose(cell.perpendicular_lengths, expected, rtol=1e-10)

        # shape
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3))
        assert cell.perpendicular_lengths.shape == (3,)

        # positive
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, 0.3, 1.0]])
        cell = TriclinicUnitCell.from_matrix(vecs)
        assert jnp.all(cell.perpendicular_lengths > 0)

        # batched
        vecs = jnp.stack(
            [
                jnp.diag(jnp.array([2.0, 3.0, 4.0])),
                jnp.eye(3) * 5.0,
            ]
        )
        cell = TriclinicUnitCell.from_matrix(vecs)
        lengths = cell.perpendicular_lengths
        assert lengths.shape == (2, 3)
        npt.assert_allclose(lengths[0], [2.0, 3.0, 4.0], rtol=1e-10)
        npt.assert_allclose(lengths[1], [5.0, 5.0, 5.0], rtol=1e-10)

    def test_min_multiplicity(self):
        """Merged: cubic + orthorhombic + batched."""
        # cubic
        unitcell = TriclinicUnitCell.from_matrix(jnp.eye(3) * 10.0)
        npt.assert_array_equal(min_multiplicity(unitcell, 4.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(unitcell, 5.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(unitcell, 8.0), [2, 2, 2])

        # orthorhombic
        unitcell = TriclinicUnitCell.from_matrix(
            jnp.array([[8.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 6.0]]),
        )
        npt.assert_array_equal(min_multiplicity(unitcell, 2.5), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(unitcell, 5.0), [2, 1, 2])

        # batched
        vecs = jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 20.0])
        unitcell = TriclinicUnitCell.from_matrix(vecs)
        result = min_multiplicity(unitcell, 8.0)
        assert result.shape == (2, 3)
        npt.assert_array_equal(result[0], [2, 2, 2])
        npt.assert_array_equal(result[1], [1, 1, 1])


class TestOrthorhombicUnitCell:
    def test_lattice_vectors_diagonal(self):
        cell = OrthorhombicUnitCell(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(cell.lattice_vectors, jnp.diag(jnp.array([2.0, 3.0, 4.0])))

    def test_inverse_lattice_vectors(self):
        cell = OrthorhombicUnitCell(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(
            cell.lattice_vectors @ cell.inverse_lattice_vectors,
            jnp.eye(3),
            atol=1e-10,
        )

    def test_volume(self):
        cell = OrthorhombicUnitCell(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(cell.volume, 24.0)

    def test_perpendicular_lengths_equal_lengths(self):
        lengths = jnp.array([2.0, 3.0, 4.0])
        cell = OrthorhombicUnitCell(lengths)
        npt.assert_allclose(cell.perpendicular_lengths, lengths)

    def test_wrap_real_to_real(self):
        cell = OrthorhombicUnitCell(jnp.array([10.0, 10.0, 10.0]))
        r = jnp.array([12.0, -3.0, 25.0])
        wrapped = cell.wrap(r)
        npt.assert_allclose(wrapped, jnp.array([2.0, -3.0, -5.0]), atol=1e-10)

    def test_wrap_matches_triclinic(self):
        """Orthorhombic wrap must agree with equivalent TriclinicUnitCell."""
        lengths = jnp.array([2.0, 3.0, 4.0])
        ortho = OrthorhombicUnitCell(lengths)
        tri = TriclinicUnitCell.from_matrix(jnp.diag(lengths))
        r = jnp.array([3.5, -1.2, 7.8])
        npt.assert_allclose(ortho.wrap(r), tri.wrap(r), atol=1e-10)

    def test_wrap_cross_space(self):
        cell = OrthorhombicUnitCell(jnp.array([4.0, 4.0, 4.0]))
        r = jnp.array([3.0, -3.0, 5.0])
        frac = cell.wrap(
            r, input_space=CoordinateSpace.REAL, output_space=CoordinateSpace.FRACTIONAL
        )
        npt.assert_allclose(frac, jnp.array([-0.25, 0.25, 0.25]), atol=1e-10)

    def test_batched(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        cell = OrthorhombicUnitCell(lengths)
        assert cell.volume.shape == (2,)
        npt.assert_allclose(cell.volume, jnp.array([24.0, 210.0]))

    def test_slicing(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        cell = OrthorhombicUnitCell(lengths)
        sub = cell[0]
        npt.assert_allclose(sub.lengths, jnp.array([2.0, 3.0, 4.0]))

    def test_satisfies_unitcell_protocol(self):
        cell = OrthorhombicUnitCell(jnp.array([1.0, 1.0, 1.0]))
        assert isinstance(cell, UnitCell)
        assert isinstance(cell, BoundaryCondition)

    def test_is_jax_pytree(self):
        cell = OrthorhombicUnitCell(jnp.array([2.0, 3.0, 4.0]))
        leaves = jax.tree.leaves(cell)
        assert len(leaves) == 1
        scaled = jax.tree.map(lambda x: x * 2, cell)
        npt.assert_allclose(scaled.lengths, jnp.array([4.0, 6.0, 8.0]))


class TestVacuum:
    def test_periodic_is_false(self):
        v = Vacuum()
        assert v.periodic == (False, False, False)

    def test_wrap_is_identity(self):
        v = Vacuum()
        r = jnp.array([100.0, -50.0, 0.3])
        npt.assert_allclose(v.wrap(r), r)

    def test_wrap_identity_all_spaces(self):
        v = Vacuum()
        r = jnp.array([1.5, -0.7, 2.3])
        for in_s in CoordinateSpace:
            for out_s in CoordinateSpace:
                npt.assert_allclose(v.wrap(r, input_space=in_s, output_space=out_s), r)

    def test_satisfies_boundary_condition(self):
        v = Vacuum()
        assert isinstance(v, BoundaryCondition)

    def test_does_not_satisfy_unitcell(self):
        v = Vacuum()
        assert not isinstance(v, UnitCell)


class TestProtocolSatisfaction:
    def test_triclinic_satisfies_unitcell(self):
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3))
        assert isinstance(cell, UnitCell)
        assert isinstance(cell, BoundaryCondition)

    def test_orthorhombic_satisfies_unitcell(self):
        cell = OrthorhombicUnitCell(jnp.array([1.0, 1.0, 1.0]))
        assert isinstance(cell, UnitCell)
        assert isinstance(cell, BoundaryCondition)
