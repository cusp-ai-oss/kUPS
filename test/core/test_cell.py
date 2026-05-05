# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Lattice and Cell types."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.cell import (
    Cell,
    CoordinateSpace,
    OrthogonalLattice,
    PeriodicCell,
    SlabCell,
    TriclinicLattice,
    VacuumCell,
    is_fully_periodic,
    is_vacuum,
    min_multiplicity,
    to_lower_triangular,
)


class TestTriclinicLattice:
    def test_from_matrix(self):
        """Merged: cubic + orthorhombic + triclinic."""
        # cubic
        vecs = jnp.eye(3)
        lat = TriclinicLattice.from_matrix(vecs)
        npt.assert_allclose(lat.lattice_vectors, vecs)
        npt.assert_allclose(lat.inverse_lattice_vectors, jnp.eye(3))
        npt.assert_allclose(lat.volume, 1.0)

        # orthorhombic
        vecs = jnp.diag(jnp.array([2.0, 3.0, 4.0]))
        lat = TriclinicLattice.from_matrix(vecs)
        npt.assert_allclose(lat.lattice_vectors, vecs)
        npt.assert_allclose(
            lat.inverse_lattice_vectors,
            jnp.diag(jnp.array([0.5, 1.0 / 3.0, 0.25])),
        )
        npt.assert_allclose(lat.volume, 24.0)

        # triclinic
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]])
        lat = TriclinicLattice.from_matrix(vecs)
        npt.assert_allclose(lat.lattice_vectors, vecs)
        npt.assert_allclose(lat.volume, 1.0)
        npt.assert_allclose(
            lat.lattice_vectors @ lat.inverse_lattice_vectors,
            jnp.eye(3),
            atol=1e-10,
        )

    def test_from_matrix_negative_volume(self):
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        L, _ = to_lower_triangular(vecs)
        lat = TriclinicLattice.from_matrix(L)
        assert lat.volume > 0
        npt.assert_allclose(lat.volume, 1.0)

    def test_wrap_real_to_real(self):
        """Merged: single_point + multiple_points + edge_cases."""
        cell = PeriodicCell(TriclinicLattice.from_matrix(jnp.eye(3)))
        r = jnp.array([1.5, -0.7, 2.3])
        npt.assert_allclose(cell.wrap(r), jnp.array([-0.5, 0.3, 0.3]), atol=1e-10)

        cell = PeriodicCell(TriclinicLattice.from_matrix(jnp.eye(3) * 2.0))
        r = jnp.array([[3.0, 1.0, -1.0], [-1.0, 4.0, 0.5], [0.0, 0.0, 0.0]])
        expected = jnp.array([[-1.0, -1.0, -1.0], [-1.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
        npt.assert_allclose(cell.wrap(r), expected, atol=1e-10)

        cell = PeriodicCell(TriclinicLattice.from_matrix(jnp.eye(3)))
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
        cell = PeriodicCell(TriclinicLattice.from_matrix(jnp.eye(3) * 2.0))

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

        cell1 = PeriodicCell(TriclinicLattice.from_matrix(jnp.eye(3)))
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
        cell = PeriodicCell(TriclinicLattice.from_matrix(vecs))
        r = jnp.array([1.5, 1.0, 0.5])
        wrapped = cell.wrap(r)
        assert not jnp.allclose(wrapped, r)
        npt.assert_allclose(wrapped, cell.wrap(wrapped), atol=1e-10)

        cell = PeriodicCell(TriclinicLattice.from_matrix(jnp.eye(3)))
        for shape in [(3,), (5, 3), (2, 4, 3), (10, 1, 3)]:
            r = jnp.ones(shape) * 1.5
            assert cell.wrap(r).shape == shape

    def test_wrap_jit_and_gradient(self):
        cell = PeriodicCell(TriclinicLattice.from_matrix(jnp.eye(3)))

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
            lat = TriclinicLattice.from_matrix(vecs)
            npt.assert_allclose(
                lat.lattice_vectors @ lat.inverse_lattice_vectors,
                jnp.eye(3),
                atol=1e-10,
            )
            if expected_vol is not None:
                npt.assert_allclose(lat.volume, expected_vol, rtol=1e-10)

    def test_perpendicular_lengths(self):
        # cubic
        lat = TriclinicLattice.from_matrix(jnp.eye(3) * 5.0)
        npt.assert_allclose(
            lat.perpendicular_lengths, jnp.array([5.0, 5.0, 5.0]), rtol=1e-10
        )

        # orthorhombic
        lat = TriclinicLattice.from_matrix(jnp.diag(jnp.array([2.0, 3.0, 4.0])))
        npt.assert_allclose(
            lat.perpendicular_lengths, jnp.array([2.0, 3.0, 4.0]), rtol=1e-10
        )

        # triclinic
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
        lat = TriclinicLattice.from_matrix(vecs)
        a, b, c = vecs
        V = lat.volume
        expected = jnp.array(
            [
                V / jnp.linalg.norm(jnp.cross(b, c)),
                V / jnp.linalg.norm(jnp.cross(a, c)),
                V / jnp.linalg.norm(jnp.cross(a, b)),
            ]
        )
        npt.assert_allclose(lat.perpendicular_lengths, expected, rtol=1e-10)

        lat = TriclinicLattice.from_matrix(jnp.eye(3))
        assert lat.perpendicular_lengths.shape == (3,)

        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, 0.3, 1.0]])
        lat = TriclinicLattice.from_matrix(vecs)
        assert jnp.all(lat.perpendicular_lengths > 0)

        vecs = jnp.stack(
            [
                jnp.diag(jnp.array([2.0, 3.0, 4.0])),
                jnp.eye(3) * 5.0,
            ]
        )
        lat = TriclinicLattice.from_matrix(vecs)
        lengths = lat.perpendicular_lengths
        assert lengths.shape == (2, 3)
        npt.assert_allclose(lengths[0], [2.0, 3.0, 4.0], rtol=1e-10)
        npt.assert_allclose(lengths[1], [5.0, 5.0, 5.0], rtol=1e-10)

    def test_min_multiplicity(self):
        cell = PeriodicCell(TriclinicLattice.from_matrix(jnp.eye(3) * 10.0))
        npt.assert_array_equal(min_multiplicity(cell, 4.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 5.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 8.0), [2, 2, 2])

        cell = PeriodicCell(
            TriclinicLattice.from_matrix(
                jnp.array([[8.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 6.0]]),
            )
        )
        npt.assert_array_equal(min_multiplicity(cell, 2.5), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 5.0), [2, 1, 2])

        vecs = jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 20.0])
        cell = PeriodicCell(TriclinicLattice.from_matrix(vecs))
        result = min_multiplicity(cell, 8.0)
        assert result.shape == (2, 3)
        npt.assert_array_equal(result[0], [2, 2, 2])
        npt.assert_array_equal(result[1], [1, 1, 1])


class TestOrthogonalLattice:
    def test_lattice_vectors_diagonal(self):
        lat = OrthogonalLattice(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(lat.lattice_vectors, jnp.diag(jnp.array([2.0, 3.0, 4.0])))

    def test_inverse_lattice_vectors(self):
        lat = OrthogonalLattice(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(
            lat.lattice_vectors @ lat.inverse_lattice_vectors,
            jnp.eye(3),
            atol=1e-10,
        )

    def test_volume(self):
        lat = OrthogonalLattice(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(lat.volume, 24.0)

    def test_perpendicular_lengths_equal_lengths(self):
        lengths = jnp.array([2.0, 3.0, 4.0])
        lat = OrthogonalLattice(lengths)
        npt.assert_allclose(lat.perpendicular_lengths, lengths)

    def test_wrap_real_to_real(self):
        cell = PeriodicCell(OrthogonalLattice(jnp.array([10.0, 10.0, 10.0])))
        r = jnp.array([12.0, -3.0, 25.0])
        wrapped = cell.wrap(r)
        npt.assert_allclose(wrapped, jnp.array([2.0, -3.0, -5.0]), atol=1e-10)

    def test_wrap_matches_triclinic(self):
        """Orthogonal wrap must agree with equivalent TriclinicLattice."""
        lengths = jnp.array([2.0, 3.0, 4.0])
        ortho = PeriodicCell(OrthogonalLattice(lengths))
        tri = PeriodicCell(TriclinicLattice.from_matrix(jnp.diag(lengths)))
        r = jnp.array([3.5, -1.2, 7.8])
        npt.assert_allclose(ortho.wrap(r), tri.wrap(r), atol=1e-10)

    def test_wrap_cross_space(self):
        cell = PeriodicCell(OrthogonalLattice(jnp.array([4.0, 4.0, 4.0])))
        r = jnp.array([3.0, -3.0, 5.0])
        frac = cell.wrap(
            r, input_space=CoordinateSpace.REAL, output_space=CoordinateSpace.FRACTIONAL
        )
        npt.assert_allclose(frac, jnp.array([-0.25, 0.25, 0.25]), atol=1e-10)

    def test_batched(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        lat = OrthogonalLattice(lengths)
        assert lat.volume.shape == (2,)
        npt.assert_allclose(lat.volume, jnp.array([24.0, 210.0]))

    def test_slicing(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        lat = OrthogonalLattice(lengths)
        sub = lat[0]
        npt.assert_allclose(sub.lengths, jnp.array([2.0, 3.0, 4.0]))

    def test_cell_satisfies_protocol(self):
        cell = PeriodicCell(OrthogonalLattice(jnp.array([1.0, 1.0, 1.0])))
        assert isinstance(cell, Cell)

    def test_is_jax_pytree(self):
        lat = OrthogonalLattice(jnp.array([2.0, 3.0, 4.0]))
        leaves = jax.tree.leaves(lat)
        assert len(leaves) == 1
        scaled = jax.tree.map(lambda x: x * 2, lat)
        npt.assert_allclose(scaled.lengths, jnp.array([4.0, 6.0, 8.0]))


class TestProtocolSatisfaction:
    def test_periodic_triclinic_satisfies_cell(self):
        cell = PeriodicCell(TriclinicLattice.from_matrix(jnp.eye(3)))
        assert isinstance(cell, Cell)

    def test_periodic_orthogonal_satisfies_cell(self):
        cell = PeriodicCell(OrthogonalLattice(jnp.array([1.0, 1.0, 1.0])))
        assert isinstance(cell, Cell)

    def test_vacuum_satisfies_cell(self):
        cell = VacuumCell(OrthogonalLattice(jnp.array([1.0, 1.0, 1.0])))
        assert isinstance(cell, Cell)

    def test_slab_satisfies_cell(self):
        cell = SlabCell(
            OrthogonalLattice(jnp.array([1.0, 1.0, 1.0])),
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
        c = PeriodicCell(OrthogonalLattice(jnp.array([10.0, 10.0, 10.0])))
        assert c.periodic == (True, True, True)
        npt.assert_allclose(c.lattice.lengths, jnp.array([10.0, 10.0, 10.0]))

    def test_vacuum_orthogonal(self):
        c = VacuumCell(OrthogonalLattice(jnp.array([10.0, 10.0, 10.0])))
        assert c.periodic == (False, False, False)
        npt.assert_allclose(c.lattice.lengths, jnp.array([10.0, 10.0, 10.0]))

    def test_vacuum_triclinic(self):
        c = VacuumCell(TriclinicLattice.from_matrix(jnp.eye(3) * 2.0))
        assert c.periodic == (False, False, False)
        npt.assert_allclose(c.lattice_vectors, jnp.eye(3) * 2.0)

    def test_slab_orthogonal(self):
        c = SlabCell(
            OrthogonalLattice(jnp.array([10.0, 10.0, 30.0])),
            periodic=(True, True, False),
        )
        assert c.periodic == (True, True, False)

    def test_slab_triclinic(self):
        c = SlabCell(
            TriclinicLattice.from_matrix(jnp.eye(3) * 5.0),
            periodic=(True, True, False),
        )
        assert c.periodic == (True, True, False)


class TestTypeGuards:
    def test_is_vacuum_positive(self):
        c = VacuumCell(OrthogonalLattice(jnp.array([1.0, 1.0, 1.0])))
        assert is_vacuum(c)
        assert not is_fully_periodic(c)

    def test_is_fully_periodic_positive(self):
        c = PeriodicCell(OrthogonalLattice(jnp.array([1.0, 1.0, 1.0])))
        assert is_fully_periodic(c)
        assert not is_vacuum(c)

    def test_slab_is_neither(self):
        c = SlabCell(
            OrthogonalLattice(jnp.array([10.0, 10.0, 30.0])),
            periodic=(True, True, False),
        )
        assert not is_vacuum(c)
        assert not is_fully_periodic(c)


class TestPeriodicPreservedUnderScaling:
    """Scaling a cell must preserve its boundary condition.

    Previously, ``__mul__`` and ``make_supercell``'s rebuild path did not
    forward ``periodic``, silently turning a vacuum or slab cell back into
    a fully-periodic one.
    """

    def test_orthogonal_mul_preserves_vacuum(self):
        v = VacuumCell(OrthogonalLattice(jnp.array([10.0, 10.0, 10.0])))
        scaled = v * 2.0
        assert scaled.periodic == (False, False, False)
        npt.assert_allclose(scaled.lattice.lengths, jnp.array([20.0, 20.0, 20.0]))

    def test_triclinic_mul_preserves_slab(self):
        c = SlabCell(
            TriclinicLattice.from_matrix(jnp.eye(3) * 5.0),
            periodic=(True, True, False),
        )
        scaled = c * 2.0
        assert scaled.periodic == (True, True, False)
