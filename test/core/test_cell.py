# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Cell types: TriclinicCell, OrthogonalCell."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.cell import (
    Cell,
    CoordinateSpace,
    OrthogonalCell,
    TriclinicCell,
    is_fully_periodic,
    is_vacuum,
    make_fully_periodic,
    make_slab,
    make_vacuum,
    min_multiplicity,
    to_lower_triangular,
)


class TestTriclinicCell:
    def test_from_lattice_vectors(self):
        """Merged: cubic + orthorhombic + triclinic."""
        # cubic
        vecs = jnp.eye(3)
        cell = TriclinicCell.from_matrix(vecs)
        npt.assert_allclose(cell.lattice_vectors, vecs)
        npt.assert_allclose(cell.inverse_lattice_vectors, jnp.eye(3))
        npt.assert_allclose(cell.volume, 1.0)

        # orthorhombic
        vecs = jnp.diag(jnp.array([2.0, 3.0, 4.0]))
        cell = TriclinicCell.from_matrix(vecs)
        npt.assert_allclose(cell.lattice_vectors, vecs)
        npt.assert_allclose(
            cell.inverse_lattice_vectors,
            jnp.diag(jnp.array([0.5, 1.0 / 3.0, 0.25])),
        )
        npt.assert_allclose(cell.volume, 24.0)

        # triclinic
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]])
        cell = TriclinicCell.from_matrix(vecs)
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
        cell = TriclinicCell.from_matrix(L)
        assert cell.volume > 0
        npt.assert_allclose(cell.volume, 1.0)

    def test_wrap_real_to_real(self):
        """Merged: single_point + multiple_points + edge_cases."""
        # single point
        cell = TriclinicCell.from_matrix(jnp.eye(3))
        r = jnp.array([1.5, -0.7, 2.3])
        npt.assert_allclose(cell.wrap(r), jnp.array([-0.5, 0.3, 0.3]), atol=1e-10)

        # multiple points
        cell = TriclinicCell.from_matrix(jnp.eye(3) * 2.0)
        r = jnp.array([[3.0, 1.0, -1.0], [-1.0, 4.0, 0.5], [0.0, 0.0, 0.0]])
        expected = jnp.array([[-1.0, -1.0, -1.0], [-1.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
        npt.assert_allclose(cell.wrap(r), expected, atol=1e-10)

        # edge cases (boundary)
        cell = TriclinicCell.from_matrix(jnp.eye(3))
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
        cell = TriclinicCell.from_matrix(jnp.eye(3) * 2.0)

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
        cell1 = TriclinicCell.from_matrix(jnp.eye(3))
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
        cell = TriclinicCell.from_matrix(vecs)
        r = jnp.array([1.5, 1.0, 0.5])
        wrapped = cell.wrap(r)
        assert not jnp.allclose(wrapped, r)
        npt.assert_allclose(wrapped, cell.wrap(wrapped), atol=1e-10)

        # shape preservation
        cell = TriclinicCell.from_matrix(jnp.eye(3))
        for shape in [(3,), (5, 3), (2, 4, 3), (10, 1, 3)]:
            r = jnp.ones(shape) * 1.5
            assert cell.wrap(r).shape == shape

    def test_wrap_jit_and_gradient(self):
        """Merged: JIT compilation + gradient."""
        cell = TriclinicCell.from_matrix(jnp.eye(3))

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
            cell = TriclinicCell.from_matrix(vecs)
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
        cell = TriclinicCell.from_matrix(jnp.eye(3) * 5.0)
        npt.assert_allclose(
            cell.perpendicular_lengths, jnp.array([5.0, 5.0, 5.0]), rtol=1e-10
        )

        # orthorhombic
        cell = TriclinicCell.from_matrix(jnp.diag(jnp.array([2.0, 3.0, 4.0])))
        npt.assert_allclose(
            cell.perpendicular_lengths, jnp.array([2.0, 3.0, 4.0]), rtol=1e-10
        )

        # triclinic
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cell = TriclinicCell.from_matrix(vecs)
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
        cell = TriclinicCell.from_matrix(jnp.eye(3))
        assert cell.perpendicular_lengths.shape == (3,)

        # positive
        vecs = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, 0.3, 1.0]])
        cell = TriclinicCell.from_matrix(vecs)
        assert jnp.all(cell.perpendicular_lengths > 0)

        # batched
        vecs = jnp.stack(
            [
                jnp.diag(jnp.array([2.0, 3.0, 4.0])),
                jnp.eye(3) * 5.0,
            ]
        )
        cell = TriclinicCell.from_matrix(vecs)
        lengths = cell.perpendicular_lengths
        assert lengths.shape == (2, 3)
        npt.assert_allclose(lengths[0], [2.0, 3.0, 4.0], rtol=1e-10)
        npt.assert_allclose(lengths[1], [5.0, 5.0, 5.0], rtol=1e-10)

    def test_min_multiplicity(self):
        """Merged: cubic + orthorhombic + batched."""
        # cubic
        cell = TriclinicCell.from_matrix(jnp.eye(3) * 10.0)
        npt.assert_array_equal(min_multiplicity(cell, 4.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 5.0), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 8.0), [2, 2, 2])

        # orthorhombic
        cell = TriclinicCell.from_matrix(
            jnp.array([[8.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 6.0]]),
        )
        npt.assert_array_equal(min_multiplicity(cell, 2.5), [1, 1, 1])
        npt.assert_array_equal(min_multiplicity(cell, 5.0), [2, 1, 2])

        # batched
        vecs = jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 20.0])
        cell = TriclinicCell.from_matrix(vecs)
        result = min_multiplicity(cell, 8.0)
        assert result.shape == (2, 3)
        npt.assert_array_equal(result[0], [2, 2, 2])
        npt.assert_array_equal(result[1], [1, 1, 1])


class TestOrthorhombicCell:
    def test_lattice_vectors_diagonal(self):
        cell = OrthogonalCell(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(cell.lattice_vectors, jnp.diag(jnp.array([2.0, 3.0, 4.0])))

    def test_inverse_lattice_vectors(self):
        cell = OrthogonalCell(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(
            cell.lattice_vectors @ cell.inverse_lattice_vectors,
            jnp.eye(3),
            atol=1e-10,
        )

    def test_volume(self):
        cell = OrthogonalCell(jnp.array([2.0, 3.0, 4.0]))
        npt.assert_allclose(cell.volume, 24.0)

    def test_perpendicular_lengths_equal_lengths(self):
        lengths = jnp.array([2.0, 3.0, 4.0])
        cell = OrthogonalCell(lengths)
        npt.assert_allclose(cell.perpendicular_lengths, lengths)

    def test_wrap_real_to_real(self):
        cell = OrthogonalCell(jnp.array([10.0, 10.0, 10.0]))
        r = jnp.array([12.0, -3.0, 25.0])
        wrapped = cell.wrap(r)
        npt.assert_allclose(wrapped, jnp.array([2.0, -3.0, -5.0]), atol=1e-10)

    def test_wrap_matches_triclinic(self):
        """Orthorhombic wrap must agree with equivalent TriclinicCell."""
        lengths = jnp.array([2.0, 3.0, 4.0])
        ortho = OrthogonalCell(lengths)
        tri = TriclinicCell.from_matrix(jnp.diag(lengths))
        r = jnp.array([3.5, -1.2, 7.8])
        npt.assert_allclose(ortho.wrap(r), tri.wrap(r), atol=1e-10)

    def test_wrap_cross_space(self):
        cell = OrthogonalCell(jnp.array([4.0, 4.0, 4.0]))
        r = jnp.array([3.0, -3.0, 5.0])
        frac = cell.wrap(
            r, input_space=CoordinateSpace.REAL, output_space=CoordinateSpace.FRACTIONAL
        )
        npt.assert_allclose(frac, jnp.array([-0.25, 0.25, 0.25]), atol=1e-10)

    def test_batched(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        cell = OrthogonalCell(lengths)
        assert cell.volume.shape == (2,)
        npt.assert_allclose(cell.volume, jnp.array([24.0, 210.0]))

    def test_slicing(self):
        lengths = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        cell = OrthogonalCell(lengths)
        sub = cell[0]
        npt.assert_allclose(sub.lengths, jnp.array([2.0, 3.0, 4.0]))

    def test_satisfies_cell_protocol(self):
        cell = OrthogonalCell(jnp.array([1.0, 1.0, 1.0]))
        assert isinstance(cell, Cell)

    def test_is_jax_pytree(self):
        cell = OrthogonalCell(jnp.array([2.0, 3.0, 4.0]))
        leaves = jax.tree.leaves(cell)
        assert len(leaves) == 1
        scaled = jax.tree.map(lambda x: x * 2, cell)
        npt.assert_allclose(scaled.lengths, jnp.array([4.0, 6.0, 8.0]))


class TestProtocolSatisfaction:
    def test_triclinic_satisfies_cell(self):
        cell = TriclinicCell.from_matrix(jnp.eye(3))
        assert isinstance(cell, Cell)

    def test_orthorhombic_satisfies_cell(self):
        cell = OrthogonalCell(jnp.array([1.0, 1.0, 1.0]))
        assert isinstance(cell, Cell)


class TestBoundaryModeFactories:
    """Runtime behavior of make_vacuum / make_fully_periodic / make_slab.

    Type-level discrimination (Cell[Vacuum] vs Cell[FullyPeriodic]) is
    enforced by pyright at static-analysis time and tested via the typing
    probes in development; here we just lock in the runtime semantics.
    """

    def test_make_vacuum_orthogonal(self):
        c = make_vacuum(OrthogonalCell(jnp.array([10.0, 10.0, 10.0])))
        assert c.periodic == (False, False, False)
        npt.assert_allclose(c.lengths, jnp.array([10.0, 10.0, 10.0]))

    def test_make_vacuum_triclinic(self):
        c = make_vacuum(TriclinicCell.from_matrix(jnp.eye(3) * 2.0))
        assert c.periodic == (False, False, False)
        npt.assert_allclose(c.lattice_vectors, jnp.eye(3) * 2.0)

    def test_make_fully_periodic_orthogonal(self):
        # Start from a vacuum cell, brand it back to fully periodic
        v = make_vacuum(OrthogonalCell(jnp.array([5.0, 5.0, 5.0])))
        p = make_fully_periodic(v)
        assert p.periodic == (True, True, True)

    def test_make_slab_orthogonal(self):
        c = make_slab(
            OrthogonalCell(jnp.array([10.0, 10.0, 30.0])),
            periodic=(True, True, False),
        )
        assert c.periodic == (True, True, False)

    def test_make_slab_triclinic(self):
        c = make_slab(
            TriclinicCell.from_matrix(jnp.eye(3) * 5.0),
            periodic=(True, True, False),
        )
        assert c.periodic == (True, True, False)


class TestBoundaryModeTypeGuards:
    def test_is_vacuum_positive(self):
        c = make_vacuum(OrthogonalCell(jnp.array([1.0, 1.0, 1.0])))
        assert is_vacuum(c)
        assert not is_fully_periodic(c)

    def test_is_fully_periodic_positive(self):
        c = OrthogonalCell(jnp.array([1.0, 1.0, 1.0]))  # default = (T, T, T)
        assert is_fully_periodic(c)
        assert not is_vacuum(c)

    def test_slab_is_neither(self):
        c = make_slab(
            OrthogonalCell(jnp.array([10.0, 10.0, 30.0])),
            periodic=(True, True, False),
        )
        assert not is_vacuum(c)
        assert not is_fully_periodic(c)


class TestPeriodicPreservedUnderScaling:
    """Regression test: scaling a cell must preserve its boundary mode.

    Previously, ``__mul__`` and ``make_supercell``'s ``make_scaled`` re-built
    the cell without forwarding ``periodic``, silently turning a vacuum or
    slab cell back into a fully-periodic one.
    """

    def test_orthogonal_mul_preserves_vacuum(self):
        v = make_vacuum(OrthogonalCell(jnp.array([10.0, 10.0, 10.0])))
        scaled = v * 2.0
        assert scaled.periodic == (False, False, False)
        npt.assert_allclose(scaled.lengths, jnp.array([20.0, 20.0, 20.0]))

    def test_triclinic_mul_preserves_slab(self):
        c = make_slab(
            TriclinicCell.from_matrix(jnp.eye(3) * 5.0),
            periodic=(True, True, False),
        )
        scaled = c * 2.0
        assert scaled.periodic == (True, True, False)
