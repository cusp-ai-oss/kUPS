# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.edges`` (the ``Edges`` dataclass)."""

import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.neighborlist.edges import Edges
from kups.core.typing import ParticleId

from ._builders import make_lh, make_systems


class TestEdgesConstruction:
    def test_creation_binary(self):
        """Binary edges (degree=2)."""
        indices = Index(
            (ParticleId(0), ParticleId(1), ParticleId(2)),
            jnp.array([[0, 1], [1, 2], [2, 0]]),
        )
        shifts = jnp.array([[[0, 0, 0]], [[1, 0, 0]], [[-1, 0, 0]]])
        edges = Edges(indices, shifts)

        assert edges.degree == 2
        assert edges.indices.shape == (3, 2)
        assert edges.shifts.shape == (3, 1, 3)
        assert len(edges) == 3
        npt.assert_array_equal(edges.indices.indices, indices.indices)
        npt.assert_array_equal(edges.shifts, shifts)

    def test_creation_ternary(self):
        """Ternary edges (degree=3)."""
        indices = Index(
            (ParticleId(0), ParticleId(1), ParticleId(2), ParticleId(3)),
            jnp.array([[0, 1, 2], [1, 2, 3]]),
        )
        shifts = jnp.array([[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]])
        edges = Edges(indices, shifts)

        assert edges.degree == 3
        assert edges.indices.shape == (2, 3)
        assert edges.shifts.shape == (2, 2, 3)
        npt.assert_array_equal(edges.indices.indices, indices.indices)
        npt.assert_array_equal(edges.shifts, shifts)

    def test_degree_one_has_zero_inner_shift_dim(self):
        """Degree-1 edges carry a ``(n, 0, 3)`` shift array."""
        indices = Index((ParticleId(0), ParticleId(1)), jnp.array([[0], [1]]))
        shifts = jnp.zeros((2, 0, 3), dtype=int)
        edges = Edges(indices, shifts)
        assert edges.degree == 1
        assert edges.shifts.shape == (2, 0, 3)

    def test_shape_validation_rejects_mismatched_shifts(self):
        """Shift shape must match ``(n, degree-1, 3)``."""
        indices = jnp.array([[0, 1], [1, 2]])
        wrong_shifts = jnp.array([[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]])
        with pytest.raises(AssertionError):
            Edges(indices, wrong_shifts)  # type: ignore[arg-type]

    def test_rejects_non_integer_indices(self):
        indices = Index((ParticleId(0), ParticleId(1)), jnp.array([[0, 1]]))
        # Build with a float index array to trip the integer-dtype assertion.
        with pytest.raises(AssertionError):
            Edges(
                Index(indices.keys, jnp.array([[0.0, 1.0]])),  # type: ignore[arg-type]
                jnp.zeros((1, 1, 3)),
            )


class TestEdgeGeometry:
    """``difference_vectors`` / ``absolute_shifts`` against a known cell."""

    def _setup(self, shift):
        positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        lh = make_lh(positions, jnp.zeros(2, dtype=int))
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, _ = make_systems(cell, jnp.array([1.0]))
        edges = Edges(
            Index((ParticleId(0), ParticleId(1)), jnp.array([[0, 1]])),
            jnp.array([[shift]], dtype=float),
        )
        return edges, lh, systems

    def test_difference_vectors_no_shift(self):
        edges, lh, systems = self._setup([0.0, 0.0, 0.0])
        diff = edges.difference_vectors(lh, systems)
        npt.assert_allclose(diff, jnp.array([[[2.0, 0.0, 0.0]]]))

    def test_difference_vectors_with_periodic_shift(self):
        # shift (1,0,0) adds one lattice vector (10 Å) to the difference.
        edges, lh, systems = self._setup([1.0, 0.0, 0.0])
        diff = edges.difference_vectors(lh, systems)
        npt.assert_allclose(diff, jnp.array([[[12.0, 0.0, 0.0]]]))

    def test_absolute_shifts_apply_lattice(self):
        edges, lh, systems = self._setup([1.0, 0.0, 0.0])
        abs_shifts = edges.absolute_shifts(lh, systems)
        npt.assert_allclose(abs_shifts, jnp.array([[[10.0, 0.0, 0.0]]]))
