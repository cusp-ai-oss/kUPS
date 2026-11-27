# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.edges`` (the ``Edges`` dataclass)."""

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core import units
from kups.core.cell import MatrixLogFrame, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.neighborlist.edges import Edges
from kups.core.typing import ParticleId
from kups.core.units import testing
from kups.core.utils.math import (
    GeneralSquareMatrix,
    LowerTriangularSquareMatrix,
    triangular_3x3_matmul,
)
from test.units import A

from ._builders import make_lh, make_systems


@contextmanager
def _x64(enabled: bool) -> Iterator[None]:
    """Scoped x64 toggle; `jax.enable_x64` only exists from jax 0.8."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


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
        npt.assert_array_equal(np.asarray(edges.indices), np.asarray(indices))
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
        npt.assert_array_equal(np.asarray(edges.indices), np.asarray(indices))
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
        positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) * A
        lh = make_lh(positions, jnp.zeros(2, dtype=int))
        cell = PeriodicCell(TriclinicFrame.from_matrix((jnp.eye(3)[None] * 10.0) * A))
        systems, _ = make_systems(cell, jnp.array([1.0]) * A)
        edges = Edges(
            Index((ParticleId(0), ParticleId(1)), jnp.array([[0, 1]])),
            jnp.array([[shift]], dtype=float),
        )
        return edges, lh, systems

    def test_difference_vectors_no_shift(self):
        edges, lh, systems = self._setup([0.0, 0.0, 0.0])
        diff = edges.difference_vectors(lh, systems)
        npt.assert_allclose(diff.to(A), jnp.array([[[2.0, 0.0, 0.0]]]))

    def test_difference_vectors_with_periodic_shift(self):
        # shift (1,0,0) adds one lattice vector (10 Å) to the difference.
        edges, lh, systems = self._setup([1.0, 0.0, 0.0])
        diff = edges.difference_vectors(lh, systems)
        npt.assert_allclose(diff.to(A), jnp.array([[[12.0, 0.0, 0.0]]]))

    def test_absolute_shifts_apply_lattice(self):
        edges, lh, systems = self._setup([1.0, 0.0, 0.0])
        abs_shifts = edges.absolute_shifts(lh, systems)
        npt.assert_allclose(abs_shifts.to(A), jnp.array([[[10.0, 0.0, 0.0]]]))


class TestAbsoluteShiftsStructure:
    """``absolute_shifts`` routes through the cell frame's static structure."""

    def _edges(self, shift: jnp.ndarray) -> Edges:
        # Two degree-2 edges so the per-edge product is exercised on a non-trivial
        # ``(n_edges, 1, 3)`` shift array.
        return Edges(
            Index((ParticleId(0), ParticleId(1)), jnp.array([[0, 1], [0, 1]])),
            shift[:, None, :].astype(float),  # (n_edges, 1, 3)
        )

    def _lh(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) * A
        return make_lh(positions, jnp.zeros(2, dtype=int))

    def test_lower_triangular_matches_old_triangular_matmul(self):
        """Lower-triangular fast path is behaviour-preserving."""
        lattice = jnp.array([[10.0, 0.0, 0.0], [1.0, 11.0, 0.0], [2.0, 3.0, 12.0]])
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice[None] * A))
        systems, _ = make_systems(cell, jnp.array([1.0]) * A)
        assert isinstance(cell.frame.matrix, LowerTriangularSquareMatrix)
        shift = jnp.array([[1.0, -2.0, 1.0], [0.0, 1.0, 0.0]])
        edges = self._edges(shift)
        new = edges.absolute_shifts(self._lh(), systems)
        vecs = cell.frame.vectors[0]
        old = triangular_3x3_matmul(vecs[None, None], edges.shifts)
        testing.assert_allclose(new, old, A, atol=1e-6)

    def test_general_rotated_cell_matches_dense_reference(self):
        """General (rotated, non-lower-tri) cell uses the full 3x3 product."""
        # A general 3x3 lattice with a non-zero upper triangle.
        lattice = jnp.array([[10.0, 1.5, 0.7], [0.4, 11.0, 2.1], [1.2, 0.9, 12.0]])
        cell = PeriodicCell(MatrixLogFrame.from_matrix(lattice[None] * A))
        systems, _ = make_systems(cell, jnp.array([1.0]) * A)
        assert type(cell.frame.matrix) is GeneralSquareMatrix
        shift = jnp.array([[1.0, -2.0, 1.0], [0.0, 1.0, 0.0]])
        edges = self._edges(shift)
        new = edges.absolute_shifts(self._lh(), systems)
        vecs = cell.frame.vectors[0]
        # Dense RIGHT-side reference: shifts @ vecs per edge/Degree slot.
        ref = units.einsum("ji,nkj->nki", vecs, edges.shifts)
        testing.assert_allclose(new, ref, A, atol=1e-5)


class TestDifferenceVectorGradients:
    """The hand-written reverse pass of ``difference_vectors`` equals autodiff."""

    @staticmethod
    def _case(degree: int, n_sys: int, dtype: type):
        rng = np.random.default_rng(degree + 10 * n_sys)
        n, n_edges = 12, 40
        system = jnp.arange(n) * n_sys // n
        positions = jnp.asarray(rng.uniform(0, 6, (n, 3)), dtype)
        vectors = np.tril(6 * np.eye(3) + rng.normal(0, 0.3, (n_sys, 3, 3)))
        vectors = jnp.asarray(vectors, dtype)
        # Edges stay inside one system; the last rows are padding.
        first = rng.integers(0, n, n_edges)
        others = rng.integers(0, n // n_sys, (n_edges, degree - 1))
        others += (first * n_sys // n * (n // n_sys))[:, None]
        raw = np.column_stack([first, others])
        raw[-5:] = n
        shifts = rng.integers(-1, 2, (n_edges, degree - 1, 3))
        shifts[-5:] = 0
        weights = jnp.asarray(rng.normal(size=(n_edges, degree - 1, 3)), dtype)
        return (
            positions,
            vectors,
            jnp.asarray(raw),
            jnp.asarray(shifts),
            system,
            weights,
        )

    @staticmethod
    def _through_edges(positions, vectors, raw, shifts, system, weights):
        particles = make_lh(positions * A, system)
        cell = PeriodicCell(TriclinicFrame.from_matrix(vectors * A))
        systems, _ = make_systems(cell, jnp.ones(len(vectors)))
        edges = Edges(Index(particles.keys, raw), shifts)
        vectors_out = edges.difference_vectors(particles, systems).to(A)
        return (weights * vectors_out**2).sum()

    @staticmethod
    def _plain(positions, vectors, raw, shifts, system, weights):
        valid = (raw < len(positions))[..., None]
        rows = jnp.where(valid, positions[jnp.minimum(raw, len(positions) - 1)], 0)
        lattice = jnp.tril(vectors)[system[raw[:, 0] % len(system)]]
        lattice = jnp.where(valid[:, :1], lattice, 0)
        in_cell = jnp.einsum("edk,ekl->edl", shifts.astype(positions.dtype), lattice)
        return (weights * (rows[:, 1:] - rows[:, :1] + in_cell) ** 2).sum()

    @pytest.mark.parametrize("degree", [2, 3, 4])
    @pytest.mark.parametrize("n_sys", [1, 3])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_gradients_match_autodiff(self, degree, n_sys, dtype):
        with _x64(dtype == jnp.float64):
            case = self._case(degree, n_sys, dtype)
            value, grads = jax.value_and_grad(self._through_edges, (0, 1))(*case)
            expected, expected_grads = jax.value_and_grad(self._plain, (0, 1))(*case)
            rtol = 50 * float(jnp.finfo(dtype).eps)
            npt.assert_allclose(value, expected, rtol=rtol)
            for got, want in zip(grads, expected_grads, strict=True):
                assert got.dtype == dtype
                npt.assert_allclose(got, want, rtol=rtol, atol=rtol * 10)

    @pytest.mark.parametrize("n_sys", [1, 3])
    def test_edges_outside_the_systems_pass_no_cell_gradient(self, n_sys):
        """Padding and buffer particles keep their cotangent, even NaN, off the cell."""
        with _x64(True):
            positions, vectors, raw, shifts, system, weights = self._case(
                2, n_sys, jnp.float64
            )
            n = len(positions)
            # The last edges become padding and edges of a buffer particle.
            raw = raw.at[-5:-2].set(n).at[-2:, 0].set(n - 1)
            shifts = shifts.at[-5:].set(1)

            def through_edges(vectors, raw, shifts, weights):
                particles = make_lh(positions * A, system)
                buffered = system.at[-1].set(n_sys)
                data = dataclasses.replace(
                    particles.data, system=Index(particles.data.system.keys, buffered)
                )
                cell = PeriodicCell(TriclinicFrame.from_matrix(vectors * A))
                systems, _ = make_systems(cell, jnp.ones(n_sys))
                edges = Edges(Index(particles.keys, raw), shifts)
                out = edges.absolute_shifts(particles.set_data(data), systems).to(A)
                return jnp.where(jnp.isnan(weights), jnp.nan, weights * out**2).sum()

            gradient = jax.grad(through_edges)
            expected = gradient(vectors, raw[:-5], shifts[:-5], weights[:-5])
            got = gradient(vectors, raw, shifts, weights.at[-5:].set(jnp.nan))
            assert float(jnp.abs(expected).max()) > 0
            npt.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_second_derivatives_match_autodiff(self):
        with _x64(True):
            case = self._case(2, 3, jnp.float64)
            tangent = jnp.ones_like(case[0])

            def hvp(fn):
                gradient = jax.grad(fn)
                return jax.jvp(lambda x: gradient(x, *case[1:]), (case[0],), (tangent,))

            for got, want in zip(hvp(self._through_edges), hvp(self._plain)):
                npt.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
