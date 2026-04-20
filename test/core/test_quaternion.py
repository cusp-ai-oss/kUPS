# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for Quaternion functionality."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.utils.quaternion import Quaternion


class TestQuaternionBasics:
    def test_creation_identity_random(self):
        """Merged: creation, invalid_shape, identity, random_shape + normalization."""
        # creation
        components = jnp.array([1.0, 0.0, 0.0, 0.0])
        q = Quaternion(components)
        npt.assert_array_equal(q.components, components)

        # invalid shape
        with pytest.raises(ValueError, match="must have shape"):
            Quaternion(jnp.array([1.0, 0.0, 0.0]))

        # identity
        q = Quaternion.identity()
        npt.assert_array_equal(q.components, jnp.array([1.0, 0.0, 0.0, 0.0]))

        # random shape + normalization
        key = jax.random.key(42)
        q = Quaternion.random(key)
        assert q.components.shape == (4,)
        q_batch = Quaternion.random(key, shape=(3, 2))
        assert q_batch.components.shape == (3, 2, 4)
        q10 = Quaternion.random(jax.random.key(123), shape=(10,))
        npt.assert_array_almost_equal(
            jnp.linalg.norm(q10.components, axis=-1),
            jnp.ones(10),
            decimal=6,
        )


class TestQuaternionOperations:
    def test_multiply(self):
        """Merged: inverse, identity_mult, inverse_mult, associativity."""
        # inverse
        q = Quaternion(jnp.array([0.5, 0.5, 0.5, 0.5]))
        npt.assert_array_equal(q.inv().components, jnp.array([0.5, -0.5, -0.5, -0.5]))

        # identity mult
        identity = Quaternion.identity()
        result1 = q * identity
        result2 = identity * q
        npt.assert_array_almost_equal(result1.components, q.components)
        npt.assert_array_almost_equal(result2.components, q.components)

        # q * q^-1 = identity
        q2 = Quaternion(jnp.array([0.6, 0.8, 0.0, 0.0]))
        result = q2 * q2.inv()
        npt.assert_array_almost_equal(
            result.components,
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            decimal=6,
        )

        # associativity
        q1 = Quaternion(jnp.array([0.5, 0.5, 0.5, 0.5]))
        q2 = Quaternion(jnp.array([0.6, 0.8, 0.0, 0.0]))
        q3 = Quaternion(jnp.array([0.0, 0.6, 0.8, 0.0]))
        npt.assert_array_almost_equal(
            ((q1 * q2) * q3).components,
            (q1 * (q2 * q3)).components,
            decimal=6,
        )

    def test_power(self):
        """Merged: power_identity + power_square."""
        q = Quaternion(jnp.array([0.6, 0.8, 0.0, 0.0]))

        # q^1 = q
        npt.assert_array_almost_equal((q**1.0).components, q.components, decimal=6)

        # q^0 = identity
        q0 = q**0.0
        assert abs(q0.components[0] - 1.0) < 1e-6
        npt.assert_array_almost_equal(q0.components[1:], jnp.zeros(3), decimal=6)

        # (q^2)^0.5 ~ q
        q2 = Quaternion(jnp.array([0.8, 0.6, 0.0, 0.0]))
        q_sqrt = (q2**2.0) ** 0.5
        diff_pos = jnp.linalg.norm(q_sqrt.components - q2.components)
        diff_neg = jnp.linalg.norm(q_sqrt.components + q2.components)
        assert min(diff_pos, diff_neg) < 1e-5

    def test_invalid_types(self):
        """Invalid multiplication and power types."""
        q = Quaternion.identity()
        with pytest.raises(TypeError, match="Unsupported type for multiplication"):
            q * 5.0  # type: ignore
        with pytest.raises(TypeError, match="Unsupported type for multiplication"):
            q * jnp.array([1, 2, 3])  # type: ignore
        with pytest.raises(TypeError, match="Unsupported type for exponentiation"):
            q ** "invalid"  # type: ignore


class TestQuaternionMatrixConversion:
    def test_matrix_conversion(self):
        """Merged: identity_matrix, rotation_properties, vector_rotation_consistency."""
        # identity -> identity matrix
        npt.assert_array_almost_equal(
            Quaternion.identity().as_matrix(),
            jnp.eye(3),
            decimal=6,
        )

        # rotation matrix properties
        q = Quaternion.random(jax.random.key(42), shape=(5,))
        matrices = q.as_matrix()
        for i in range(5):
            R = matrices[i]
            npt.assert_array_almost_equal(R @ R.T, jnp.eye(3), decimal=6)
            npt.assert_almost_equal(jnp.linalg.det(R), 1.0, decimal=6)

        # vector rotation consistency (90 deg around z)
        angle = jnp.pi / 2
        axis = jnp.array([0.0, 0.0, 1.0])
        q_comp = jnp.concatenate([jnp.cos(angle / 2)[None], axis * jnp.sin(angle / 2)])
        q = Quaternion(q_comp)
        v = jnp.array([1.0, 0.0, 0.0])
        R = q.as_matrix()
        npt.assert_array_almost_equal(R @ v, v @ q, decimal=6)
        npt.assert_array_almost_equal(R @ v, jnp.array([0.0, 1.0, 0.0]), decimal=6)


class TestQuaternionVectorRotation:
    def test_vector_rotation(self):
        """Merged: rmatmul_shape, identity_rotation, batch_rotation."""
        q = Quaternion.identity()

        # valid 3D vector
        v = jnp.array([1.0, 2.0, 3.0])
        result = v @ q
        assert result.shape == (3,)
        npt.assert_array_equal(result, v)

        # invalid shapes
        with pytest.raises(ValueError, match="Expected last dimension.*to be 3"):
            jnp.array([1.0, 2.0]) @ q  # type: ignore
        with pytest.raises(
            TypeError, match="Unsupported type for right multiplication"
        ):
            5.0 @ q  # type: ignore

        # batch rotation
        vectors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        npt.assert_array_almost_equal(vectors @ q, vectors, decimal=6)

        # indexing
        components = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        q = Quaternion(components)
        npt.assert_array_equal(q[0].components, jnp.array([1.0, 0.0, 0.0, 0.0]))
        npt.assert_array_equal(q[1:].components, components[1:])


class TestQuaternionEdgeCases:
    def test_edge_cases(self):
        """Merged: near_zero_power, identity_power, normalization_preservation."""
        # near zero power
        small_angle = 1e-8
        q = Quaternion(
            jnp.array([jnp.cos(small_angle / 2), jnp.sin(small_angle / 2), 0.0, 0.0])
        )
        assert jnp.all(jnp.isfinite((q**2.0).components))

        # identity^p = identity
        q_id = Quaternion.identity()
        for p in [0.0, 0.5, 1.0, 2.0, 3.0]:
            result = q_id**p
            assert jnp.all(jnp.isfinite(result.components)), f"NaN at p={p}"
            npt.assert_array_almost_equal(
                result.components,
                jnp.array([1.0, 0.0, 0.0, 0.0]),
                decimal=6,
            )

        # normalization preservation
        key = jax.random.key(99)
        q1 = Quaternion.random(key)
        q2 = Quaternion.random(jax.random.split(key)[0])
        npt.assert_almost_equal(jnp.linalg.norm((q1 * q2).components), 1.0, decimal=6)
        npt.assert_almost_equal(jnp.linalg.norm((q1**2.0).components), 1.0, decimal=6)


class TestQuaternionJAXCompatibility:
    def test_jit_and_vmap(self):
        """Merged: JIT compilation + vmap compatibility."""

        # JIT
        @jax.jit
        def quaternion_ops(q1_components, q2_components):
            q1 = Quaternion(q1_components)
            q2 = Quaternion(q2_components)
            return (q1 * q2).components, (q1**2.0).components, q1.as_matrix()

        q1_comp = jnp.array([1.0, 0.0, 0.0, 0.0])
        q2_comp = jnp.array([0.0, 1.0, 0.0, 0.0])
        mult, power, matrix = quaternion_ops(q1_comp, q2_comp)
        assert mult.shape == (4,)
        assert power.shape == (4,)
        assert matrix.shape == (3, 3)

        # vmap
        @jax.vmap
        def batch_multiply(q1_batch, q2_batch):
            return (Quaternion(q1_batch) * Quaternion(q2_batch)).components

        batch_size = 5
        q1_batch = jnp.tile(q1_comp, (batch_size, 1))
        q2_batch = jnp.tile(q2_comp, (batch_size, 1))
        result = batch_multiply(q1_batch, q2_batch)
        assert result.shape == (batch_size, 4)
        for i in range(1, batch_size):
            npt.assert_array_equal(result[0], result[i])
