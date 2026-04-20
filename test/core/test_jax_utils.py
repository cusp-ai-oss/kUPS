# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.utils.jax import (
    NotJaxCompatibleError,
    dataclass,
    field,
    isin,
    kahan_summation,
    no_jax_tracing,
    no_post_init,
    non_differentiable,
    sequential_vmap_with_vjp,
    skip_post_init_if_disabled,
    tree_concat,
    tree_stack,
    tree_zeros_like,
)


@dataclass
class _Point:
    x: float
    y: float


class TestDataclass:
    """Tests for the custom jax dataclass decorator."""

    def test_dataclass_basic_properties(self):
        """Test frozen, equality, repr, hash, marker, kwargs_passed_through."""
        p1 = _Point(1.0, 2.0)
        p2 = _Point(1.0, 2.0)
        p3 = _Point(2.0, 1.0)

        # Creation
        assert p1.x == 1.0 and p1.y == 2.0

        # Frozen
        with pytest.raises(dataclasses.FrozenInstanceError):
            p1.x = 3.0  # type: ignore

        # Equality
        assert p1 == p2 and p1 != p3

        # Repr
        repr_str = repr(p1)
        assert "Point" in repr_str and "1.0" in repr_str

        # Hash
        assert len({p1, p2, p3}) == 2

        # Marker
        assert getattr(_Point, "_jax_dataclass") is True

    def test_dataclass_pytree_operations(self):
        """Test is_jax_pytree, complex_types, empty, idempotent."""
        # Basic pytree
        v = _Point(1.0, 2.0)
        leaves, treedef = jax.tree_util.tree_flatten(v)
        assert len(leaves) == 2 and 1.0 in leaves
        v_r = jax.tree_util.tree_unflatten(treedef, leaves)
        assert v_r.x == v.x and v_r.y == v.y

        # Complex types
        @dataclass
        class NestedData:
            values: list[float]
            metadata: dict[str, Any]

        data = NestedData([1.0, 2.0], {"source": "test"})
        leaves, treedef = jax.tree_util.tree_flatten(data)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert reconstructed.values == data.values
        assert reconstructed.metadata == data.metadata

        # Empty
        @dataclass
        class EmptyClass:
            pass

        leaves, treedef = jax.tree_util.tree_flatten(EmptyClass())
        assert len(leaves) == 0
        assert isinstance(jax.tree_util.tree_unflatten(treedef, leaves), EmptyClass)

        # Idempotent
        @dataclass
        class TestClass:
            value: int

        leaves, _ = jax.tree_util.tree_flatten(TestClass(42))
        assert 42 in leaves

    def test_dataclass_fields_and_defaults(self):
        """Test default values, custom field, optional fields."""

        # Defaults
        @dataclass
        class Settings:
            name: str
            debug: bool = False
            max_iter: int = 100

        s = Settings("test")
        assert s.debug is False and s.max_iter == 100
        s2 = Settings("t", debug=True)
        assert s2.debug is True

        # Custom field
        @dataclass
        class Model:
            weights: jax.Array
            bias: jax.Array = field(default_factory=lambda: jnp.zeros(10))
            training: bool = field(default=True, static=True)

        m = Model(jnp.ones((10, 5)))
        assert m.bias.shape == (10,) and m.training is True

        # Optional
        @dataclass
        class OptionalFields:
            required: int
            optional: int | None = None
            with_default: str = "default"

        obj = OptionalFields(42)
        assert obj.optional is None and obj.with_default == "default"
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        r = jax.tree_util.tree_unflatten(treedef, leaves)
        assert r.required == 42

    def test_dataclass_jax_transformations(self):
        """Test jit, grad, and complex JAX transforms (all through JIT)."""

        # JIT with arrays
        @dataclass
        class ArrayContainer:
            data: jax.Array
            scale: float

        container = ArrayContainer(jnp.array([1.0, 2.0, 3.0]), 2.0)
        result = jax.jit(lambda c: ArrayContainer(c.data * 2, c.scale))(container)
        assert jnp.allclose(result.data, jnp.array([2.0, 4.0, 6.0]))

        # Grad
        @dataclass
        class Params:
            w: jax.Array
            b: jax.Array

        def loss_fn(params, x, y):
            return jnp.mean((params.w * x + params.b - y) ** 2)

        params = Params(jnp.array(2.0), jnp.array(0.5))
        x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 4.0, 6.0])
        grads = jax.jit(jax.grad(loss_fn))(params, x, y)
        assert isinstance(grads, Params)
        assert grads.w.shape == params.w.shape

        # Complex: jit(grad) update loop
        @dataclass
        class ModelState:
            weights: jax.Array
            bias: jax.Array
            step: int = field(default=0, static=True)

        def update_fn(state, x, y):
            g = jax.grad(
                lambda s, x, y: jnp.mean((jnp.dot(x, s.weights) + s.bias - y) ** 2)
            )(state, x, y)
            return ModelState(
                state.weights - 0.01 * g.weights,
                state.bias - 0.01 * g.bias,
                state.step + 1,
            )

        state = ModelState(weights=jnp.array([1.0, 2.0]), bias=jnp.array(0.0))
        new_state = jax.jit(update_fn)(
            state, jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([1.0, 2.0])
        )
        assert new_state.step == 1
        assert not jnp.allclose(new_state.weights, state.weights)

    def test_dataclass_static_and_nested(self):
        """Test static fields and nested dataclasses."""

        # Static
        @dataclass
        class Config:
            learning_rate: float
            model_name: str = field(default="default", static=True)

        leaves, treedef = jax.tree_util.tree_flatten(Config(0.01))
        assert len(leaves) == 1 and 0.01 in leaves
        r = jax.tree_util.tree_unflatten(treedef, leaves)
        assert r.model_name == "default"

        # Nested
        @dataclass
        class Circle:
            center: _Point
            radius: float

        circle = Circle(_Point(1.0, 2.0), 5.0)
        leaves, treedef = jax.tree_util.tree_flatten(circle)
        assert len(leaves) == 3
        r = jax.tree_util.tree_unflatten(treedef, leaves)
        assert r.center.x == 1.0 and r.radius == 5.0

    def test_dataclass_inheritance(self):
        """Test that dataclass works with inheritance."""

        @dataclass
        class BaseModel:
            name: str

        @dataclass
        class LinearModel(BaseModel):
            weights: jax.Array
            bias: jax.Array

        model = LinearModel("linear", jnp.ones((5, 3)), jnp.zeros(3))
        assert model.name == "linear"
        leaves, _ = jax.tree_util.tree_flatten(model)
        assert len(leaves) == 3


class TestSequentialVmapWithVjp:
    """Tests for sequential_vmap_with_vjp function."""

    _seq_square = staticmethod(jax.jit(sequential_vmap_with_vjp(lambda x: x**2)))
    _seq_add_scaled = staticmethod(
        jax.jit(sequential_vmap_with_vjp(lambda x, y: x + 2 * y))
    )
    _seq_split = staticmethod(jax.jit(sequential_vmap_with_vjp(lambda x: (x**2, x**3))))

    def test_forward(self):
        """Test forward: basic, multi_args, pytree_output."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([0.5, 1.0, 1.5])

        # Basic
        assert jnp.allclose(self._seq_square(x), jax.vmap(lambda x: x**2)(x))

        # Multi args
        assert jnp.allclose(
            self._seq_add_scaled(x, y), jax.vmap(lambda x, y: x + 2 * y)(x, y)
        )

        # Pytree output
        sq, cube = self._seq_split(x)
        assert jnp.allclose(sq, x**2) and jnp.allclose(cube, x**3)

    def test_gradients(self):
        """Test gradients: single, multi, matches_vmap, pytree_output."""
        x = jnp.array([1.0, 2.0, 3.0])

        # Single arg grad
        grad = jax.jit(jax.grad(lambda x: jnp.sum(self._seq_square(x))))(x)
        assert jnp.allclose(grad, 2 * x)

        # Multi arg grad
        seq_product = jax.jit(sequential_vmap_with_vjp(lambda x, y: x * y))
        y = jnp.array([4.0, 5.0, 6.0])
        grad_x, grad_y = jax.jit(
            jax.grad(lambda x, y: jnp.sum(seq_product(x, y)), argnums=(0, 1))
        )(x, y)
        assert jnp.allclose(grad_x, y) and jnp.allclose(grad_y, x)

        # Matches vmap grad
        seq_cubic = sequential_vmap_with_vjp(lambda x: x**3)
        vmap_cubic = jax.vmap(lambda x: x**3)
        seq_g = jax.jit(jax.grad(lambda x: jnp.sum(seq_cubic(x))))(x)
        vmap_g = jax.jit(jax.grad(lambda x: jnp.sum(vmap_cubic(x))))(x)
        assert jnp.allclose(seq_g, vmap_g)

        # Pytree output grad
        grad = jax.jit(
            jax.grad(
                lambda x: (
                    jnp.sum(self._seq_split(x)[0]) + jnp.sum(self._seq_split(x)[1])
                )
            )
        )(x)
        assert jnp.allclose(grad, 2 * x + 3 * x**2)

    def test_outer_vmap(self):
        """Test with explicit outer vmap: forward + gradient."""
        seq_row_sum = jax.jit(sequential_vmap_with_vjp(lambda x: jnp.sum(x)))
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Forward
        result = jax.jit(jax.vmap(seq_row_sum))(x)
        assert jnp.allclose(result, jnp.array([3.0, 7.0, 11.0]))

        # Gradient
        seq_row_sum_sq = jax.jit(sequential_vmap_with_vjp(lambda x: jnp.sum(x) ** 2))
        x2 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        grad = jax.jit(jax.grad(lambda x: jnp.sum(jax.vmap(seq_row_sum_sq)(x))))(x2)
        row_sums = jnp.sum(x2, axis=1, keepdims=True)
        assert jnp.allclose(grad, 2 * jnp.broadcast_to(row_sums, x2.shape))


class TestIsin:
    """Tests for the isin function."""

    def test_isin(self):
        # Basic membership
        assert jnp.array_equal(
            isin(jnp.array([1, 3, 5, 7]), jnp.array([3, 5]), max_item=10),
            jnp.array([False, True, True, False]),
        )
        # Empty reference
        assert not jnp.any(
            isin(jnp.array([0, 1, 2]), jnp.array([], dtype=jnp.int32), max_item=5)
        )
        # All present
        assert jnp.all(isin(jnp.array([0, 1, 2]), jnp.array([0, 1, 2, 3]), max_item=5))


class TestKahanSummation:
    """Tests for kahan_summation."""

    def test_kahan_summation(self):
        # Basic sum
        result, comp = kahan_summation(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        npt.assert_allclose(result, jnp.array([4.0, 6.0]))

        # Chained with compensation
        total, comp = kahan_summation(jnp.array([1.0]), jnp.array([2.0]))
        total, comp = kahan_summation(total, jnp.array([3.0]), compensate=comp)
        npt.assert_allclose(total, jnp.array([6.0]))


class TestTreeOps:
    """Tests for tree_concat, tree_stack, tree_zeros_like."""

    def test_tree_ops(self):
        a, b = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
        npt.assert_array_equal(tree_concat(a, b), [1, 2, 3, 4])

        result = tree_stack(a, b)
        assert result.shape == (2, 2)
        npt.assert_array_equal(result[0], [1, 2])

        tree = {"a": jnp.ones(3), "b": jnp.ones((2, 2))}
        z = tree_zeros_like(tree)
        npt.assert_array_equal(z["a"], jnp.zeros(3))
        npt.assert_array_equal(z["b"], jnp.zeros((2, 2)))


class TestNonDifferentiable:
    """Tests for non_differentiable."""

    def test_identity_behavior(self):
        x = jnp.array([1.0, 2.0])
        npt.assert_array_equal(non_differentiable(x), x)

    def test_raises_on_grad(self):
        with pytest.raises(NotImplementedError, match="not differentiable"):
            jax.grad(lambda x: jnp.sum(non_differentiable(x)))(jnp.array(1.0))


class TestNoJaxTracing:
    """Tests for no_jax_tracing decorator."""

    def test_raises_inside_jax_transforms(self):
        """Test jit, vmap, grad, pytree_tracer, kwarg_tracer all raise."""

        @no_jax_tracing
        def protected(x):
            return x * 2

        # jit
        with pytest.raises(NotJaxCompatibleError):
            jax.jit(protected)(jnp.array([1, 2, 3]))

        # vmap
        with pytest.raises(NotJaxCompatibleError):
            jax.vmap(protected)(jnp.array([[1, 2], [3, 4]]))

        # grad
        @no_jax_tracing
        def protected_sum(x):
            return jnp.sum(x**2)

        with pytest.raises(NotJaxCompatibleError):
            jax.grad(protected_sum)(jnp.array([1.0, 2.0]))

        # pytree tracer
        @dataclass
        class Data:
            x: jax.Array
            y: jax.Array

        @no_jax_tracing
        def process(data: Data):
            return data.x + data.y

        with pytest.raises(NotJaxCompatibleError):
            jax.jit(lambda x, y: process(Data(x, y)))(
                jnp.array([1, 2]), jnp.array([3, 4])
            )

        # kwarg tracer
        @no_jax_tracing
        def kw_func(x, scale):
            return x * scale

        with pytest.raises(NotJaxCompatibleError):
            jax.jit(lambda x, s: kw_func(x, scale=s))(
                jnp.array([1, 2, 3]), jnp.array(2.0)
            )

    def test_direct_calls_work(self):
        """Test basic, pytree, kwargs, non_array direct calls."""

        @no_jax_tracing
        def multiply(x, scale=2):
            return x * scale

        # Basic
        assert jnp.allclose(multiply(jnp.array([1, 2, 3])), jnp.array([2, 4, 6]))

        # Kwargs
        assert jnp.allclose(
            multiply(jnp.array([1, 2, 3]), scale=3), jnp.array([3, 6, 9])
        )

        # Pytree
        @dataclass
        class Data:
            x: jax.Array
            y: jax.Array

        @no_jax_tracing
        def process(data: Data):
            return data.x + data.y

        assert jnp.allclose(
            process(Data(jnp.array([1, 2]), jnp.array([3, 4]))), jnp.array([4, 6])
        )

        # Non-array
        @no_jax_tracing
        def format_val(x, message):
            return f"{message}: {x}"

        assert format_val(42, "value") == "value: 42"

    def test_preserves_name_and_error_message(self):
        """Test function name preservation and error message content."""

        @no_jax_tracing
        def my_named_function(x):
            return x * 2

        assert my_named_function.__name__ == "my_named_function"

        with pytest.raises(NotJaxCompatibleError, match="my_named_function"):
            jax.jit(my_named_function)(jnp.array([1, 2, 3]))


class TestSkipIfDisabled:
    """Tests for skip_if_disabled / no_post_init."""

    def test_skip_if_disabled(self):
        @dataclass
        class Validated:
            x: int

            @skip_post_init_if_disabled
            def __post_init__(self):
                if self.x < 0:
                    raise ValueError("x must be non-negative")

        # Runs by default
        Validated(x=1)
        with pytest.raises(ValueError, match="non-negative"):
            Validated(x=-1)

        # Suppressed inside context
        with no_post_init():
            obj = Validated(x=-1)
        assert obj.x == -1

        # Restored after context
        with pytest.raises(ValueError, match="non-negative"):
            Validated(x=-1)
