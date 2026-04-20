# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JAX-PyTorch bridge.

Requires the torch_dev dependency group: `uv sync --group torch_dev`
"""

import jax
import jax.numpy as jnp
import pytest

# Skip entire module if torch not available
torch = pytest.importorskip("torch", minversion="2.0.0")


class TestDtypeConversion:
    """Tests for dtype conversion between JAX and PyTorch."""

    def test_all_supported_round_trip(self):
        """Test all supported JAX<->torch dtype conversions round-trip."""
        from kups.core.utils.torch import (
            _JAX_TO_TORCH_DTYPE,
            _jax_to_torch_dtype,
            _torch_to_jax_dtype,
        )

        for jax_name, expected_torch in _JAX_TO_TORCH_DTYPE.items():
            try:
                jax_dtype = jnp.dtype(jax_name)
                assert _jax_to_torch_dtype(jax_dtype) == expected_torch, (
                    f"jax->torch failed for {jax_name}"
                )
                assert _torch_to_jax_dtype(expected_torch) == jax_dtype, (
                    f"torch->jax failed for {jax_name}"
                )
            except TypeError:
                pass

    def test_unsupported_raises(self):
        """Test unsupported dtypes raise ValueError."""
        from kups.core.utils.torch import _jax_to_torch_dtype, _torch_to_jax_dtype

        with pytest.raises(ValueError, match="Unsupported JAX dtype"):
            _jax_to_torch_dtype("unsupported_dtype")
        with pytest.raises(ValueError, match="Unsupported torch dtype"):
            _torch_to_jax_dtype("not_a_torch_dtype")


class TestGetModuleDevice:
    def test_get_module_device(self):
        from kups.core.utils.torch import _get_module_device

        # Module with parameters
        assert _get_module_device(torch.nn.Linear(10, 5)) == torch.device("cpu")

        # Buffer-only module
        class BufferOnlyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(10))

        assert _get_module_device(BufferOnlyModule()) == torch.device("cpu")

        # Empty module
        class EmptyModule(torch.nn.Module):
            pass

        assert _get_module_device(EmptyModule()) == torch.device("cpu")


class TestTorchDtypeSpec:
    def test_dtype_spec_creation(self):
        from kups.core.utils.torch import TorchDtypeSpec

        spec = TorchDtypeSpec(shape=(4, 2), dtype=torch.float32)
        assert spec.shape == (4, 2)
        assert spec.dtype == torch.float32

    def test_dtype_spec_with_dynamic_dims(self):
        from kups.core.utils.torch import TorchDtypeSpec

        spec = TorchDtypeSpec(shape=(-1, 4), dtype=torch.float64)
        assert spec.shape == (-1, 4)


class TestAssembleFlat:
    """Tests for _assemble_flat."""

    def test_tensors_and_scalars(self):
        """Test tensors-only and scalar-interleaved assembly."""
        from kups.core.utils.torch import ScalarSpec, _assemble_flat

        t1 = torch.ones(4, dtype=torch.float32)
        t2 = torch.zeros(3, dtype=torch.int32)

        # Tensors only
        result = _assemble_flat([t1, t2], [None, None], [])
        assert result[0] is t1 and result[1] is t2

        # Scalar interleaved
        t = torch.ones(4, dtype=torch.float32)
        spec = [None, ScalarSpec(python_type=int)]
        result = _assemble_flat([t], spec, [3])
        assert result[0] is t and result[1] == 3 and isinstance(result[1], int)

    def test_dtype_cast(self):
        """Test dtype cast applied / no-op when matching."""
        from kups.core.utils.torch import InputSpecLeaf, TorchDtypeSpec, _assemble_flat

        # Cast needed
        t64 = torch.ones(4, dtype=torch.float64)
        spec: list[InputSpecLeaf] = [TorchDtypeSpec(shape=(4,), dtype=torch.float32)]
        assert _assemble_flat([t64], spec, [])[0].dtype == torch.float32

        # No cast needed
        t32 = torch.ones(4, dtype=torch.float32)
        assert _assemble_flat([t32], spec, [])[0] is t32

        # Mixed
        from kups.core.utils.torch import ScalarSpec

        t1 = torch.ones(4, dtype=torch.float64)
        t2 = torch.zeros(2, dtype=torch.int32)
        mixed_spec = [
            TorchDtypeSpec(shape=(4,), dtype=torch.float32),
            ScalarSpec(python_type=bool),
            None,
        ]
        result = _assemble_flat([t1, t2], mixed_spec, [True])
        assert (
            result[0].dtype == torch.float32 and result[1] is True and result[2] is t2
        )


class TestInputSpecValidation:
    def test_error_cases(self):
        from kups.core.utils.torch import ScalarSpec, _validate_input_spec

        with pytest.raises(ValueError, match="input_spec has 2 entries, got 1"):
            _validate_input_spec([jnp.ones(4)], [None, None])
        with pytest.raises(TypeError, match="expected int"):
            _validate_input_spec([3.14], [ScalarSpec(python_type=int)])
        with pytest.raises(TypeError, match="spec declares tensor but got scalar"):
            _validate_input_spec([3.14], [None])
        with pytest.raises(TypeError, match="must be array-like"):
            _validate_input_spec(["not_an_array"], [None])

    def test_valid_specs_pass(self):
        from kups.core.utils.torch import (
            ScalarSpec,
            TorchDtypeSpec,
            _validate_input_spec,
        )

        _validate_input_spec(
            [jnp.ones(4), 3, 2.0],
            [
                TorchDtypeSpec(shape=(4,), dtype=torch.float32),
                ScalarSpec(python_type=int),
                ScalarSpec(python_type=float),
            ],
        )
        _validate_input_spec([jnp.ones(4), 2.0], [None, ScalarSpec(python_type=float)])


def _make_args_info(args_tuple):
    """Build (array_leaves, flat_spec, scalar_vals, in_tree) for tests."""
    from kups.core.utils.torch import ScalarSpec, _infer_spec

    args_flat, in_tree = jax.tree.flatten((args_tuple, {}))
    flat_spec = _infer_spec(args_flat)
    array_leaves = [
        a for a, s in zip(args_flat, flat_spec) if not isinstance(s, ScalarSpec)
    ]
    scalar_vals = [a for a, s in zip(args_flat, flat_spec) if isinstance(s, ScalarSpec)]
    return array_leaves, flat_spec, scalar_vals, in_tree


class TestTorchModuleWrapper:
    @pytest.fixture
    def simple_linear(self):
        return torch.nn.Linear(4, 2, bias=False)

    def test_wrapper_creation(self, simple_linear):
        from kups.core.utils.torch import TorchDtypeSpec, TorchModuleWrapper

        # Basic creation
        wrapper = TorchModuleWrapper(simple_linear)
        assert wrapper.module is simple_linear
        assert wrapper._compile is True
        assert wrapper.vmap_method == "broadcast_all"

        # With options
        w2 = TorchModuleWrapper(simple_linear, _compile=False, vmap_method="sequential")
        assert w2._compile is False and w2.vmap_method == "sequential"

        # With input spec
        spec = [TorchDtypeSpec(shape=(-1, 4), dtype=torch.float32)]
        w3 = TorchModuleWrapper(simple_linear, input_spec=spec)
        assert w3.input_spec is spec

    def test_pytree_and_cache(self, simple_linear):
        from kups.core.utils.torch import TorchModuleWrapper

        wrapper = TorchModuleWrapper(simple_linear)

        # Is pytree
        leaves, treedef = jax.tree_util.tree_flatten(wrapper)
        assert isinstance(
            jax.tree_util.tree_unflatten(treedef, leaves), TorchModuleWrapper
        )

        # Device cache
        device = torch.device("cpu")
        assert wrapper.get_for_device(device) is wrapper.get_for_device(device)

        # Output cache
        array_leaves, flat_spec, scalar_vals, in_tree = _make_args_info(
            (jnp.ones((3, 4), dtype=jnp.float32),)
        )
        info1 = wrapper._get_output_info(array_leaves, flat_spec, scalar_vals, in_tree)
        info2 = wrapper._get_output_info(array_leaves, flat_spec, scalar_vals, in_tree)
        assert info1 is info2

    def test_output_info_shapes(self, simple_linear):
        from kups.core.utils.torch import TorchModuleWrapper

        wrapper = TorchModuleWrapper(simple_linear)
        array_leaves, flat_spec, scalar_vals, in_tree = _make_args_info(
            (jnp.ones((3, 4), dtype=jnp.float32),)
        )
        output_shapes, _ = wrapper._get_output_info(
            array_leaves, flat_spec, scalar_vals, in_tree
        )
        assert len(output_shapes) == 1
        assert output_shapes[0].shape == (3, 2)
        assert output_shapes[0].dtype == jnp.float32


class TestInputValidation:
    def test_invalid_input_raises(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class DummyModule(torch.nn.Module):
            def forward(self, x):
                return x

        wrapper = TorchModuleWrapper(DummyModule())
        with pytest.raises(TypeError, match="must be array-like"):
            wrapper("not_an_array")

    def test_scalar_spec_type_mismatch_at_call_time(self):
        """ScalarSpec(int) but receives float -- raises TypeError via __call__."""
        from kups.core.utils.torch import ScalarSpec, TorchModuleWrapper

        class DummyModule(torch.nn.Module):
            def forward(self, x, n):
                return x

        wrapper = TorchModuleWrapper(
            DummyModule(), input_spec=[None, ScalarSpec(python_type=int)]
        )
        with pytest.raises(TypeError, match="expected int"):
            wrapper(jnp.ones(4), 3.14)


class TestInputSpecDtypes:
    """Tests for dtype specification via TorchDtypeSpec in _get_output_info."""

    def test_mock_uses_spec_dtype(self):
        """_get_output_info applies TorchDtypeSpec cast in mock forward."""
        from kups.core.utils.torch import (
            InputSpecLeaf,
            TorchDtypeSpec,
            TorchModuleWrapper,
        )

        received_dtype: dict = {}

        class DtypeModule(torch.nn.Module):
            def forward(self, x):
                received_dtype["dtype"] = x.dtype
                return x

        wrapper = TorchModuleWrapper(DtypeModule())
        array_leaf = jnp.ones((3, 4), dtype=jnp.float64)
        flat_spec: list[InputSpecLeaf] = [
            TorchDtypeSpec(shape=(-1, 4), dtype=torch.float32)
        ]
        _, in_tree = jax.tree.flatten(([array_leaf], {}))

        wrapper._get_output_info([array_leaf], flat_spec, [], in_tree)
        assert received_dtype["dtype"] == torch.float32

    def test_mock_uses_jax_dtype_when_no_spec(self):
        """_get_output_info uses JAX-inferred dtype when spec is None."""
        from kups.core.utils.torch import ScalarSpec, TorchModuleWrapper, _infer_spec

        received_dtype: dict = {}

        class DtypeModule(torch.nn.Module):
            def forward(self, x):
                received_dtype["dtype"] = x.dtype
                return x

        wrapper = TorchModuleWrapper(DtypeModule())
        array_leaf = jnp.ones((3, 4), dtype=jnp.float32)
        flat_spec = _infer_spec([array_leaf])
        array_leaves = [
            a for a, s in zip([array_leaf], flat_spec) if not isinstance(s, ScalarSpec)
        ]
        _, in_tree = jax.tree.flatten(([array_leaf], {}))

        wrapper._get_output_info(array_leaves, flat_spec, [], in_tree)
        assert received_dtype["dtype"] == torch.float32

    def test_scalar_output_raises_type_error(self):
        """Module returning a Python scalar in its output raises TypeError."""
        from kups.core.utils.torch import ScalarSpec, TorchModuleWrapper, _infer_spec

        class ScalarOutputModule(torch.nn.Module):
            def forward(self, x):
                return x, 42

        wrapper = TorchModuleWrapper(ScalarOutputModule())
        array_leaf = jnp.ones((3,), dtype=jnp.float32)
        flat_spec = _infer_spec([array_leaf])
        array_leaves = [
            a for a, s in zip([array_leaf], flat_spec) if not isinstance(s, ScalarSpec)
        ]
        _, in_tree = jax.tree.flatten(([array_leaf], {}))

        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            wrapper._get_output_info(array_leaves, flat_spec, [], in_tree)


class TestScalarSpecWithWrapper:
    """Tests for ScalarSpec in input_spec (CPU-only, via mock forward)."""

    def test_scalar_spec_passes_as_python_value(self):
        """ScalarSpec input arrives as a Python scalar in the mock forward."""
        from kups.core.utils.torch import TorchModuleWrapper

        received_types: dict = {}

        class ScaleModule(torch.nn.Module):
            def forward(self, x, scale):
                received_types["scale"] = type(scale)
                return x * scale

        wrapper = TorchModuleWrapper(ScaleModule())
        array_leaf = jnp.ones((3, 4), dtype=jnp.float32)

        array_leaves, flat_spec, scalar_vals, in_tree = _make_args_info(
            (array_leaf, 2.0)
        )

        output_shapes, _ = wrapper._get_output_info(
            array_leaves, flat_spec, scalar_vals, in_tree
        )
        assert output_shapes[0].shape == (3, 4)
        assert received_types["scale"] is float

    def test_infer_spec_bool_before_int(self):
        from kups.core.utils.torch import ScalarSpec, _infer_spec

        spec = _infer_spec([True, 3, 2.0])
        assert isinstance(spec[0], ScalarSpec) and spec[0].python_type is bool
        assert isinstance(spec[1], ScalarSpec) and spec[1].python_type is int
        assert isinstance(spec[2], ScalarSpec) and spec[2].python_type is float

    def test_infer_spec_array_gets_none(self):
        from kups.core.utils.torch import _infer_spec

        spec = _infer_spec([jnp.ones(4), jnp.zeros(3)])
        assert spec[0] is None
        assert spec[1] is None


class TestPythonScalarPassthrough:
    """Scalars must not be converted to arrays -- they pass through as Python values."""

    def test_scalar_arrives_as_python_type_in_mock_forward(self):
        """Mock forward receives bool as bool, not tensor."""
        from kups.core.utils.torch import TorchModuleWrapper

        received_types: dict = {}

        class TypeCheckModule(torch.nn.Module):
            def forward(self, x, flag):
                received_types["x"] = type(x)
                received_types["flag"] = type(flag)
                return x * 2 if flag else x

        wrapper = TorchModuleWrapper(TypeCheckModule())
        array_leaf = jnp.ones((3, 4), dtype=jnp.float32)

        array_leaves, flat_spec, scalar_vals, in_tree = _make_args_info(
            (array_leaf, True)
        )

        wrapper._get_output_info(array_leaves, flat_spec, scalar_vals, in_tree)
        assert received_types["x"] == torch.Tensor
        assert received_types["flag"] is bool

    def test_int_scalar_passes_as_int(self):
        """Mock forward receives int as Python int (used in repeat)."""
        from kups.core.utils.torch import TorchModuleWrapper

        received_types: dict = {}

        class RepeatModule(torch.nn.Module):
            def forward(self, x, n):
                received_types["n"] = type(n)
                return x.repeat(n, 1)

        wrapper = TorchModuleWrapper(RepeatModule())
        array_leaf = jnp.ones((1, 4), dtype=jnp.float32)

        array_leaves, flat_spec, scalar_vals, in_tree = _make_args_info((array_leaf, 3))

        output_shapes, _ = wrapper._get_output_info(
            array_leaves, flat_spec, scalar_vals, in_tree
        )
        assert received_types["n"] is int
        assert output_shapes[0].shape == (3, 4)

    def test_different_scalar_values_give_different_cache_entries(self):
        """scalar_vals=(2,) and scalar_vals=(3,) -> different cache keys."""
        from kups.core.utils.torch import TorchModuleWrapper

        class RepeatModule(torch.nn.Module):
            def forward(self, x, n):
                return x.repeat(n, 1)

        wrapper = TorchModuleWrapper(RepeatModule())
        array_leaf = jnp.ones((1, 4), dtype=jnp.float32)

        al2, fs2, sv2, it2 = _make_args_info((array_leaf, 2))
        al3, fs3, sv3, it3 = _make_args_info((array_leaf, 3))

        info_2 = wrapper._get_output_info(al2, fs2, sv2, it2)
        info_3 = wrapper._get_output_info(al3, fs3, sv3, it3)

        assert info_2[0][0].shape == (2, 4)
        assert info_3[0][0].shape == (3, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_python_int_passes_through(self):
        from kups.core.utils.torch import TorchModuleWrapper

        received_types: dict = {}

        class IntModule(torch.nn.Module):
            def forward(self, x, n):
                received_types["n"] = type(n)
                return x * n

        wrapper = TorchModuleWrapper(IntModule().cuda())
        result = wrapper(jnp.ones(4, dtype=jnp.float32), 3)
        assert received_types["n"] is int
        assert jnp.allclose(result, jnp.full(4, 3.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_python_float_passes_through(self):
        from kups.core.utils.torch import TorchModuleWrapper

        received_types: dict = {}

        class FloatModule(torch.nn.Module):
            def forward(self, x, scale):
                received_types["scale"] = type(scale)
                return x * scale

        wrapper = TorchModuleWrapper(FloatModule().cuda())
        result = wrapper(jnp.ones(4, dtype=jnp.float32), 2.5)
        assert received_types["scale"] is float
        assert jnp.allclose(result, jnp.full(4, 2.5))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_python_bool_passes_through(self):
        from kups.core.utils.torch import TorchModuleWrapper

        received_types: dict = {}

        class FlagModule(torch.nn.Module):
            def forward(self, x, double_it):
                received_types["double_it"] = type(double_it)
                return x * 2 if double_it else x

        wrapper = TorchModuleWrapper(FlagModule().cuda())
        result = wrapper(jnp.ones(4, dtype=jnp.float32), True)
        assert received_types["double_it"] is bool
        assert jnp.allclose(result, jnp.full(4, 2.0))


class TestWrapperMultipleUses:
    def test_wrapper_different_shapes(self):
        from kups.core.utils.torch import TorchModuleWrapper

        module = torch.nn.Linear(4, 2)
        wrapper = TorchModuleWrapper(module)

        al1, fs1, sv1, it1 = _make_args_info((jnp.ones((3, 4), dtype=jnp.float32),))
        al2, fs2, sv2, it2 = _make_args_info((jnp.ones((5, 4), dtype=jnp.float32),))

        info1 = wrapper._get_output_info(al1, fs1, sv1, it1)
        info2 = wrapper._get_output_info(al2, fs2, sv2, it2)

        assert info1[0][0].shape == (3, 2)
        assert info2[0][0].shape == (5, 2)


# Integration tests for actual forward passes require CUDA.
# To run: uv run --group torch_dev pytest test/core/test_torch.py -v -k "Integration"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestIntegration:
    def test_forward_pass(self):
        from kups.core.utils.torch import TorchModuleWrapper

        model = torch.nn.Linear(4, 2).cuda()
        result = TorchModuleWrapper(model)(jnp.ones((3, 4), dtype=jnp.float32))
        assert result.shape == (3, 2)

    def test_jit_compilation(self):
        from kups.core.utils.torch import TorchModuleWrapper

        model = torch.nn.Linear(4, 2).cuda()
        jitted = jax.jit(TorchModuleWrapper(model))
        assert jitted(jnp.ones((3, 4), dtype=jnp.float32)).shape == (3, 2)

    def test_vmap_broadcast(self):
        from kups.core.utils.torch import TorchModuleWrapper

        model = torch.nn.Linear(4, 2).cuda()
        vmapped = jax.vmap(TorchModuleWrapper(model, vmap_method="broadcast_all"))
        assert vmapped(jnp.ones((8, 4), dtype=jnp.float32)).shape == (8, 2)

    def test_correctness_forward(self):
        from kups.core.utils.torch import TorchModuleWrapper

        model = torch.nn.Linear(4, 2, bias=False).cuda()
        with torch.no_grad():
            model.weight.copy_(torch.eye(2, 4))

        result = TorchModuleWrapper(model)(
            jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
        )
        assert jnp.allclose(result, jnp.array([[1.0, 2.0]]), atol=1e-6)

    def test_scalar_input(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class ScaleModule(torch.nn.Module):
            def forward(self, x, scale):
                return x * scale

        result = TorchModuleWrapper(ScaleModule().cuda())(
            jnp.ones(4, dtype=jnp.float32), 2.0
        )
        assert jnp.allclose(result, jnp.full(4, 2.0))

    def test_dtype_spec_casting(self):
        from kups.core.utils.torch import TorchDtypeSpec, TorchModuleWrapper

        model = torch.nn.Linear(4, 2).cuda()
        wrapper = TorchModuleWrapper(
            model, input_spec=[TorchDtypeSpec(shape=(-1, 4), dtype=torch.float32)]
        )
        assert wrapper(jnp.ones((1, 4), dtype=jnp.float64)).shape == (1, 2)

    def test_empty_batch(self):
        from kups.core.utils.torch import TorchModuleWrapper

        model = torch.nn.Linear(4, 2).cuda()
        assert TorchModuleWrapper(model)(jnp.ones((0, 4), dtype=jnp.float32)).shape == (
            0,
            2,
        )

    def test_explicit_gradient_pattern(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class GradModule(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                x = x.detach().requires_grad_(True)
                y = self.inner(x)
                grad = torch.autograd.grad(y.sum(), x)[0]
                return y.detach(), grad.detach()

        model = torch.nn.Linear(4, 2).cuda()
        y, grad = TorchModuleWrapper(GradModule(model))(
            jnp.ones((1, 4), dtype=jnp.float32)
        )
        assert y.shape == (1, 2)
        assert grad.shape == (1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVJPSupport:
    def test_vjp_scalar_output(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class SimpleEnergy(torch.nn.Module):
            def forward(self, positions):
                return (positions**2).sum()

        wrapper = TorchModuleWrapper(SimpleEnergy().cuda(), enable_vjp=True)
        positions = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        forces = jax.grad(wrapper)(positions)
        assert jnp.allclose(forces, 2 * positions, atol=1e-5)

    def test_vjp_matches_torch_autograd(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class QuadraticEnergy(torch.nn.Module):
            def forward(self, x):
                return (x**3).sum()

        wrapper = TorchModuleWrapper(QuadraticEnergy().cuda(), enable_vjp=True)
        positions = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
        jax_grad = jax.grad(lambda x: wrapper(x))(positions)

        x_torch = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, device="cuda"
        )
        x_torch.requires_grad_(True)
        (x_torch**3).sum().backward()
        assert jnp.allclose(jax_grad, x_torch.grad.cpu().numpy(), atol=1e-5)

    def test_vjp_with_jit(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class SimpleEnergy(torch.nn.Module):
            def forward(self, positions):
                return (positions**2).sum()

        wrapper = TorchModuleWrapper(SimpleEnergy().cuda(), enable_vjp=True)

        @jax.jit
        def compute_forces(x):
            return jax.grad(lambda pos: wrapper(pos))(x)

        positions = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        assert jnp.allclose(compute_forces(positions), 2 * positions, atol=1e-5)

    def test_vjp_multiple_inputs_partial_differentiation(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class WeightedSum(torch.nn.Module):
            def forward(self, x, weights):
                return (x * weights).sum()

        wrapper = TorchModuleWrapper(WeightedSum().cuda(), enable_vjp=True)
        x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        weights = jnp.array([[2.0, 3.0, 4.0]], dtype=jnp.float32)
        grad_x = jax.grad(lambda pos: wrapper(pos, weights))(x)
        assert jnp.allclose(grad_x, weights, atol=1e-5)

    def test_vjp_integer_input_ignored(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class IndexedSum(torch.nn.Module):
            def forward(self, x, indices):
                return x.sum()

        wrapper = TorchModuleWrapper(IndexedSum().cuda(), enable_vjp=True)
        x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        indices = jnp.array([0, 1, 2], dtype=jnp.int32)
        grad_x = jax.grad(lambda pos: wrapper(pos, indices))(x)
        assert jnp.allclose(grad_x, jnp.ones_like(x), atol=1e-5)

    def test_vjp_with_linear_layer(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class LinearEnergy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(torch.ones(2, 4))

            def forward(self, x):
                return self.linear(x).sum()

        wrapper = TorchModuleWrapper(LinearEnergy().cuda(), enable_vjp=True)
        x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
        assert jnp.allclose(
            jax.grad(lambda pos: wrapper(pos))(x), jnp.full_like(x, 2.0), atol=1e-5
        )

    def test_vjp_preserves_forward_behavior(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x**2

        model = SimpleModule().cuda()
        x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        result_no_vjp = TorchModuleWrapper(model, enable_vjp=False)(x)
        result_vjp = TorchModuleWrapper(model, enable_vjp=True)(x)
        assert jnp.allclose(result_no_vjp, result_vjp, atol=1e-6)

    def test_vjp_vector_output(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class VectorOutput(torch.nn.Module):
            def forward(self, x):
                return x**2

        wrapper = TorchModuleWrapper(VectorOutput().cuda(), enable_vjp=True)
        x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        primals, vjp_fn = jax.vjp(wrapper, x)
        (grad,) = vjp_fn(jnp.ones_like(primals))
        assert jnp.allclose(grad, 2 * x, atol=1e-5)

    def test_vjp_tuple_output(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class TupleOutput(torch.nn.Module):
            def forward(self, x):
                return (x**2).sum(), -2 * x

        wrapper = TorchModuleWrapper(TupleOutput().cuda(), enable_vjp=True)
        x = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        grad = jax.grad(lambda pos: wrapper(pos)[0])(x)
        assert jnp.allclose(grad, 2 * x, atol=1e-5)

    def test_vjp_with_vmap(self):
        from kups.core.utils.torch import TorchModuleWrapper

        class SimpleEnergy(torch.nn.Module):
            def forward(self, positions):
                return (positions**2).sum()

        wrapper = TorchModuleWrapper(SimpleEnergy().cuda(), enable_vjp=True)
        batched = jnp.array(
            [[[1.0, 2.0, 3.0]], [[2.0, 3.0, 4.0]], [[0.5, 1.0, 1.5]]], dtype=jnp.float32
        )
        grads = jax.vmap(jax.grad(lambda pos: wrapper(pos)))(batched)
        assert jnp.allclose(grads, 2 * batched, atol=1e-5)
