# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""JAX-PyTorch interoperability bridge for neural network potentials.

Enables calling PyTorch nn.Module instances from JAX code. Uses DLPack for
zero-copy tensor sharing and supports multi-GPU execution via per-device
module caching.

Gradient Support:
    For JAX autodiff through PyTorch models, use enable_vjp=True:

        wrapper = TorchModuleWrapper(model, enable_vjp=True)
        forces = -jax.grad(lambda x: wrapper(x).sum())(positions)

    Limitations:
    - Nested differentiation (Hessians) is NOT supported
    - Module outputs must preserve grad_fn (not call .detach())

    For modules computing gradients internally (e.g., MACE with compute_force=True),
    use requires_grad=True instead of enable_vjp=True:

        class GradModule(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                x = x.detach().requires_grad_(True)
                y = self.inner(x)
                grad = torch.autograd.grad(y.sum(), x)[0]
                return y.detach(), grad.detach()

        wrapper = TorchModuleWrapper(GradModule(model), requires_grad=True)

Example:
    ```python
    import torch
    from kups.core.utils.torch import TorchModuleWrapper

    model = torch.nn.Linear(10, 5)
    wrapper = TorchModuleWrapper(model)

    output = wrapper(jax_array)
    ```

Requires the `torch_dev` dependency group: `uv sync --group torch_dev`
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

import jax
import jax.core
import jax.experimental.buffer_callback as jbc
import jax.numpy as jnp

try:
    import torch  # pyright: ignore[reportMissingImports]
    import torch.utils.dlpack as torch_dlpack  # pyright: ignore[reportMissingImports]
except ImportError:
    raise ValueError(
        "Torch not found. "
        "Using the kUPS torch bridge requires install with the [torch_dev] dependency group."
    )

from kups.core.utils.jax import dataclass, field

__all__ = [
    "TorchModuleWrapper",
    "TorchDtypeSpec",
    "ScalarSpec",
    "InputSpecLeaf",
]

_DEVICE_CACHE: dict[int, dict[torch.device, torch.nn.Module]] = {}
_OUTPUT_CACHE: dict[int, dict[tuple, tuple]] = {}
_CACHE_LOCK = threading.Lock()

_JAX_TO_TORCH_DTYPE = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

_TORCH_TO_JAX_DTYPE = {
    torch.bool: jnp.bool_,
    torch.uint8: jnp.uint8,
    torch.int8: jnp.int8,
    torch.int16: jnp.int16,
    torch.int32: jnp.int32,
    torch.int64: jnp.int64,
    torch.float16: jnp.float16,
    torch.float32: jnp.float32,
    torch.float64: jnp.float64,
    torch.bfloat16: jnp.bfloat16,
    torch.complex64: jnp.complex64,
    torch.complex128: jnp.complex128,
}


def _jax_to_torch_dtype(jax_dtype: Any) -> torch.dtype:
    """Convert JAX dtype to torch dtype."""
    name = jax_dtype.name if hasattr(jax_dtype, "name") else str(jax_dtype)
    if name not in _JAX_TO_TORCH_DTYPE:
        raise ValueError(f"Unsupported JAX dtype: {jax_dtype}")
    return _JAX_TO_TORCH_DTYPE[name]


def _torch_to_jax_dtype(torch_dtype: Any) -> Any:
    """Convert torch dtype to JAX dtype."""
    if torch_dtype not in _TORCH_TO_JAX_DTYPE:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return _TORCH_TO_JAX_DTYPE[torch_dtype]


def _get_module_device(module: torch.nn.Module) -> torch.device:
    """Infer device from module's first parameter or buffer."""
    return next(
        (p.device for p in module.parameters()),
        next((b.device for b in module.buffers()), torch.device("cpu")),
    )


type Scalar = int | float | bool


@dataclass
class TorchDtypeSpec:
    """Specification for expected torch tensor dtype.

    Use when JAX/PyTorch dtype promotion rules would cause mismatches.
    For example, when JAX x64 mode produces float64 but torch expects float32.

    Attributes:
        shape: Expected tensor shape. Use -1 for dynamic dimensions.
        dtype: Target torch dtype to cast to.

    Example:
        ```python
        # Handle x64 mode where JAX produces float64 but torch expects float32
        spec = TorchDtypeSpec(shape=(-1, 4), dtype=torch.float32)
        wrapper = TorchModuleWrapper(model, input_spec=[spec])

        x = jnp.ones((1, 4), dtype=jnp.float64)
        result = wrapper(x)  # Input cast to float32
        ```
    """

    shape: tuple[int, ...] = field(static=True)
    dtype: torch.dtype = field(static=True)


@dataclass
class ScalarSpec:
    """Declares an input is a Python scalar of the given type.

    Use in input_spec to explicitly mark an argument as a Python scalar
    (int, float, or bool) rather than a JAX array. Scalars pass through
    to the module unchanged — they are not converted to tensors.

    Attributes:
        python_type: The expected Python type (int, float, or bool).

    Example:
        ```python
        spec = [None, ScalarSpec(python_type=int)]
        wrapper = TorchModuleWrapper(model, input_spec=spec)
        result = wrapper(jax_array, 3)  # 3 stays as Python int
        ```
    """

    python_type: type[int] | type[float] | type[bool] = field(static=True)


type InputSpecLeaf = TorchDtypeSpec | ScalarSpec | None


def _infer_spec(flat_args: list[Any]) -> list[InputSpecLeaf]:
    """Infer input spec from actual arguments."""
    return [
        ScalarSpec(python_type=bool if isinstance(a, bool) else type(a))
        if isinstance(a, (int, float, bool))
        else None
        for a in flat_args
    ]


def _validate_input_spec(flat_args: list[Any], flat_spec: list[InputSpecLeaf]) -> None:
    """Validate that arguments match their input spec declarations."""
    if len(flat_spec) != len(flat_args):
        raise ValueError(
            f"input_spec has {len(flat_spec)} entries, got {len(flat_args)}"
        )
    for i, (arg, spec) in enumerate(zip(flat_args, flat_spec, strict=True)):
        if isinstance(spec, ScalarSpec):
            if not isinstance(arg, spec.python_type):
                raise TypeError(
                    f"input_spec[{i}]: expected {spec.python_type.__name__}, "
                    f"got {type(arg).__name__}"
                )
        elif isinstance(arg, (int, float, bool)):
            raise TypeError(
                f"input_spec[{i}]: spec declares tensor but got scalar "
                f"{type(arg).__name__}"
            )
        elif not (hasattr(arg, "shape") and hasattr(arg, "dtype")):
            raise TypeError(
                f"Argument {i} must be array-like with shape and dtype, "
                f"got {type(arg).__name__}."
            )


def _assemble_flat(
    tensors: list[torch.Tensor],
    flat_spec: list[InputSpecLeaf],
    scalar_vals: list[Any],
) -> list[Any]:
    """Build full flat input list: interleave tensors and scalars, apply dtype casts.

    Args:
        tensors: One torch.Tensor per non-scalar position (array_leaves order).
        flat_spec: One InputSpecLeaf per original argument.
        scalar_vals: Python scalar values, one per ScalarSpec in flat_spec.

    Returns:
        Flat list matching the original args_flat order, ready for
        jax.tree.unflatten(in_tree, result).
    """
    scalar_it = iter(scalar_vals)
    tensor_it = iter(tensors)
    result = []
    for spec in flat_spec:
        if isinstance(spec, ScalarSpec):
            result.append(next(scalar_it))
        else:
            t = next(tensor_it)
            result.append(
                t.to(spec.dtype)
                if isinstance(spec, TorchDtypeSpec) and t.dtype != spec.dtype
                else t
            )
    return result


@dataclass
class TorchModuleWrapper:
    """Wraps a PyTorch nn.Module for use in JAX.

    Enables calling PyTorch nn.Module instances from JAX code.
    Handles device placement via per-device caching and DLPack-based
    zero-copy tensor conversion.

    Gradient Support:
        For JAX autodiff through PyTorch models, use enable_vjp=True:

            wrapper = TorchModuleWrapper(model, enable_vjp=True)
            forces = -jax.grad(lambda x: wrapper(x).sum())(positions)

        Limitations:
        - Nested differentiation (Hessians) is NOT supported
        - Module outputs must preserve grad_fn (not call .detach())

        For modules computing gradients internally (e.g., MACE with compute_force=True),
        use requires_grad=True instead of enable_vjp=True.

    Attributes:
        module: PyTorch module to wrap.
        input_spec: Optional flat list of InputSpecLeaf (one per flattened argument).
            Use TorchDtypeSpec for explicit dtype casting, ScalarSpec to declare
            Python scalar inputs, or None for tensors with inferred dtype.
            If None (default), spec is inferred automatically.
        vmap_method: How to handle vmap. Default "broadcast_all" assumes the
            module handles batching natively. Use "sequential" for modules
            that don't support batching.
        requires_grad: Whether to enable gradients during forward pass. Default
            False (uses torch.no_grad for better performance). Set to True for
            modules that use autograd internally (e.g., MACE with compute_force=True).
        _compile: Whether to use torch.compile for optimization. Default True.

    Example:
        ```python
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )
        wrapper = TorchModuleWrapper(model)

        output = wrapper(jax_input)

        # With jit
        jitted = jax.jit(wrapper)
        output = jitted(jax_input)

        # With vmap (module must handle batching)
        vmapped = jax.vmap(wrapper)
        output = vmapped(batched_input)
        ```
    """

    module: torch.nn.Module = field(static=True)
    input_spec: Any | None = field(static=True, default=None)
    vmap_method: str = field(static=True, default="broadcast_all")
    requires_grad: bool = field(static=True, default=False)
    enable_vjp: bool = field(static=True, default=False)
    _compile: bool = field(static=True, default=True)

    def get_for_device(self, device: torch.device) -> torch.nn.Module:
        """Get module for a specific device, caching for efficiency."""
        wrapper_id = id(self)
        with _CACHE_LOCK:
            if wrapper_id not in _DEVICE_CACHE:
                _DEVICE_CACHE[wrapper_id] = {}
            cache = _DEVICE_CACHE[wrapper_id]
            if device not in cache:
                cache[device] = self.module.to(device)
            return cache[device]

    def _grad_context(self) -> Any:
        """Return context manager for gradient control.

        Returns torch.no_grad() for better performance when requires_grad=False,
        or nullcontext() when requires_grad=True (for modules using internal autograd).
        """
        if self.requires_grad:
            return contextlib.nullcontext()
        return torch.no_grad()

    def _get_output_info(
        self,
        array_leaves: list[Any],
        flat_spec: list[InputSpecLeaf],
        scalar_vals: list[Any],
        in_tree: Any,
    ) -> tuple[tuple[jax.core.ShapedArray, ...], Any]:
        """Get cached output shapes/dtypes, running mock forward if needed.

        Args:
            array_leaves: JAX array leaves (non-scalar inputs, in flat_spec order).
            flat_spec: One InputSpecLeaf per original argument.
            scalar_vals: Python scalar values, one per ScalarSpec in flat_spec.
            in_tree: Treedef for (args, kwargs) from jax.tree.flatten.
        """
        wrapper_id = id(self)
        shapes_key = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in array_leaves)
        key = (shapes_key, in_tree, tuple(flat_spec), tuple(scalar_vals))

        with _CACHE_LOCK:
            if wrapper_id not in _OUTPUT_CACHE:
                _OUTPUT_CACHE[wrapper_id] = {}
            cache = _OUTPUT_CACHE[wrapper_id]
            if key in cache:
                return cache[key]

        device = _get_module_device(self.module)

        # Create mock tensors with the correct dtype for each array position.
        # TorchDtypeSpec overrides JAX-inferred dtype; otherwise use JAX dtype.
        array_specs = [s for s in flat_spec if not isinstance(s, ScalarSpec)]
        mock_tensors = [
            torch.zeros(
                a.shape,
                dtype=(
                    s.dtype
                    if isinstance(s, TorchDtypeSpec)
                    else _jax_to_torch_dtype(a.dtype)
                ),
                device=device,
            )
            for a, s in zip(array_leaves, array_specs, strict=True)
        ]

        flat = _assemble_flat(mock_tensors, flat_spec, scalar_vals)
        fn_args, fn_kwargs = jax.tree.unflatten(in_tree, flat)

        with self._grad_context():
            out_torch = self.module(*fn_args, **fn_kwargs)

        out_flat, out_tree = jax.tree.flatten(out_torch)
        for i, t in enumerate(out_flat):
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"Module output leaf [{i}] must be a torch.Tensor, "
                    f"got {type(t).__name__}. buffer_callback requires tensor outputs only."
                )
        output_shapes = tuple(
            jax.core.ShapedArray(t.shape, _torch_to_jax_dtype(t.dtype))
            for t in out_flat
        )

        with _CACHE_LOCK:
            cache[key] = (output_shapes, out_tree)
        return cache[key]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped PyTorch module with JAX arrays.

        Python scalars (int, float, bool) are passed through to the PyTorch
        module unchanged — they are NOT converted to JAX/torch arrays.
        Only array-like inputs go through DLPack.
        """
        args_flat, in_tree = jax.tree.flatten((args, kwargs))

        flat_spec: list[InputSpecLeaf] = (
            _infer_spec(args_flat) if self.input_spec is None else self.input_spec
        )
        _validate_input_spec(args_flat, flat_spec)

        array_leaves = [
            a for a, s in zip(args_flat, flat_spec) if not isinstance(s, ScalarSpec)
        ]
        scalar_vals: list[Any] = [
            a for a, s in zip(args_flat, flat_spec) if isinstance(s, ScalarSpec)
        ]

        output_shapes, out_tree = self._get_output_info(
            array_leaves, flat_spec, scalar_vals, in_tree
        )

        if self.enable_vjp:
            return self._call_with_vjp(
                array_leaves, flat_spec, scalar_vals, in_tree, output_shapes, out_tree
            )
        return self._forward_impl(
            array_leaves, flat_spec, scalar_vals, in_tree, output_shapes, out_tree
        )

    def _forward_impl(
        self,
        array_leaves: list[Any],
        flat_spec: list[InputSpecLeaf],
        scalar_vals: list[Any],
        in_tree: Any,
        output_shapes: Any,
        out_tree: Any,
    ) -> Any:
        """Execute the forward pass via buffer_callback."""
        wrapper = self
        grad_context = self._grad_context

        def callback(cuda_ctx: Any, out_buffers: Any, *in_buffers: Any) -> None:
            torch_stream = torch.cuda.ExternalStream(cuda_ctx.stream)
            device = torch_stream.device

            with torch.cuda.stream(torch_stream), grad_context():
                tensors = [torch_dlpack.from_dlpack(b) for b in in_buffers]
                flat = _assemble_flat(tensors, flat_spec, scalar_vals)
                fn_args, fn_kwargs = jax.tree.unflatten(in_tree, flat)

                outputs_flat, _ = jax.tree.flatten(
                    wrapper.get_for_device(device)(*fn_args, **fn_kwargs)
                )
                for out_buf, tensor in zip(out_buffers, outputs_flat, strict=True):
                    torch_dlpack.from_dlpack(out_buf).copy_(tensor)

        out_flat = jbc.buffer_callback(
            callback=callback,
            result_shape_dtypes=output_shapes,
            vmap_method=self.vmap_method,
        )(*array_leaves)

        return jax.tree.unflatten(out_tree, out_flat)

    def _call_with_vjp(
        self,
        array_leaves: list[Any],
        flat_spec: list[InputSpecLeaf],
        scalar_vals: list[Any],
        in_tree: Any,
        output_shapes: Any,
        out_tree: Any,
    ) -> Any:
        """VJP-enabled call using jax.custom_vjp with symbolic_zeros."""
        wrapper = self

        @jax.custom_vjp
        def call_fn(*inputs: Any) -> Any:
            return wrapper._forward_impl(
                list(inputs), flat_spec, scalar_vals, in_tree, output_shapes, out_tree
            )

        def fwd(*wrapped_inputs: Any) -> Any:
            inputs = tuple(w.value for w in wrapped_inputs)
            perturbed = tuple(w.perturbed for w in wrapped_inputs)
            result = call_fn(*inputs)
            return result, (inputs, perturbed)

        def bwd(residuals: Any, g: Any) -> Any:
            inputs, perturbed = residuals
            return wrapper._backward_impl(
                inputs, perturbed, g, flat_spec, scalar_vals, in_tree
            )

        call_fn.defvjp(fwd, bwd, symbolic_zeros=True)
        return call_fn(*array_leaves)

    def _backward_impl(
        self,
        inputs: tuple[Any, ...],
        perturbed: tuple[bool, ...],
        grad_output: Any,
        flat_spec: list[InputSpecLeaf],
        scalar_vals: list[Any],
        in_tree: Any,
    ) -> tuple[Any | None, ...]:
        """Compute gradients via buffer_callback with PyTorch autograd."""
        wrapper = self

        needs_grad = [
            perturbed[i] and jnp.issubdtype(inp.dtype, jnp.floating)
            for i, inp in enumerate(inputs)
        ]

        if not any(needs_grad):
            return tuple(None for _ in inputs)

        grad_flat, _ = jax.tree.flatten(grad_output)

        grad_shapes = tuple(
            jax.core.ShapedArray(inp.shape, inp.dtype)
            for i, inp in enumerate(inputs)
            if needs_grad[i]
        )

        n_inputs = len(inputs)
        array_specs = [s for s in flat_spec if not isinstance(s, ScalarSpec)]

        def grad_callback(cuda_ctx: Any, out_buffers: Any, *in_buffers: Any) -> None:
            torch_stream = torch.cuda.ExternalStream(cuda_ctx.stream)
            device = torch_stream.device

            with torch.cuda.stream(torch_stream):
                raw_tensors = [
                    torch_dlpack.from_dlpack(b) for b in in_buffers[:n_inputs]
                ]

                requiring_grad: list[torch.Tensor] = []
                grad_inputs: list[torch.Tensor] = []
                for i, (t, spec) in enumerate(
                    zip(raw_tensors, array_specs, strict=True)
                ):
                    if isinstance(spec, TorchDtypeSpec) and t.dtype != spec.dtype:
                        t = t.to(spec.dtype)
                    if needs_grad[i]:
                        t = t.detach().requires_grad_(True)
                        requiring_grad.append(t)
                    grad_inputs.append(t)

                flat = _assemble_flat(grad_inputs, flat_spec, scalar_vals)
                fn_args, fn_kwargs = jax.tree.unflatten(in_tree, flat)

                outputs_flat, _ = jax.tree.flatten(
                    wrapper.get_for_device(device)(*fn_args, **fn_kwargs)
                )

                if any(needs_grad) and all(
                    not isinstance(o, torch.Tensor) or o.grad_fn is None
                    for o in outputs_flat
                ):
                    import warnings

                    warnings.warn(
                        "enable_vjp=True but module outputs have no grad_fn. "
                        "Gradients will be zero. Ensure module doesn't call .detach().",
                        UserWarning,
                    )

                grad_tensors = [
                    torch_dlpack.from_dlpack(buf) for buf in in_buffers[n_inputs:]
                ]

                grads = torch.autograd.grad(
                    outputs_flat,
                    requiring_grad,
                    grad_outputs=grad_tensors,
                    allow_unused=True,
                )

                for out_buf, grad in zip(out_buffers, grads, strict=True):
                    if grad is None:
                        torch_dlpack.from_dlpack(out_buf).zero_()
                    else:
                        torch_dlpack.from_dlpack(out_buf).copy_(grad)

        grad_results = jbc.buffer_callback(
            callback=grad_callback,
            result_shape_dtypes=grad_shapes,
            vmap_method=self.vmap_method,
        )(*inputs, *grad_flat)

        result = []
        grad_idx = 0
        for i in range(len(inputs)):
            if needs_grad[i]:
                result.append(grad_results[grad_idx])
                grad_idx += 1
            else:
                result.append(None)

        return tuple(result)
