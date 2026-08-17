# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Segment sums accumulated in a wider float, and their gather adjoint.

Provides [segment_sum][kups.core.utils.segment.segment_sum], a drop-in
``jax.ops.segment_sum`` whose scatter accumulates in a wider float, and
[segment_take][kups.core.utils.segment.segment_take], the gather that is its
exact adjoint.
"""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.extend.mlir.dialects import stablehlo
from jax.interpreters import ad, batching, mlir, xla
from jax.typing import ArrayLike, DTypeLike

type Zero = ad.Zero
type Undefined = ad.UndefinedPrimal
type Mode = jax.lax.GatherScatterMode

_CLIP = jax.lax.GatherScatterMode.CLIP


def _accumulator_dtype(dtype: DTypeLike) -> np.dtype[Any]:
    """Next float wider than `dtype`.

    Args:
        dtype: Inexact dtype of the contributions.

    Returns:
        Numpy dtype of the accumulator, equal to `dtype` when nothing wider fits.
    """
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return np.dtype(np.complex128)
    if jnp.finfo(dtype).bits <= 16:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _tensor(shape: tuple[int, ...], element: Any) -> Any:
    """Build a ranked MLIR tensor type.

    Args:
        shape: Dimensions of the tensor.
        element: MLIR element type.

    Returns:
        Ranked tensor type of `shape` over `element`.
    """
    return mlir.ir.RankedTensorType.get(list(shape), element)


def _gather(data: Array, segment_ids: Array) -> Array:
    """Gather rows of `data`, ids outside ``[0, len(data))`` picking up zero.

    Uses the scatter's own out-of-range rule, so negative ids fill rather than
    index from the end as ``jnp.take`` would.

    Args:
        data: Rows to gather from, shape ``(num_segments, *feature_dims)``.
        segment_ids: Row index per output, of any shape.

    Returns:
        Gathered rows of shape ``(*segment_ids.shape, *feature_dims)``.
    """
    features = data.shape[1:]
    if data.shape[0] == 0:
        return jnp.zeros((*segment_ids.shape, *features), data.dtype)
    rows = jax.lax.gather(
        data,
        segment_ids.reshape(-1, 1),
        jax.lax.GatherDimensionNumbers(
            offset_dims=tuple(range(1, 1 + len(features))),
            collapsed_slice_dims=(0,),
            start_index_map=(0,),
        ),
        slice_sizes=(1, *features),
        mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
        fill_value=0,
    )
    return rows.reshape(*segment_ids.shape, *features)


def _resolve_mode(mode: Mode | str | None) -> Mode:
    """Normalise an out-of-range policy to the two modes this module honours.

    Args:
        mode: Policy as a ``jax.lax.GatherScatterMode``, as one of the strings
            ``"drop"``, ``"fill"`` or ``"clip"``, or ``None`` for the default.

    Returns:
        Either ``FILL_OR_DROP`` or ``CLIP``.

    Raises:
        ValueError: If `mode` is unknown, ``"promise_in_bounds"`` or ``"one_hot"``.
    """
    resolved = jax.lax.GatherScatterMode.from_any(mode)
    if resolved is jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS:
        raise ValueError(
            "mode='promise_in_bounds' is unsupported because XLA clamps an "
            "out-of-range gather but drops an out-of-range scatter, so the sum and "
            "the take would stop being adjoints and gradients would be silently "
            "wrong; say which you mean with 'drop' or 'clip'."
        )
    if resolved is jax.lax.GatherScatterMode.ONE_HOT:
        raise ValueError(
            "mode='one_hot' is unsupported because JAX has no gather lowering for "
            "it, only a dense one-hot matmul inside jnp.take_along_axis; use 'drop' "
            "or 'clip', or call jnp.take_along_axis directly for the matmul."
        )
    return resolved


def _clip_ids(segment_ids: Array, num_segments: int) -> Array:
    """Clamp ids into ``[0, num_segments)``, widening narrow ids for the bound.

    Args:
        segment_ids: Ids to clamp, of any shape.
        num_segments: Rows the ids address.

    Returns:
        Ids inside ``[0, num_segments)``, returned unchanged when `num_segments`
        is zero and there is no row to clamp to.
    """
    if num_segments == 0:
        return segment_ids
    if jnp.iinfo(segment_ids.dtype).max < num_segments:
        segment_ids = segment_ids.astype(jnp.int32)
    # Bounds in the ids' own dtype, which a Python int would widen under x64.
    limits = np.array([0, num_segments - 1], segment_ids.dtype)
    return jnp.clip(segment_ids, limits[0], limits[1])


def _static_fill(fill_value: ArrayLike) -> np.ndarray | None:
    """Value of `fill_value` when trace time already knows it.

    Args:
        fill_value: Candidate fill, a scalar or a traced value.

    Returns:
        `fill_value` as a numpy array, or ``None`` when it is traced and its value
        is unknown until run time.
    """
    try:
        return np.asarray(fill_value)
    except TypeError:
        return None


def _fill_out_of_range(
    rows: Array, segment_ids: Array, num_segments: int, fill_value: ArrayLike
) -> Array:
    """Replace rows whose id falls outside ``[0, num_segments)`` with `fill_value`.

    Args:
        rows: Gathered rows of shape ``(*segment_ids.shape, *feature_dims)``.
        segment_ids: Row index per output.
        num_segments: Rows of the table the ids address.
        fill_value: Value the out-of-range outputs take, cast to `rows`' dtype.

    Returns:
        `rows` with every out-of-range output replaced by `fill_value`.
    """
    valid = (segment_ids >= 0) & (segment_ids < num_segments)
    keep = valid.reshape(*valid.shape, *(1,) * (rows.ndim - valid.ndim))
    return jnp.where(keep, rows, jnp.asarray(fill_value, rows.dtype))


def _flatten_batch(
    data: Array,
    segment_ids: Array,
    data_dim: int | None,
    ids_dim: int,
    rows: int,
) -> tuple[Array, Array, int, tuple[int, ...]]:
    """Fold a mapped segmentation into one flat call with per-row id offsets.

    Batch row ``b`` addresses ``[b * rows, (b + 1) * rows)`` of the flattened
    output, and an id invalid in its own row is sent to ``size * rows``, out of
    range globally so that a drop cannot land in a neighbouring row.

    Args:
        data: Batched contributions, mapped over `data_dim`.
        segment_ids: Batched ids, mapped over `ids_dim`.
        data_dim: Mapped axis of `data`, or ``None`` when it is unmapped.
        ids_dim: Mapped axis of `segment_ids`.
        rows: Rows of the unmapped operand's leading axis.

    Returns:
        Tuple of flattened contributions, flattened ids whose invalid entries
        address no row, the batch size, and the trailing feature dims.
    """
    size = data.shape[data_dim] if data_dim is not None else segment_ids.shape[ids_dim]
    ids = batching.bdim_at_front(segment_ids, ids_dim, size)
    data = (
        jnp.broadcast_to(data, (size, *data.shape))
        if data_dim is None
        else jnp.moveaxis(data, data_dim, 0)
    )
    if jnp.iinfo(ids.dtype).max < size * rows:
        ids = ids.astype(jnp.int32)
    offset = jnp.arange(size, dtype=ids.dtype)[:, None] * rows
    valid = (ids >= 0) & (ids < rows)
    flat_ids = jnp.where(valid, ids + offset, size * rows).reshape(-1)
    flat_data = data.reshape(size * data.shape[1], *data.shape[2:])
    return flat_data, flat_ids, size, data.shape[2:]


segment_sum_p = Primitive("kups_segment_sum")
segment_take_p = Primitive("kups_segment_take")


@segment_sum_p.def_abstract_eval
def _segment_sum_abstract_eval(
    data: ShapedArray, segment_ids: ShapedArray, *, num_segments: int
) -> ShapedArray:
    """Bins of `data`'s dtype; the wide accumulator never reaches an aval."""
    del segment_ids
    return ShapedArray((num_segments, *data.shape[1:]), data.dtype)


@segment_take_p.def_abstract_eval
def _segment_take_abstract_eval(
    data: ShapedArray, segment_ids: ShapedArray
) -> ShapedArray:
    """One gathered row per id."""
    return ShapedArray((*segment_ids.shape, *data.shape[1:]), data.dtype)


segment_sum_p.def_impl(partial(xla.apply_primitive, segment_sum_p))
segment_take_p.def_impl(partial(xla.apply_primitive, segment_take_p))


def _segment_sum_lowering(
    ctx: mlir.LoweringRuleContext,
    data: Any,
    segment_ids: Any,
    *,
    num_segments: int,
) -> list[Any]:
    """Scatter-add into a wider accumulator, converted back on exit.

    Args:
        ctx: Lowering context carrying the input and output avals.
        data: MLIR value of the contributions.
        segment_ids: MLIR value of the non-negative ids.
        num_segments: Number of output bins, fixed by the output aval.

    Returns:
        Single-element list holding the binned MLIR value in the input's dtype.
    """
    del num_segments
    data_aval = cast(ShapedArray, ctx.avals_in[0])
    ids_aval = cast(ShapedArray, ctx.avals_in[1])
    out_aval = cast(ShapedArray, ctx.avals_out[0])
    narrow_dtype = np.dtype(out_aval.dtype)
    wide_dtype = _accumulator_dtype(narrow_dtype)
    widen = wide_dtype != narrow_dtype
    wide = mlir.dtype_to_ir_type(wide_dtype)
    operand = stablehlo.broadcast_in_dim(
        _tensor(out_aval.shape, wide),
        mlir.ir_constant(np.zeros((), wide_dtype)),
        mlir.dense_int_array([]),
    )
    updates = stablehlo.convert(_tensor(data_aval.shape, wide), data) if widen else data
    op = stablehlo.ScatterOp(
        [_tensor(out_aval.shape, wide)],
        [operand],
        segment_ids,
        [updates],
        stablehlo.ScatterDimensionNumbers.get(
            update_window_dims=list(range(1, len(data_aval.shape))),
            inserted_window_dims=[0],
            input_batching_dims=[],
            scatter_indices_batching_dims=[],
            scattered_dims_to_operand_dims=[0],
            index_vector_dim=len(ids_aval.shape),
        ),
        indices_are_sorted=mlir.ir.BoolAttr.get(False),
        unique_indices=mlir.ir.BoolAttr.get(False),
    )
    scalar = _tensor((), wide)
    block = op.update_computation.blocks.append(scalar, scalar)
    with mlir.ir.InsertionPoint(block):
        stablehlo.return_([stablehlo.add(block.arguments[0], block.arguments[1])])
    if not widen:
        return [op.results[0]]
    narrow = mlir.dtype_to_ir_type(narrow_dtype)
    return [stablehlo.convert(_tensor(out_aval.shape, narrow), op.results[0])]


mlir.register_lowering(segment_sum_p, _segment_sum_lowering)
mlir.register_lowering(segment_take_p, mlir.lower_fun(_gather, multiple_results=False))


def _segment_sum_jvp(
    tangent: Array, data: Array, segment_ids: Array, *, num_segments: int
) -> Array:
    """Rebind on the contribution tangent, the ids carrying none.

    Args:
        tangent: Tangent of the contributions.
        data: Contributions, needed only to place the ids.
        segment_ids: Bin index per contribution.
        num_segments: Number of output bins.

    Returns:
        Tangent of the bin sums.
    """
    del data
    return segment_sum_p.bind(tangent, segment_ids, num_segments=num_segments)


def _segment_sum_transpose(
    cotangent: Array | Zero,
    data: Array | Undefined,
    segment_ids: Array,
    *,
    num_segments: int,
) -> list[Array | Zero | None]:
    """Gather the cotangent back to each contribution.

    Args:
        cotangent: Cotangent of the bin sums.
        data: Contributions, undefined when they are the linear operand.
        segment_ids: Bin index per contribution.
        num_segments: Number of output bins, carried by `cotangent`'s shape.

    Returns:
        Cotangent per operand, ``None`` for the non-linear ids.
    """
    del num_segments
    if not ad.is_undefined_primal(data):
        return [None, None]
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(data.aval), None]
    return [segment_take_p.bind(cotangent, segment_ids), None]


def _segment_take_jvp(tangent: Array, data: Array, segment_ids: Array) -> Array:
    """Rebind on the row tangent, the ids carrying none.

    Args:
        tangent: Tangent of the table.
        data: Table, needed only to place the ids.
        segment_ids: Row index per output.

    Returns:
        Tangent of the gathered rows.
    """
    del data
    return segment_take_p.bind(tangent, segment_ids)


def _segment_take_transpose(
    cotangent: Array | Zero, data: Array | Undefined, segment_ids: Array
) -> list[Array | Zero | None]:
    """Sum the cotangents landing in each row, in the wide accumulator.

    Args:
        cotangent: Cotangent of the gathered rows.
        data: Table, undefined when it is the linear operand.
        segment_ids: Row index per output.

    Returns:
        Cotangent per operand, ``None`` for the non-linear ids.
    """
    if not ad.is_undefined_primal(data):
        return [None, None]
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(data.aval), None]
    return [
        segment_sum_p.bind(cotangent, segment_ids, num_segments=data.aval.shape[0]),
        None,
    ]


def _segment_sum_batcher(
    args: tuple[Array, Array],
    dims: tuple[int | None, int | None],
    *,
    num_segments: int,
) -> tuple[Array, int]:
    """Map the sum, an unmapped segmentation becoming a further feature axis.

    Args:
        args: Batched contributions and ids.
        dims: Mapped axis of each argument, ``None`` where it is unmapped.
        num_segments: Number of output bins per batch row.

    Returns:
        Tuple of the batched bin sums and their mapped axis.
    """
    data, segment_ids = args
    data_dim, ids_dim = dims
    if ids_dim is None:
        assert data_dim is not None
        data = jnp.moveaxis(data, data_dim, data.ndim - 1)
        out = segment_sum_p.bind(data, segment_ids, num_segments=num_segments)
        return out, out.ndim - 1
    flat_data, flat_ids, size, features = _flatten_batch(
        data, segment_ids, data_dim, ids_dim, num_segments
    )
    out = segment_sum_p.bind(flat_data, flat_ids, num_segments=size * num_segments)
    return out.reshape(size, num_segments, *features), 0


def _segment_take_batcher(
    args: tuple[Array, Array], dims: tuple[int | None, int | None]
) -> tuple[Array, int]:
    """Map the gather, an unmapped table needing only a flattened id axis.

    Args:
        args: Batched table and ids.
        dims: Mapped axis of each argument, ``None`` where it is unmapped.

    Returns:
        Tuple of the batched gathered rows and their mapped axis.
    """
    data, segment_ids = args
    data_dim, ids_dim = dims
    if ids_dim is None:
        assert data_dim is not None
        data = jnp.moveaxis(data, data_dim, data.ndim - 1)
        out = segment_take_p.bind(data, segment_ids)
        return out, out.ndim - 1
    if data_dim is None:
        size = segment_ids.shape[ids_dim]
        ids = batching.bdim_at_front(segment_ids, ids_dim, size)
        out = segment_take_p.bind(data, ids.reshape(-1))
        return out.reshape(size, ids.shape[1], *data.shape[1:]), 0
    rows = data.shape[1] if data_dim == 0 else data.shape[0]
    flat_data, flat_ids, size, features = _flatten_batch(
        data, segment_ids, data_dim, ids_dim, rows
    )
    out = segment_take_p.bind(flat_data, flat_ids)
    return out.reshape(size, flat_ids.shape[0] // size, *features), 0


# A ``None`` rule per primitive marks the integral ids as carrying no tangent.
ad.defjvp(segment_sum_p, _segment_sum_jvp, None)
ad.primitive_transposes[segment_sum_p] = _segment_sum_transpose
batching.primitive_batchers[segment_sum_p] = _segment_sum_batcher

ad.defjvp(segment_take_p, _segment_take_jvp, None)
ad.primitive_transposes[segment_take_p] = _segment_take_transpose
batching.primitive_batchers[segment_take_p] = _segment_take_batcher


def segment_sum(
    data: Array,
    segment_ids: Array,
    num_segments: int,
    *,
    mode: Mode | str | None = None,
) -> Array:
    """Sum `data` into `num_segments` bins, accumulating in a wider float.

    Drop-in replacement for ``jax.ops.segment_sum``: the same value in exact
    arithmetic, closer to it in floating point, and the same `mode` for segment
    ids outside ``[0, num_segments)``, dropped by default, negatives included.
    The scatter accumulates one float wider than `data` and converts back once,
    f32 in f64 and f16 or bf16 in f32, so a bin taking ``k`` contributions rounds
    once rather than ``k`` times. Integer data and empty inputs fall through to
    the plain scatter, which is already exact. `segment_ids` may additionally
    cover several leading axes of `data` rather than only the first, which
    ``jax.ops.segment_sum`` rejects.

    Linear in `data` and non-differentiable in the integral `segment_ids`, so it
    transforms under ``jit``, ``vmap`` over either argument, forward mode,
    reverse mode, and repeated differentiation. The reverse pass is
    [segment_take][kups.core.utils.segment.segment_take] under the same `mode`,
    its exact adjoint.

    Args:
        data: Contributions of shape ``(*segment_ids.shape, *feature_dims)``.
        segment_ids: Bin index per contribution, of any shape that is a prefix of
            `data`'s.
        num_segments: Number of output bins.
        mode: How to treat ids outside ``[0, num_segments)``, as a
            ``jax.lax.GatherScatterMode`` or as one of the strings ``"drop"``,
            ``"fill"`` or ``"clip"``. The default ``None`` drops them so they
            contribute nothing, negatives included; ``"clip"`` instead clamps each
            id into range, piling negatives into bin ``0`` and ids at or above
            `num_segments` into the last bin, which is far slower than dropping
            when many ids are out of range because those two bins then take every
            such update. ``"promise_in_bounds"`` and ``"one_hot"`` are rejected.

    Returns:
        Bin sums of shape ``(num_segments, *feature_dims)`` with `data`'s dtype,
        `feature_dims` being the axes of `data` that `segment_ids` does not cover.

    Raises:
        ValueError: If `data` is scalar, `segment_ids` is scalar, not integral, or
            its shape is not a prefix of `data`'s, or `mode` is unknown or
            unsupported.

    Example:
        ```python
        # Per-atom sums of float32 edge messages, accumulated in float64.
        per_atom = segment_sum(messages, edges.indices.indices[:, 0], n_atoms)
        ```
    """
    if data.ndim == 0:
        raise ValueError(f"data must be at least 1-D, got shape {data.shape}.")
    if segment_ids.ndim == 0:
        raise ValueError("segment_ids must be at least 1-D, got shape ().")
    if segment_ids.shape != data.shape[: segment_ids.ndim]:
        raise ValueError(
            f"segment_ids shape {segment_ids.shape} must be a prefix of data's "
            f"{data.shape}."
        )
    if not jnp.issubdtype(segment_ids.dtype, jnp.integer):
        raise ValueError(
            f"segment_ids must be integral, got dtype {segment_ids.dtype}."
        )
    if _resolve_mode(mode) is _CLIP:
        segment_ids = _clip_ids(segment_ids, num_segments)
    if segment_ids.ndim > 1:
        data = data.reshape(-1, *data.shape[segment_ids.ndim :])
        segment_ids = segment_ids.reshape(-1)
    if data.shape[0] == 0 or not jnp.issubdtype(data.dtype, jnp.inexact):
        return jax.ops.segment_sum(data, segment_ids, num_segments, mode="drop")
    return segment_sum_p.bind(data, segment_ids, num_segments=num_segments)


def segment_take(
    data: Array,
    segment_ids: Array,
    *,
    mode: Mode | str | None = None,
    fill_value: ArrayLike | None = None,
) -> Array:
    """Gather rows of `data` by segment id, summing the cotangents stably.

    The forward pass moves rows and is exact; the accuracy is in the reverse
    pass, which sums the cotangents landing in each row through
    [segment_sum][kups.core.utils.segment.segment_sum] rather than the serial
    ``jnp.zeros(...).at[segment_ids].add(...)`` that differentiating
    ``data[segment_ids]`` gives. The two are exact adjoints under the same `mode`
    and a zero `fill_value`, so ``(segment_take(data, ids) * cotangent).sum()``
    equals
    ``(data * segment_sum(cotangent, ids, len(data))).sum()``, and each is the
    other's transpose, so differentiating either repeatedly keeps using the wide
    accumulator instead of falling back to the plain scatter.

    Args:
        data: Rows to gather from, shape ``(num_segments, *feature_dims)``.
        segment_ids: Row index per output, of any shape.
        mode: How to treat ids outside ``[0, num_segments)``, spelled as for
            [segment_sum][kups.core.utils.segment.segment_sum]. A negative id is
            out of range here rather than an index from the end as in
            ``jnp.take``. The default ``None`` gives every out-of-range output
            `fill_value`, so it fills exactly the ids the sum drops; ``"clip"``
            clamps them to the first or last row instead. An empty `data` has no
            row to clamp to, so ``"clip"`` gives every output zero there.
        fill_value: Value the out-of-range outputs take under the default `mode`,
            zero so the gather stays the sum's exact adjoint. Unlike ``jnp.take``
            a `fill_value` of ``None`` means zero rather than NaN. A nonzero fill
            is an additive constant, so the gather becomes affine rather than
            linear: forward and reverse mode stay correct and ignore it, but
            ``jax.linear_transpose`` then transposes only the linear part, as
            ``jnp.take`` does. Clamping leaves nothing out of range, so
            ``mode="clip"`` ignores the fill as ``jnp.take`` does, except that one
            already known to be nonzero at trace time is rejected rather than
            silently dropped.

    Returns:
        Gathered rows of shape ``(*segment_ids.shape, *feature_dims)`` with
        `data`'s dtype.

    Raises:
        ValueError: If `data` or `segment_ids` is scalar, `mode` is unknown or
            unsupported, or `fill_value` is known at trace time to be nonzero
            under ``mode="clip"``.

    Example:
        ```python
        # Per-edge copies of atom features, whose gradient scatters back stably.
        edge_emb = segment_take(node_emb, edges.indices.indices[:, 0])
        ```
    """
    if data.ndim == 0:
        raise ValueError(f"data must be at least 1-D, got shape {data.shape}.")
    if segment_ids.ndim == 0:
        raise ValueError("segment_ids must be at least 1-D, got shape ().")
    fill_value = 0 if fill_value is None else fill_value
    known_fill = _static_fill(fill_value)
    clipped = _resolve_mode(mode) is _CLIP
    if clipped:
        if known_fill is not None and known_fill.any():
            raise ValueError(
                "a nonzero fill_value is meaningless under mode='clip', which "
                "clamps every id into range instead of filling; drop it or pass "
                "mode='fill'."
            )
        segment_ids = _clip_ids(segment_ids, data.shape[0])
    if data.shape[0] == 0 or not jnp.issubdtype(data.dtype, jnp.inexact):
        rows = _gather(data, segment_ids)
    else:
        flat = segment_take_p.bind(data, segment_ids.reshape(-1))
        rows = flat.reshape(*segment_ids.shape, *data.shape[1:])
    # Clamped ids leave nothing out of range, so only the default mode fills.
    if clipped or (known_fill is not None and not known_fill.any()):
        return rows
    return _fill_out_of_range(rows, segment_ids, data.shape[0], fill_value)
