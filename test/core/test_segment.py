# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from kups.core.utils.segment import segment_sum, segment_take

type Case = tuple[Array, Array, int]
type Loss = Callable[[Array], Array]
type ModeArg = jax.lax.GatherScatterMode | str | None

# A negative id, `num_segments` itself, and an id far above it, all dropped.
POISON = jnp.array([0, -1, 2, 1, -7, 9, 4], jnp.int32)

# Every spelling of the default policy, and of the clamping one.
DROP_SPELLINGS: list[ModeArg] = [
    None,
    "fill",
    "drop",
    jax.lax.GatherScatterMode.FILL_OR_DROP,
]
CLIP_SPELLINGS: list[ModeArg] = ["clip", jax.lax.GatherScatterMode.CLIP]

# The two honoured policies, as both functions and `_reference` spell them.
MODES = ["drop", "clip"]

# Rejected modes, each mapped to the phrase its error must explain itself by.
BAD_MODES = {
    "promise_in_bounds": r"promise_in_bounds.*adjoints",
    "one_hot": r"one_hot.*no gather lowering",
    "wrap": r'Unknown gather mode "wrap"',
}


@pytest.fixture(autouse=True)
def _disable_x64() -> Iterator[None]:
    """Force float32 for this module without leaking the global flag."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _lognormal(features: tuple[int, ...]) -> Case:
    """Positive contributions spanning decades, ids that include drops, eight bins."""
    keys = jax.random.split(jax.random.key(0))
    data = jnp.exp(3 * jax.random.normal(keys[0], (257, *features), jnp.float32))
    return data, jax.random.randint(keys[1], (257,), -1, 9), 8


def _reference(
    data: ArrayLike, segment_ids: ArrayLike, num_segments: int, mode: str = "drop"
) -> np.ndarray:
    """Bin sums accumulated in float64 numpy, ids outside the range following `mode`.

    Args:
        data: Contributions, binned along their leading axis.
        segment_ids: Bin index per contribution.
        num_segments: Number of output bins.
        mode: ``"clip"`` to clamp out-of-range ids into range, anything else to drop
            them.

    Returns:
        Bin sums of shape ``(num_segments, *data.shape[1:])`` in float64.
    """
    values = np.asarray(data, np.float64)
    ids = np.asarray(segment_ids)
    out = np.zeros((num_segments, *values.shape[1:]), np.float64)
    if mode == "clip":
        ids = np.clip(ids, 0, num_segments - 1)
    keep = (ids >= 0) & (ids < num_segments)
    np.add.at(out, ids[keep], values[keep])
    return out


def _masked_take(
    data: Array, segment_ids: Array, mode: str = "drop", fill_value: ArrayLike = 0
) -> Array:
    """Gather following `mode`, built from `jnp.take` and differentiable.

    Args:
        data: Rows to gather from.
        segment_ids: Row index per output.
        mode: ``"clip"`` to clamp out-of-range ids into range, anything else to give
            them `fill_value`.
        fill_value: Value the out-of-range outputs take when they are not clamped.

    Returns:
        Gathered rows of shape ``(*segment_ids.shape, *data.shape[1:])``.
    """
    if mode == "clip":
        return jnp.take(data, jnp.clip(segment_ids, 0, data.shape[0] - 1), axis=0)
    keep = (segment_ids >= 0) & (segment_ids < data.shape[0])
    rows = jnp.take(data, jnp.where(keep, segment_ids, 0), axis=0)
    return jnp.where(
        keep.reshape(*keep.shape, *(1,) * (data.ndim - 1)), rows, fill_value
    )


def _mapped(rows: Array, axis: int | None) -> Array:
    """`rows`' leading axis moved to `axis`, or its first row when unmapped.

    Args:
        rows: Stack of per-batch-row arguments.
        axis: Axis the mapped argument is to be passed along, ``None`` to pass a
            single unmapped row.

    Returns:
        The argument as `vmap` at `in_axes=axis` expects it.
    """
    return rows[0] if axis is None else jnp.moveaxis(rows, 0, axis)


def _hvp(loss: Loss, tangent: Array) -> Loss:
    """Hessian-vector product of `loss` along `tangent`, reverse over reverse."""

    def directional(x: Array) -> Array:
        return jnp.sum(jax.grad(loss)(x) * tangent)

    return jax.grad(directional)


def _max_ulp_error(out: ArrayLike, exact: np.ndarray) -> float:
    """Largest deviation from `exact` in ulp of its float32 rounding."""
    ulp = np.maximum(
        np.spacing(np.asarray(exact, np.float32)).astype(np.float64), 1e-38
    )
    return float(np.max(np.abs(np.asarray(out, np.float64) - exact) / ulp))


class TestSegmentSum:
    @pytest.mark.parametrize("features", [(), (3,), (2, 2)])
    def test_matches_float64_reference(self, features: tuple[int, ...]) -> None:
        """Rounds the float64 sum once, identically eager and under `jit`."""
        data, ids, num_segments = _lognormal(features)
        out = segment_sum(data, ids, num_segments)
        assert out.dtype == jnp.float32
        assert _max_ulp_error(out, _reference(data, ids, num_segments)) <= 0.5
        jitted = jax.jit(segment_sum, static_argnums=2)(data, ids, num_segments)
        npt.assert_array_equal(out, jitted)

    def test_out_of_range_dropped(self) -> None:
        """Negative ids, `num_segments` itself, and larger ids contribute nothing."""
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)
        for mode in DROP_SPELLINGS:
            out = segment_sum(data, POISON, 4, mode=mode)
            npt.assert_array_equal(out, [1.0, 4.0, 3.0, 0.0])
            npt.assert_array_equal(out, jax.ops.segment_sum(data, POISON, 4))

    def test_clip_clamps_out_of_range_ids(self) -> None:
        """Clip piles negatives into bin zero and ids past the range into the last."""
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)
        for mode in CLIP_SPELLINGS:
            out = segment_sum(data, POISON, 4, mode=mode)
            npt.assert_array_equal(out, [8.0, 4.0, 3.0, 13.0])
            npt.assert_array_equal(
                out, jax.ops.segment_sum(data, POISON, 4, mode="clip")
            )
        # With no bin to clamp to the ids are dropped and nothing is observable.
        npt.assert_array_equal(
            segment_sum(data, POISON, 0, mode="clip"), jnp.zeros((0,))
        )

    @pytest.mark.parametrize(
        ("dtype", "spike", "addends"),
        [
            (jnp.float16, 4096.0, 4),
            (jnp.bfloat16, 4096.0, 32),
            (jnp.float32, 1e8, 8),
            (jnp.complex64, 1e8, 8),
        ],
    )
    def test_accumulates_in_a_wider_float(
        self, dtype: DTypeLike, spike: float, addends: int
    ) -> None:
        """A cancelling spike leaves the addends that `data`'s own dtype rounds away."""
        data = jnp.asarray([spike, *[1.0] * addends, -spike], dtype)
        ids = jnp.zeros(addends + 2, jnp.int32)
        out = segment_sum(data, ids, 1)
        assert out.dtype == dtype
        npt.assert_array_equal(out, jnp.asarray([addends], dtype))
        # Accumulating in `dtype` rounds the addends away against the spike; how much
        # of them survives is a backend detail, but none of it is the exact sum.
        assert np.asarray(jax.ops.segment_sum(data, ids, 1))[0] != addends

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("drop", [10.0, 0.0, 1000.0, 100.0, 0.0, 0.0, 0.0]),
            ("clip", [10.0, 10.0, 1000.0, 100.0, 10.0, 10000.0, 10000.0]),
        ],
    )
    def test_grad_is_gather(self, mode: str, expected: list[float]) -> None:
        """Reverse mode gathers the cotangent under the same `mode`."""
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)
        cotangent = jnp.array([10.0, 100.0, 1000.0, 10000.0], jnp.float32)

        def loss(x: Array) -> Array:
            return jnp.sum(segment_sum(x, POISON, 4, mode=mode) * cotangent)

        grad = jax.grad(loss)(data)
        npt.assert_array_equal(grad, expected)
        npt.assert_array_equal(grad, _masked_take(cotangent, POISON, mode))
        npt.assert_array_equal(grad, segment_take(cotangent, POISON, mode=mode))

    @pytest.mark.parametrize("mode", MODES)
    def test_forward_mode_and_transpose(self, mode: str) -> None:
        """Forward mode pushes the tangent through and the transpose is the gather."""
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)
        tangent = jnp.arange(8.0, 15.0, dtype=jnp.float32)
        summed = partial(segment_sum, segment_ids=POISON, num_segments=4, mode=mode)
        primal, out_tangent = jax.jvp(summed, (data,), (tangent,))
        npt.assert_array_equal(primal, segment_sum(data, POISON, 4, mode=mode))
        npt.assert_array_equal(out_tangent, segment_sum(tangent, POISON, 4, mode=mode))
        cotangent = jnp.arange(1.0, 5.0, dtype=jnp.float32)
        (transposed,) = jax.linear_transpose(summed, data)(cotangent)
        npt.assert_array_equal(transposed, _masked_take(cotangent, POISON, mode))

    @pytest.mark.parametrize("mode", MODES)
    def test_second_order(self, mode: str) -> None:
        """Forward-over-reverse and reverse-over-reverse match the plain scatter."""
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)
        tangent = jnp.ones(7, jnp.float32)

        def ours(x: Array) -> Array:
            return jnp.sum(segment_sum(x, POISON, 4, mode=mode) ** 3)

        def ref(x: Array) -> Array:
            return jnp.sum(jax.ops.segment_sum(x, POISON, 4, mode=mode) ** 3)

        npt.assert_allclose(jax.grad(ours)(data), jax.grad(ref)(data), rtol=1e-6)
        npt.assert_allclose(
            jax.jvp(jax.grad(ours), (data,), (tangent,))[1],
            jax.jvp(jax.grad(ref), (data,), (tangent,))[1],
            rtol=1e-6,
        )
        npt.assert_allclose(
            _hvp(ours, tangent)(data), _hvp(ref, tangent)(data), rtol=1e-6
        )

    def test_second_order_stays_on_the_wide_primitives(self) -> None:
        """Repeated differentiation never falls back to a plain scatter."""
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)

        def loss(x: Array) -> Array:
            return jnp.sum(segment_sum(x, POISON, 4) ** 3)

        jaxpr = jax.make_jaxpr(_hvp(loss, data))(data)
        primitives = {str(eqn.primitive) for eqn in jaxpr.jaxpr.eqns}
        assert not any("scatter" in name for name in primitives)
        assert {"kups_segment_sum", "kups_segment_take"} <= primitives

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("in_axes", [(0, None), (None, 0), (0, 0), (1, 1)])
    def test_jit_and_vmap(
        self, in_axes: tuple[int | None, int | None], mode: str
    ) -> None:
        """Maps over contributions, ids, or both, each row resolving its own ids."""
        data = jnp.arange(1.0, 15.0, dtype=jnp.float32).reshape(2, 7)
        ids = jnp.stack([POISON, jnp.full((7,), -1, jnp.int32)])
        args = (_mapped(data, in_axes[0]), _mapped(ids, in_axes[1]))
        rows = [
            (
                data[b if in_axes[0] is not None else 0],
                ids[b if in_axes[1] is not None else 0],
            )
            for b in range(2)
        ]
        summed = partial(segment_sum, num_segments=4, mode=mode)
        out = jax.jit(jax.vmap(summed, in_axes=in_axes))(*args)
        npt.assert_array_equal(out, [_reference(d, i, 4, mode) for d, i in rows])

    @pytest.mark.parametrize("mode", MODES)
    def test_nested_vmap(self, mode: str) -> None:
        """Two mapped axes keep every innermost segmentation to its own bins."""
        data = jnp.arange(1.0, 43.0, dtype=jnp.float32).reshape(2, 3, 7)
        inner = jnp.stack([POISON, jnp.roll(POISON, 2), jnp.full((7,), -1, jnp.int32)])
        ids = jnp.stack([inner, inner[::-1]])
        summed = partial(segment_sum, num_segments=4, mode=mode)
        out = jax.vmap(jax.vmap(summed))(data, ids)
        npt.assert_array_equal(
            out,
            [
                [_reference(d, i, 4, mode) for d, i in zip(rows, row_ids)]
                for rows, row_ids in zip(data, ids)
            ],
        )

    @pytest.mark.parametrize("mode", MODES)
    def test_vmap_of_grad(self, mode: str) -> None:
        """Differentiating under a mapped segmentation gathers per row."""
        data = jnp.arange(1.0, 15.0, dtype=jnp.float32).reshape(2, 7)
        ids = jnp.stack([POISON, POISON + 1])
        cotangent = jnp.array([10.0, 100.0, 1000.0, 10000.0], jnp.float32)

        def loss(x: Array, segment_ids: Array) -> Array:
            return jnp.sum(segment_sum(x, segment_ids, 4, mode=mode) * cotangent)

        npt.assert_array_equal(
            jax.vmap(jax.grad(loss))(data, ids),
            jax.vmap(partial(_masked_take, mode=mode), in_axes=(None, 0))(
                cotangent, ids
            ),
        )

    def test_integer_data_falls_through(self) -> None:
        """Integer contributions defer to the already exact plain scatter."""
        data = jnp.arange(1, 8, dtype=jnp.int32)
        out = segment_sum(data, POISON, 4)
        assert out.dtype == jnp.int32
        npt.assert_array_equal(out, [1, 4, 3, 0])
        npt.assert_array_equal(out, jax.ops.segment_sum(data, POISON, 4, mode="drop"))
        npt.assert_array_equal(
            segment_sum(data, POISON, 4, mode="clip"),
            jax.ops.segment_sum(data, POISON, 4, mode="clip"),
        )

    def test_empty_and_degenerate_shapes(self) -> None:
        """No contributions, no segments, and zero-width features stay well shaped."""
        empty = jnp.zeros((0, 2), jnp.float32)
        npt.assert_array_equal(
            segment_sum(empty, jnp.zeros((0,), jnp.int32), 3), jnp.zeros((3, 2))
        )
        npt.assert_array_equal(
            segment_sum(empty, jnp.zeros((0,), jnp.int32), 3, mode="clip"),
            jnp.zeros((3, 2)),
        )
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)
        npt.assert_array_equal(segment_sum(data, POISON, 0), jnp.zeros((0,)))

        def binless(x: Array) -> Array:
            return jnp.sum(segment_sum(x, POISON, 0))

        npt.assert_array_equal(jax.grad(binless)(data), jnp.zeros(7))
        npt.assert_array_equal(
            segment_sum(jnp.zeros((7, 0), jnp.float32), POISON, 4), jnp.zeros((4, 0))
        )

    def test_narrow_index_dtype(self) -> None:
        """int8 ids widen for the sentinel, the batched offsets, and the clip bound."""
        ids = jnp.array([0, 1, 1, 2, 2, 2], jnp.int8)
        data = jnp.arange(6.0, dtype=jnp.float32)
        npt.assert_array_equal(segment_sum(data, ids, 200)[:3], [0.0, 3.0, 12.0])
        npt.assert_array_equal(
            segment_sum(data, ids, 200, mode="clip")[:3], [0.0, 3.0, 12.0]
        )
        mapped = jax.vmap(partial(segment_sum, num_segments=100), in_axes=(None, 0))(
            data, jnp.stack([ids, ids])
        )
        npt.assert_array_equal(mapped[:, :3], [[0.0, 3.0, 12.0]] * 2)

    def test_no_wide_dtype_reaches_an_aval(self) -> None:
        """The accumulator lives in the lowering, never in a jaxpr."""
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)

        def loss(x: Array) -> Array:
            return jnp.sum(segment_sum(x, POISON, 4) ** 2)

        for jaxpr in (
            jax.make_jaxpr(partial(segment_sum, num_segments=4))(data, POISON),
            jax.make_jaxpr(partial(segment_sum, num_segments=4, mode="clip"))(
                data, POISON
            ),
            jax.make_jaxpr(jax.grad(loss))(data),
            jax.make_jaxpr(jax.hessian(loss))(data),
        ):
            dtypes = [
                np.dtype(dtype)
                for eqn in jaxpr.jaxpr.eqns
                for var in (*eqn.invars, *eqn.outvars)
                if (dtype := getattr(var.aval, "dtype", None)) is not None
            ]
            assert all(dtype.itemsize <= 4 for dtype in dtypes)

    def test_clip_bounds_stay_in_the_id_dtype(self) -> None:
        """The clamp bounds never widen the ids, which x64 makes observable."""
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)
        jax.config.update("jax_enable_x64", True)
        try:
            clipped = partial(segment_sum, num_segments=4, mode="clip")
            jaxpr = str(jax.make_jaxpr(clipped)(data, jnp.asarray(POISON, jnp.int32)))
        finally:
            jax.config.update("jax_enable_x64", False)
        assert "i64" not in jaxpr

    def test_lowering_scatters_in_the_wide_accumulator(self) -> None:
        """The emitted scatter runs in f64 even though x64 is off."""
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)
        traced = jax.jit(partial(segment_sum, num_segments=4)).trace(data, POISON)
        text = traced.lower(lowering_platforms=("cpu",)).as_text()
        assert "scatter" in text
        assert re.search(r"\bf64\b", text)

    def test_rejects_bad_arguments(self) -> None:
        """Bad shapes or id dtypes, unsupported modes, and a fill value raise."""
        with pytest.raises(ValueError, match="1-D"):
            segment_sum(jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32), 4)
        with pytest.raises(ValueError, match="1-D"):
            segment_sum(jnp.zeros((4,), jnp.float32), jnp.asarray(0, jnp.int32), 4)
        with pytest.raises(ValueError, match="prefix"):
            segment_sum(jnp.zeros((4,), jnp.float32), jnp.zeros((3,), jnp.int32), 4)
        with pytest.raises(ValueError, match="prefix"):
            segment_sum(jnp.zeros((4, 3), jnp.float32), jnp.zeros((3,), jnp.int32), 4)
        data = jnp.arange(1.0, 8.0, dtype=jnp.float32)
        with pytest.raises(ValueError, match="segment_ids must be integral"):
            segment_sum(data, POISON.astype(jnp.float32), 4)
        for mode, message in BAD_MODES.items():
            with pytest.raises(ValueError, match=message):
                segment_sum(data, POISON, 4, mode=mode)
        # A dropped update writes nothing, so there is no slot to fill and the sum
        # takes no `fill_value`, no more than `jax.ops.segment_sum` does.
        with pytest.raises(TypeError, match="fill_value"):
            # pyrefly: ignore [unexpected-keyword]
            segment_sum(data, POISON, 4, fill_value=1.0)

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("features", [(), (3,)])
    def test_ids_covering_several_leading_axes(
        self, features: tuple[int, ...], mode: str
    ) -> None:
        """Ids of any prefix shape bin every axis they cover, under either `mode`."""
        data, ids, num_segments = _lognormal(features)
        data, ids = data[:256], ids[:256]
        shaped = ids.reshape(32, 8)
        out = segment_sum(
            data.reshape(32, 8, *features), shaped, num_segments, mode=mode
        )
        npt.assert_array_equal(out, segment_sum(data, ids, num_segments, mode=mode))
        assert _max_ulp_error(out, _reference(data, ids, num_segments, mode)) < 1.5


class TestSegmentTake:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("features", [(), (2, 3)])
    def test_adjoint_identity(self, features: tuple[int, ...], mode: str) -> None:
        """Contracting the gather against a cotangent equals contracting the sum."""
        rng = np.random.default_rng(0)
        data = jnp.asarray(rng.integers(-8, 8, (7, *features)), jnp.float32)
        cotangent = jnp.asarray(rng.integers(-8, 8, (4, *features)), jnp.float32)
        npt.assert_array_equal(
            jnp.sum(segment_sum(data, POISON, 4, mode=mode) * cotangent),
            jnp.sum(data * segment_take(cotangent, POISON, mode=mode)),
        )

    def test_gathers_and_drops(self) -> None:
        """Rows come through unchanged; ids outside the table gather zero."""
        table = jnp.arange(1.0, 5.0, dtype=jnp.float32)
        for mode in DROP_SPELLINGS:
            out = segment_take(table, POISON, mode=mode)
            assert out.dtype == jnp.float32
            npt.assert_array_equal(out, [1.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0])
            npt.assert_array_equal(out, _masked_take(table, POISON))
        # `jnp.take` agrees on every id it does not read from the end of the table.
        ids = jnp.array([0, 4, 2, 1, 9, 3], jnp.int32)
        npt.assert_array_equal(
            segment_take(table, ids), jnp.take(table, ids, mode="fill", fill_value=0.0)
        )
        npt.assert_array_equal(
            segment_take(jnp.zeros((0, 2), jnp.float32), POISON), jnp.zeros((7, 2))
        )

    def test_clip_gathers_the_clamped_row(self) -> None:
        """Clip clamps each id to the first or last row rather than filling."""
        table = jnp.arange(1.0, 5.0, dtype=jnp.float32)
        clamped = jnp.take(table, POISON, mode="clip")
        npt.assert_array_equal(clamped, [1.0, 1.0, 3.0, 2.0, 1.0, 4.0, 4.0])
        for mode in CLIP_SPELLINGS:
            npt.assert_array_equal(segment_take(table, POISON, mode=mode), clamped)
        # An empty table has no row to clamp to, so every output is zero.
        npt.assert_array_equal(
            segment_take(jnp.zeros((0, 2), jnp.float32), POISON, mode="clip"),
            jnp.zeros((7, 2)),
        )
        # A zero fill changes nothing, whether trace time knows its value or not.
        npt.assert_array_equal(
            segment_take(table, POISON, mode="clip", fill_value=0), clamped
        )
        clipped = jax.jit(partial(segment_take, segment_ids=POISON, mode="clip"))
        npt.assert_array_equal(clipped(table, fill_value=jnp.float32(0.0)), clamped)

    def test_grad_is_the_wide_sum(self) -> None:
        """Reverse mode is `segment_sum` bit for bit, keeping its accuracy."""
        data = jnp.asarray([1e8, *[1.0] * 8, -1e8], jnp.float32)
        ids = jnp.zeros(10, jnp.int32)

        def loss(table: Array) -> Array:
            return jnp.sum(segment_take(table, ids) * data)

        grad = jax.grad(loss)(jnp.zeros(1, jnp.float32))
        npt.assert_array_equal(grad, segment_sum(data, ids, 1))
        npt.assert_array_equal(grad, [8.0])

    @pytest.mark.parametrize("mode", MODES)
    def test_forward_mode_and_second_order(self, mode: str) -> None:
        """jvp, grad, and reverse-over-reverse match the masked `jnp.take`."""
        table = jnp.arange(1.0, 5.0, dtype=jnp.float32)
        tangent = jnp.arange(4.0, 8.0, dtype=jnp.float32)

        def gathered(t: Array) -> Array:
            return segment_take(t, POISON, mode=mode)

        npt.assert_array_equal(
            jax.jvp(gathered, (table,), (tangent,))[1],
            segment_take(tangent, POISON, mode=mode),
        )

        def ours(t: Array) -> Array:
            return jnp.sum(segment_take(t, POISON, mode=mode) ** 3)

        def ref(t: Array) -> Array:
            return jnp.sum(_masked_take(t, POISON, mode) ** 3)

        npt.assert_allclose(jax.grad(ours)(table), jax.grad(ref)(table), rtol=1e-6)
        npt.assert_allclose(
            _hvp(ours, tangent)(table), _hvp(ref, tangent)(table), rtol=1e-6
        )

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("in_axes", [(0, None), (None, 0), (0, 0), (1, 1)])
    def test_jit_and_vmap(
        self, in_axes: tuple[int | None, int | None], mode: str
    ) -> None:
        """Maps over the table, the ids, or both, each row resolving its own ids."""
        tables = jnp.arange(1.0, 9.0, dtype=jnp.float32).reshape(2, 4)
        rows = jnp.stack([POISON, jnp.roll(POISON, 2)])
        args = (_mapped(tables, in_axes[0]), _mapped(rows, in_axes[1]))
        npt.assert_array_equal(
            jax.jit(jax.vmap(partial(segment_take, mode=mode), in_axes=in_axes))(*args),
            jax.vmap(partial(_masked_take, mode=mode), in_axes=in_axes)(*args),
        )

    @pytest.mark.parametrize("mode", MODES)
    def test_nested_vmap(self, mode: str) -> None:
        """Two mapped axes gather each innermost row from its own table."""
        tables = jnp.arange(1.0, 25.0, dtype=jnp.float32).reshape(2, 3, 4)
        ids = jnp.stack([jnp.stack([POISON, jnp.roll(POISON, 2), -POISON])] * 2)
        gathered = jax.vmap(jax.vmap(partial(segment_take, mode=mode)))(tables, ids)
        npt.assert_array_equal(
            gathered,
            jax.vmap(jax.vmap(partial(_masked_take, mode=mode)))(tables, ids),
        )

    def test_take_accepts_ids_of_higher_rank(self) -> None:
        """The gather shapes its output to the ids and stays the sum's adjoint."""
        table = jnp.exp(jax.random.normal(jax.random.key(1), (8, 3), jnp.float32))
        ids = jnp.stack([POISON, POISON[::-1]])
        rows = segment_take(table, ids)
        assert rows.shape == (*ids.shape, 3)
        npt.assert_array_equal(
            rows, _masked_take(table, ids.reshape(-1)).reshape(rows.shape)
        )
        cotangent = jax.random.normal(jax.random.key(2), rows.shape, jnp.float32)
        (gradient,) = jax.vjp(partial(segment_take, segment_ids=ids), table)[1](
            cotangent
        )
        npt.assert_allclose(
            gradient, segment_sum(cotangent, ids, table.shape[0]), rtol=1e-6
        )

    def test_fill_value_replaces_dropped_rows(self) -> None:
        """Dropped ids take `fill_value`, `None` meaning zero, and grad ignores it."""
        table = jnp.arange(1.0, 5.0, dtype=jnp.float32)
        out = segment_take(table, POISON, fill_value=99.0)
        npt.assert_array_equal(out, [1.0, 99.0, 3.0, 2.0, 99.0, 99.0, 99.0])
        npt.assert_array_equal(out, _masked_take(table, POISON, fill_value=99.0))
        npt.assert_array_equal(
            segment_take(table, POISON, fill_value=None), segment_take(table, POISON)
        )
        npt.assert_array_equal(
            segment_take(jnp.zeros((0, 2), jnp.float32), POISON, fill_value=7.0),
            jnp.full((7, 2), 7.0),
        )

        def loss(t: Array, fill: float) -> Array:
            return jnp.sum(segment_take(t, POISON, fill_value=fill) ** 2)

        npt.assert_array_equal(jax.grad(loss)(table, 99.0), jax.grad(loss)(table, 0.0))

    def test_jvp_fills_the_tangent_with_zero(self) -> None:
        """A nonzero fill is a constant, so the tangent still fills with zero."""
        table = jnp.arange(1.0, 5.0, dtype=jnp.float32)
        tangent = jnp.arange(4.0, 8.0, dtype=jnp.float32)
        gathered = partial(segment_take, segment_ids=POISON, fill_value=99.0)
        primal, out_tangent = jax.jvp(gathered, (table,), (tangent,))
        npt.assert_array_equal(primal, segment_take(table, POISON, fill_value=99.0))
        npt.assert_array_equal(out_tangent, [4.0, 0.0, 6.0, 5.0, 0.0, 0.0, 0.0])
        npt.assert_array_equal(out_tangent, segment_take(tangent, POISON))

    def test_fall_throughs_honour_mode(self) -> None:
        """The integer gather clamps and fills exactly as the primitive path does."""
        table = jnp.arange(1, 5, dtype=jnp.int32)
        npt.assert_array_equal(
            segment_take(table, POISON, mode="clip"),
            jnp.take(table, POISON, mode="clip"),
        )
        npt.assert_array_equal(
            segment_take(table, POISON, fill_value=7), [1, 7, 3, 2, 7, 7, 7]
        )

    def test_rejects_bad_arguments(self) -> None:
        """Scalar tables, scalar ids, unsupported modes, and a nonzero clipped fill."""
        table = jnp.arange(4, dtype=jnp.float32)
        with pytest.raises(ValueError, match="1-D"):
            segment_take(jnp.zeros((), jnp.float32), POISON)
        with pytest.raises(ValueError, match="segment_ids"):
            segment_take(jnp.zeros((4,), jnp.float32), jnp.asarray(0, jnp.int32))
        for mode, message in BAD_MODES.items():
            with pytest.raises(ValueError, match=message):
                segment_take(table, POISON, mode=mode)
        with pytest.raises(ValueError, match="fill_value.*clip"):
            segment_take(table, POISON, mode="clip", fill_value=9.0)
