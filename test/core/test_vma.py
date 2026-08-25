# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Gates for the vma shim: custom-primitive avals keep their varying manual axes.

Without the shim an output aval would drop its axes, be typed as replicated,
and ``shard_map(check_vma=True)`` would reject the program (or the transpose's
cotangent would mismatch a varying primal). Runs on the simulated 4-device CPU
mesh from the pytest ``XLA_FLAGS`` env.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.assertion import runtime_assert
from kups.core.result import as_result_function
from kups.core.utils.segment import segment_sum
from kups.core.utils.vma import out_aval

_VARYING = jax.sharding.PartitionSpec("X")
_REPL = jax.sharding.PartitionSpec()


def _mesh() -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.array(jax.devices()), axis_names=("X",))


def test_out_aval_is_plain_outside_manual_region() -> None:
    data = jax.core.ShapedArray((6, 3), jnp.float64)
    ids = jax.core.ShapedArray((6,), jnp.int32)
    out = out_aval("test_prim", (4, 3), data, ids)
    assert out.shape == (4, 3) and out.dtype == data.dtype


def test_segment_sum_grad_under_shard_map() -> None:
    """Without the vma rules the segment output avals are typed replicated and
    differentiating through ``shard_map`` raises a cotangent-type mismatch —
    the failure the domain-decomposed potentials hit."""
    if len(jax.devices()) < 2:
        pytest.skip("needs a multi-device mesh")
    ids = jnp.array([0, 1, 0, 1])

    def per_device(x: jax.Array) -> jax.Array:
        return jax.lax.psum(segment_sum(x * x, ids, num_segments=2).sum(), "X")

    f = jax.shard_map(per_device, mesh=_mesh(), in_specs=(_VARYING,), out_specs=_REPL)
    x = jnp.arange(float(4 * len(jax.devices())))
    total, grad = jax.value_and_grad(f)(x)
    npt.assert_allclose(total, (x * x).sum())
    npt.assert_allclose(grad, 2 * x)


def test_assertion_primitive_keeps_vma_under_shard_map() -> None:
    if len(jax.devices()) < 2:
        pytest.skip("needs a multi-device mesh")

    def per_device(x: jax.Array) -> jax.Array:
        # Assertion values must be device-invariant to leave the manual
        # region through the replicated assertion context, so mesh-reduce.
        bound = jax.lax.pmax(x.max(), "X")
        runtime_assert(
            bound < 100.0, "x must stay below 100, got {bound}", {"bound": bound}
        )
        return jax.lax.psum(x.sum(), "X")

    f = jax.shard_map(per_device, mesh=_mesh(), in_specs=(_VARYING,), out_specs=_REPL)
    result = as_result_function(f)(jnp.arange(8.0))
    result.raise_assertion()
    assert len(result.assertions) == 1
    assert jnp.allclose(result.value, jnp.arange(8.0).sum())


def test_varying_assertion_is_rejected_by_replicated_context() -> None:
    """A device-varying assertion value must not silently leave the manual region.

    Pins the contract the domain-decomposition capacities rely on: a
    per-device predicate is typed varying and rejected at trace time, instead
    of one device's verdict standing in for the mesh — enforced by the noop
    primitive's pass-through avals, which is why ``_scalar_bool`` needs no
    vma copy of its own.
    """
    if len(jax.devices()) < 2:
        pytest.skip("needs a multi-device mesh")

    def per_device(x: jax.Array) -> jax.Array:
        runtime_assert(x.max() < 100.0, "per-device bound {m}", {"m": x.max()})
        return jax.lax.psum(x.sum(), "X")

    f = jax.shard_map(per_device, mesh=_mesh(), in_specs=(_VARYING,), out_specs=_REPL)
    with pytest.raises(Exception, match="vary|replicat"):
        as_result_function(f)(jnp.arange(8.0))
