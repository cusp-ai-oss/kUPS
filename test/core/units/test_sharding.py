# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Device placement and sharding of quantities."""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from kups.core import units
from kups.core.units import Length
from kups.core.units import catalog as u

from ._helpers import arr, ty, val

MESH = jax.make_mesh((1,), ("x",))
SHARDED = NamedSharding(MESH, P("x"))
# for with_sharding_constraint; the compiler still chooses the layout
AUTO_SHARDED = NamedSharding(Mesh(jax.devices()[:1], ("x",)), P("x"))


def spec(x: Length | jax.Array) -> P:
    """Return the PartitionSpec of a value's NamedSharding."""
    sh = x.sharding
    assert isinstance(sh, NamedSharding), sh
    return sh.spec


def test_placement_keeps_label():
    q = arr([1.0, 2.0, 3.0]) * u.angstrom
    placed = jax.device_put(q, SHARDED)
    assert units.is_quantity(placed)
    assert spec(placed) == P("x")
    assert units.dim_of(placed) == units.dim_of(u.m)
    assert units.base_of(placed) is units.base_of(q)
    assert val(placed[1]) == val(q[1])
    assert placed.sharding == placed.value.sharding  # from the magnitudes
    # mixed pytrees place leaf by leaf
    tree = jax.device_put({"r": q, "n": arr([1.0, 2.0, 3.0])}, SHARDED)
    assert units.dim_of(tree["r"]) == units.dim_of(u.m)
    assert not units.is_quantity(tree["n"])
    # device_get returns numpy magnitudes
    assert isinstance(jax.device_get(placed).v, np.ndarray)
    assert spec(jax.device_put(arr([1.0]), SHARDED)) == P("x")
    out = jax.jit(lambda x: x * 2.0, in_shardings=SHARDED, out_shardings=SHARDED)(q)
    assert ty(out) == "float64[3]{Å|atomistic}" and spec(out) == P("x")
    held = jax.lax.with_sharding_constraint(q, AUTO_SHARDED)
    assert units.dim_of(held) == units.dim_of(u.m) and spec(held) == P("x")


def test_shard_map_and_pmap():
    r = jax.device_put(arr([1.0, 2.0]) * u.angstrom, SHARDED)
    seen: list[bool] = []

    def double(x: Length) -> Length:
        seen.append(units.is_quantity(x))
        return x * 2.0

    out = jax.shard_map(double, mesh=MESH, in_specs=P("x"), out_specs=P("x"))(r)
    assert seen == [True]
    assert units.dim_of(out) == units.dim_of(u.m)
    assert float(out[0].to(u.angstrom)) == pytest.approx(2.0)
    # the decorator form takes the same keywords
    mapped = jax.shard_map(mesh=MESH, in_specs=P("x"), out_specs=P("x"))(double)
    assert units.dim_of(mapped(r)) == units.dim_of(u.m)
    assert units.dim_of(jax.pmap(double)(arr([[1.0]]) * u.angstrom)) == units.dim_of(
        u.m
    )


def test_placement_is_jaxs_own():
    moved = jax.device_put(arr([1.0, 2.0]) * u.angstrom, SHARDED)
    assert val(jax.device_get(moved)[0]) == 1.0


def test_jit_shardings_take_quantities():
    r = jax.device_put(arr([1.0, 2.0, 3.0]) * u.angstrom, SHARDED)
    k = 2.0 * u.eV / u.angstrom**2

    def energy(x: units.Length) -> units.Energy:
        return 0.5 * k * x**2

    out = jax.jit(energy, in_shardings=SHARDED, out_shardings=SHARDED)(r)
    assert spec(out) == P("x")
    assert units.dim_of(out) == units.dim_of(u.J)
    assert float(out[1].to(u.eV)) == pytest.approx(4.0)
    # placement does not weaken the dimension check
    with pytest.raises(units.DimensionError):
        out.to(u.m)
