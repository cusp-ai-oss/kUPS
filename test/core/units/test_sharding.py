# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Device placement and sharding of quantities."""

from typing import Any

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from kups.core import units
from kups.core.units import catalog as u

from ._helpers import arange, arr, ones, ty, val

MESH = jax.make_mesh((1,), ("x",))
SHARDED = NamedSharding(MESH, P("x"))
# for with_sharding_constraint; the compiler still chooses the layout
AUTO_SHARDED = NamedSharding(Mesh(jax.devices()[:1], ("x",)), P("x"))


def spec(x: Any) -> Any:
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


def test_jit_shardings_take_quantities():
    r = jax.device_put(arr([1.0, 2.0, 3.0]) * u.angstrom, SHARDED)
    k = 2.0 * u.eV / u.angstrom**2

    def energy(x: Any) -> Any:
        return 0.5 * k * x**2

    out = jax.jit(energy, in_shardings=SHARDED, out_shardings=SHARDED)(r)
    assert spec(out) == P("x")
    assert units.dim_of(out) == units.dim_of(u.J)
    assert float(out[1].to(u.eV)) == pytest.approx(4.0)
    # placement does not weaken the dimension check
    with pytest.raises(units.DimensionError):
        out.to(u.m)
    # the gradient is still labelled dim(out)/dim(arg)
    force = jax.jit(units.grad(lambda x: energy(x).sum()))(r)
    assert units.dim_of(force) == units.dim_of(u.eV / u.angstrom)


def test_shard_map_and_pmap():
    r = jax.device_put(arr([1.0, 2.0]) * u.angstrom, SHARDED)
    seen: list[Any] = []

    def double(x: Any) -> Any:
        seen.append(units.is_quantity(x))
        return x * 2.0

    out: Any = jax.shard_map(double, mesh=MESH, in_specs=P("x"), out_specs=P("x"))(r)
    assert seen == [True]
    assert units.dim_of(out) == units.dim_of(u.m)
    assert float(out[0].to(u.angstrom)) == pytest.approx(2.0)
    # the decorator form takes the same keywords
    mapped = jax.shard_map(mesh=MESH, in_specs=P("x"), out_specs=P("x"))(double)
    assert units.dim_of(mapped(r)) == units.dim_of(u.m)
    assert units.dim_of(jax.pmap(double)(arr([[1.0]]) * u.angstrom)) == units.dim_of(
        u.m
    )


needs_four = pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices")


def mesh4() -> tuple[Mesh, NamedSharding]:
    """A 4-device mesh and the sharding that splits axis 0 over it."""
    mesh = jax.make_mesh((4,), ("x",))
    return mesh, NamedSharding(mesh, P("x"))


@needs_four
def test_multi_device_placement_keeps_label():
    _, sharded = mesh4()
    r = jax.device_put(arange(16.0) * u.angstrom, sharded)
    assert units.is_quantity(r)
    assert spec(r) == P("x")
    assert units.dim_of(r) == units.dim_of(u.m)
    assert units.base_of(r) is units.ATOMISTIC
    assert float(r.sum().to(u.angstrom)) == pytest.approx(120.0)  # 0+1+...+15
    tree = jax.device_put({"r": r, "n": arange(16.0)}, sharded)
    assert spec(tree["r"]) == P("x") and spec(tree["n"]) == P("x")
    assert units.dim_of(tree["r"]) == units.dim_of(u.m)


@needs_four
def test_multi_device_jit_and_grad():
    _, sharded = mesh4()
    r = jax.device_put(arange(1.0, 17.0) * u.angstrom, sharded)
    k = 2.0 * u.eV / u.angstrom**2

    def energy(x: Any) -> Any:
        return 0.5 * k * x**2

    out = jax.jit(energy, in_shardings=sharded, out_shardings=sharded)(r)
    assert spec(out) == P("x")
    assert units.dim_of(out) == units.dim_of(u.J)
    assert float(out.max().to(u.eV)) == pytest.approx(256.0)  # 0.5*2*16^2
    force = jax.jit(units.grad(lambda x: energy(x).sum()), in_shardings=sharded)(r)
    assert units.dim_of(force) == units.dim_of(u.eV / u.angstrom)
    assert float(force.min().to(u.eV / u.angstrom)) == pytest.approx(2.0)  # 2*1


@needs_four
def test_multi_device_shard_map_psum():
    mesh, sharded = mesh4()

    def per_shard(x: Any) -> Any:
        # sum of this shard, psummed across all shards
        total = jax.lax.psum(x.sum().value, "x")
        return units.from_dim(total, units.dim_of(x), units.base_of(x))

    r = jax.device_put(ones(16) * u.angstrom, sharded)
    out: Any = jax.shard_map(per_shard, mesh=mesh, in_specs=P("x"), out_specs=P())(r)
    assert units.dim_of(out) == units.dim_of(u.m)
    assert float(out.to(u.angstrom)) == pytest.approx(16.0)


def test_placement_is_jaxs_own():
    moved = jax.device_put(arr([1.0, 2.0]) * u.angstrom, SHARDED)
    assert val(jax.device_get(moved)[0]) == 1.0
