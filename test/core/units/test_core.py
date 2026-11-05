# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for base systems."""

import asyncio
import itertools
import pickle
import subprocess
import sys
from fractions import Fraction

import pytest

from kups.core import units
from kups.core.units import catalog as u

from ._helpers import (
    ASE_TIME,
    ulps,
)


def test_default_base_is_atomistic():
    assert units.active_base() is units.ATOMISTIC
    assert units.SI.base_units() == {
        "m": 1.0,
        "kg": 1.0,
        "s": 1.0,
        "A": 1.0,
        "K": 1.0,
    }


@pytest.mark.parametrize("field", ["pins", "labels", "name"])
def test_base_metadata_is_immutable(field: str):
    bs = units.BaseSystem("custom", u.SI_BASE, ("m", "kg", "s", "A", "K"))
    original_hash = hash(bs)
    replacements = {
        "pins": (u.km, *bs.pins[1:]),
        "labels": ("km", *bs.labels[1:]),
        "name": "changed",
    }
    with pytest.raises(AttributeError):
        setattr(bs, field, replacements[field])
    assert bs.scale(u.m.dim) ** 2 == bs.scale((u.m**2).dim)
    assert bs == units.SI and hash(bs) == original_hash


def test_base_owns_its_metadata():
    pins = list(u.SI_BASE)
    labels = ["m", "kg", "s", "A", "K"]
    bs = units.BaseSystem("custom", pins, labels)  # pyrefly: ignore[bad-argument-type]
    pins[0], labels[0] = u.km, "km"
    assert bs.pins == u.SI_BASE
    assert bs.labels[0] == "m"
    assert bs.scale(u.m.dim) ** 2 == bs.scale((u.m**2).dim)


def test_base_hash_after_unpickling_in_another_process():
    custom = units.BaseSystem(
        "custom", (u.km, *units.SI.pins[1:]), ("km", *units.SI.labels[1:])
    )
    probe = """
import pickle
import sys
from kups.core import units
from kups.core.units import catalog as u

custom = units.BaseSystem(
    "custom", (u.km, *units.SI.pins[1:]), ("km", *units.SI.labels[1:])
)
for restored, local in zip(
    pickle.loads(sys.stdin.buffer.read()), (units.SI, units.ATOMISTIC, custom),
    strict=True,
):
    assert restored == local
    assert hash(restored) == hash(local)
    assert {restored: "found"}[local] == "found"
    assert len({restored, local}) == 1
"""
    out = subprocess.run(
        [sys.executable, "-c", probe],
        input=pickle.dumps((units.SI, units.ATOMISTIC, custom)),
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr.decode()


def test_reconfigured_base():
    # spelling the preset by hand gives the same base, sharing its jit cache
    with units.base(mass=u.amu, energy=u.eV, length=u.angstrom, charge=u.e) as bs:
        assert bs == units.ATOMISTIC
    # base identity is the induced scales, so these all resolve to SI itself
    for kw in ({"force": u.N}, {"energy": u.J}, {"len2": u.m**2}):
        with units.base(**kw) as bs:
            assert bs is units.SI


def test_base_solver():
    # pinning length, mass and energy yields ASE's time unit in any pin order
    pins = {"length": u.angstrom, "mass": u.amu, "energy": u.eV}
    for perm in itertools.permutations(pins.items()):
        with units.base(**dict(perm)) as bs:
            assert bs.base_units()["s"] == ASE_TIME
            # scale is a homomorphism: two routes to force give identical bits
            force = bs.scale((1, 1, -2, 0, 0))
            assert force == bs.scale((2, 1, -2, 0, 0)) / bs.scale((1, 0, 0, 0, 0))
            assert force == 1.6021766339999998e-09
    # any dimensionally independent units work, not just known kinds
    with units.base(len2=u.angstrom**2, mass=u.amu, energy=u.eV) as bs:
        assert bs.base_units()["s"] == ASE_TIME
    # each pin's scale is reproduced exactly
    for p in units.ATOMISTIC.pins:
        assert units.ATOMISTIC.scale(p.dim) == p.scale
    for e1 in itertools.product(range(-1, 2), repeat=3):
        for e2 in itertools.product(range(-1, 2), repeat=3):
            d1, d2 = (*e1, 0, 0), (*e2, 0, 0)
            ds = tuple(a + b for a, b in zip(d1, d2, strict=True))
            assert units.SI.scale(ds) == units.SI.scale(d1) * units.SI.scale(d2)
            assert (
                ulps(
                    units.ATOMISTIC.scale(ds),
                    units.ATOMISTIC.scale(d1) * units.ATOMISTIC.scale(d2),
                )
                <= 2
            )


def test_incoherent_base():
    for order in (
        ["length", "mass", "energy", "time"],
        ["time", "energy", "mass", "length"],
    ):
        kw = {"length": u.angstrom, "mass": u.amu, "energy": u.eV, "time": u.fs}
        with pytest.raises(units.DimensionError) as e:
            units.base(**{k: kw[k] for k in order}).__enter__()
        msg = str(e.value)
        assert all(f"={n}" in msg for n in ("Å", "amu", "eV", "fs"))
        assert "not 1" in msg and "is dimensionless" in msg
    with pytest.raises(units.DimensionError, match="energy must be"):
        units.base(energy=u.m).__enter__()  # a known kind must match its dimension


def test_dependent_base():
    with pytest.raises(
        units.DimensionError, match=r"\(length=m\)\^-1 \* \(length2=km\)\^1"
    ):
        units.base(length=u.m, length2=u.km).__enter__()
    with pytest.raises(units.DimensionError, match="drop one of them"):
        units.base(length=u.m, time=u.s, speed=u.m / u.s).__enter__()
    with pytest.raises(units.DimensionError):  # six units cannot be independent
        units.base(a=u.m, b=u.kg, c=u.s, d=u.A, e=u.K, f=u.J).__enter__()


def test_async_scopes():
    # each task keeps its own base across an await
    async def inner(bs: units.BaseSystem) -> tuple[str, str]:
        with units.base(bs):
            before = units.active_base().name
            await asyncio.sleep(0)
            return before, units.active_base().name

    async def main() -> list[tuple[str, str]]:
        return list(await asyncio.gather(inner(units.ATOMISTIC), inner(units.SI)))

    assert asyncio.run(main()) == [("atomistic", "atomistic"), ("SI", "SI")]
    assert units.active_base() is units.ATOMISTIC  # the caller's base is untouched


def test_base_validation():
    with pytest.raises(units.DimensionError, match="at least one pinned unit"):
        units.base().__enter__()
    with pytest.raises(units.DimensionError, match="at least one pinned unit"):
        units.set_base()
    with pytest.raises(units.DimensionError, match="fractional dimension"):
        units.base(rootlen=u.angstrom ** Fraction(1, 2)).__enter__()
    wrong: object = "angstrom"
    with pytest.raises(
        TypeError, match="kups.core.units: length= must be a Unit, not str"
    ):
        units.base(length=wrong).__enter__()  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(
        TypeError, match="kups.core.units: the base system must be a BaseSystem"
    ):
        units.base(wrong).__enter__()  # pyrefly: ignore[bad-argument-type]


def test_derived_symbols():
    # a derived unit with no catalogue name displays as a quotient
    at = units.ATOMISTIC
    assert at.symbol((1, 1, -2, 0, 0)) == "eV/Å"  # a force
    assert at.symbol((-1, 1, -2, 0, 0)) == "eV/Å^3"  # a stress
    assert at.symbol((0, 1, -2, 0, 0)) == "eV/Å^2"  # a hessian entry
    assert at.symbol((0, 0, 1, 0, 0)) == "amu^1/2.Å/eV^1/2"  # a time
    assert at.unit((1, 1, -2, 0, 0)) == u.eV / u.angstrom  # the pins, multiplied out
    assert units.SI.symbol((1, 0, -1, 0, 0)) == "m/s"
    assert units.SI.symbol((0, 0, -1, 0, 0)) == "Hz"  # a catalogue name wins
    assert units.SI.symbol((1, 0, 0, 0, 0)) == "m"
    assert units.SI.symbol((0, 0, 0, 0, 0)) == ""  # dimensionless has no symbol


@pytest.mark.parametrize("inner_base", [units.SI, units.ATOMISTIC])
def test_scope_nesting(inner_base: units.BaseSystem):
    outer, inner = units.base(units.SI), units.base(inner_base)
    outer.__enter__()
    inner.__enter__()
    for cm in (outer, inner):  # exiting the outer scope first corrupts the stack
        with pytest.raises(units.DimensionError, match="not the innermost one"):
            cm.__exit__(None, None, None)
    assert units.active_base() is units.ATOMISTIC  # state was reset to the default
    with units.base(units.SI):  # proper nesting still restores each level
        with units.base(inner_base):
            assert units.active_base() is inner_base
        assert units.active_base() is units.SI


def test_scope_restores_base_after_exception():
    with units.base(units.SI):
        with (
            pytest.raises(RuntimeError, match="inside the scope"),
            units.base(units.SI),
        ):
            raise RuntimeError("inside the scope")
        assert units.active_base() is units.SI
    assert units.active_base() is units.ATOMISTIC
