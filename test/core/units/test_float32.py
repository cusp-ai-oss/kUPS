# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Float32 tests (x64 off, jax's default): rescales must be right or raise."""

import os
import subprocess
import sys
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core import units
from kups.core.units import _precision
from kups.core.units import catalog as u
from kups.core.units import constants as k
from kups.core.units import testing as ut

from ._helpers import arr, x64_mode

# attaching units in float32 emits precision RuntimeWarnings
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

# enables x64 after importing kups.core.units; needs a fresh process
_LATE_X64 = """
from kups.core import units
import jax; jax.config.update('jax_enable_x64', True)
from kups.core.units import constants as k
print(k.c.value.dtype, float(k.c.value), float(k.h.value))
"""


def _units_warnings(
    caught: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    """Filter kups.core.units' precision warnings out of a recorded warning list.

    Args:
        caught: warnings recorded by a ``catch_warnings(record=True)`` block.

    Returns:
        The kups.core.units warnings only, so unrelated warnings cannot change a count.
    """
    return [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "kups.core.units:" in str(w.message)
    ]


@pytest.fixture(autouse=True)
def float32():
    _precision._WARNED.clear()
    with x64_mode(False):
        yield
    _precision._WARNED.clear()


def test_to_rescales_once():
    # two-factor conversions can hit a subnormal intermediate that flushes to zero
    cases: list[Any] = [
        u.fs**3,
        u.angstrom**3 * u.fs,
        u.amu * u.angstrom**3,
        u.nm**6,
        1 / u.fs**3,
    ]
    for unit in cases:
        got = float((1.0 * unit).to(unit))
        assert jnp.isfinite(got) and abs(got - 1.0) < 1e-6, unit
    assert float((12.5 * u.amu * u.angstrom**2).to(u.amu * u.nm**2)) == pytest.approx(
        0.125, rel=1e-6
    )
    assert float((1.0 * u.angstrom**4).to(u.nm**4)) == pytest.approx(1e-4, rel=1e-6)


def test_unit_operand_range_guard():
    # without the guard, in SI 63.6 * u.eV * u.angstrom**6 silently evaluates to 0.0
    with units.base(units.SI):
        for build in (
            lambda: 63.6 * u.eV * u.angstrom**6,
            lambda: (1e20 * u.m) * (u.fs**3),
            lambda: (1.0 * u.eV) / u.angstrom**6,
            lambda: units.full_like(arr([1.0, 1.0]) * u.m**6, u.angstrom**6),
            lambda: units.full_like(
                arr([1.0, 1.0]) * u.J / u.m**6, u.eV / u.angstrom**6
            ),
            # pyrefly: ignore[no-matching-overload]
            lambda: units.minimum(1.0 * u.m**4, u.angstrom**4),
            lambda: jax.jit(lambda q: q * u.angstrom**6)(1.0 * u.eV),
        ):
            with pytest.raises(units.DimensionError):
                build()
        # suggests x64 when that would give the needed range
        with pytest.raises(units.DimensionError, match=r"enable jax\.config\.update"):
            units.full_like(arr([1.0, 1.0]) * u.m**6, u.angstrom**6)
        # values float32 can hold still work
        assert units.full_like(arr([1.0, 1.0]) * u.m, u.angstrom).value.shape == (2,)
    fourth = arr([1.0, 1.0]) * u.angstrom**4
    with pytest.raises(units.DimensionError):
        units.full_like(fourth, u.m**4)
    assert float(((3.0 * u.angstrom) * u.angstrom).to(u.angstrom**2)) == pytest.approx(
        3.0, rel=1e-6
    )
    assert float((5.0 * u.kcal / u.mol + 1.0 * u.eV).to(u.eV)) == pytest.approx(
        1.2168205, rel=1e-6
    )


def test_destructive_rescale_refused():
    q = jnp.float32(0.05) * u.amu * u.angstrom
    with pytest.raises(units.DimensionError):
        q.rebase(units.SI)
    with units.base(units.SI):  # where metres attach for free and .to rescales
        with pytest.raises(units.DimensionError):
            (1e30 * u.m).to(u.angstrom)
        with pytest.raises(units.DimensionError):
            (jnp.array([1.0, 1e30, 2.0]) * u.m).to(u.angstrom)
        # zeros, infs and nans do not count as destroyed
        mixed = jnp.array([0.0, jnp.inf, jnp.nan, 1.0]) * u.m
        assert mixed.to(u.angstrom).shape == (4,)
        assert (jnp.array([]) * u.m).to(u.angstrom).size == 0
        assert float(jax.jit(lambda x: x.to(u.angstrom))(1.0 * u.m)) == 1e10


def test_unrepresentable_factor_refused():
    # refused up front: 0 * inf would be nan
    with pytest.raises(units.DimensionError):
        jnp.zeros((2, 3)) * u.kg**2
    with pytest.raises(units.DimensionError):
        0.0 * u.m**12


def test_host_magnitudes_range_checked():
    # checked before transfer, while subnormals still exist
    for build in (
        lambda: 1e-38 * u.angstrom,
        lambda: np.float32(1e-40) * u.km,
        lambda: np.float64(1e-300) * u.angstrom,
        lambda: np.float64(1e39) * u.mm,
        lambda: units.from_dim(np.array([1e-46, 2.5, 1e39]), units.dim_of(u.m)),
    ):
        with pytest.raises(units.DimensionError):
            build()
    for build in (
        lambda: 3.0 * u.angstrom,
        lambda: 0.0 * u.angstrom,
        lambda: np.array([0.0, 1.0, -0.0]) * u.angstrom,
        lambda: np.array([np.nan, np.inf, -np.inf]) * u.m,
        lambda: jnp.arange(3.0) * u.eV,
        lambda: jax.jit(lambda x: x * u.angstrom)(jnp.float32(3.0)),
    ):
        assert build() is not None


def test_constants_follow_base():
    # in float32 an SI-based h squares to zero
    assert float((k.h**2).to(u.eV**2 * u.s**2)) > 0.0
    ryd = float((k.hbar**2 / (2 * k.m_e * k.a_0**2)).to(u.eV))
    assert ryd == pytest.approx(13.6056931, rel=1e-6)
    bohr = float((4 * np.pi * k.eps_0 * k.hbar**2 / (k.m_e * k.e**2)).to(u.angstrom))
    assert bohr == pytest.approx(0.52917721, rel=1e-6)


def test_late_x64_switch():
    # constants are built on first access, so they see the switch
    env = {key: v for key, v in os.environ.items() if key != "JAX_ENABLE_X64"}
    out = subprocess.run(
        [sys.executable, "-c", _LATE_X64],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.split()
    assert out[0] == "float64"
    # the atomistic magnitudes, rounded once from the exact value; float32 differs
    assert float(out[1]) == 30520.38830711602
    assert float(out[2]) == 0.4062340137536605


def test_allclose_float32_rtol():
    # a fixed 1e-7 rtol could not pass at float32 precision
    ut.assert_allclose(1.0 * u.hartree, 27.211386245988 * u.eV, u.eV)
    ut.assert_allclose(1.0 * u.eV, 23.060547830619 * u.kcal / u.mol, u.eV)
    with pytest.raises(AssertionError):
        ut.assert_allclose(1.0 * u.eV, 1.000002 * u.eV, u.eV)


def test_scalar_repr():
    # jax's formatter, not the float64 expansion of a float32
    assert repr(0.1 * u.angstrom) == "0.1 Å"
    assert str(0.3 * u.eV) == "0.3 eV"
    assert f"{0.1 * u.angstrom:.2e}" == "1.00e-01 Å"
    with units.base(units.SI):
        assert repr(3.0 * u.angstrom) == "3e-10 m"
        assert f"{3.0 * u.angstrom:.2e}" == "3.00e-10 m"


def test_warning_advice_fits_case():
    with pytest.warns(RuntimeWarning) as rec:
        1.0 * u.mol  # dimensionless: no base can rescale it
    assert "work in a base" not in str(rec[0].message)
    _precision._WARNED.clear()
    with pytest.warns(RuntimeWarning) as rec:
        1.0 * u.J  # SI-specific example must not appear in another base
    assert "(3.4*u.angstrom)**6" not in str(rec[0].message)


def test_float32_warnings():
    # float32 warns once per unit and base, and never destroys a magnitude quietly
    with warnings.catch_warnings(record=True) as caught:  # in SI
        warnings.simplefilter("always")  # the hazard is underflow
        with units.base(units.SI):
            q = jnp.float32(3.4) * u.bohr
            _ = jnp.float32(1.0) * u.bohr
    warned = _units_warnings(caught)
    assert len(warned) == 1 and "underflow" in str(warned[0].message)
    assert "7 decimal digits" in str(warned[0].message)  # float32's precision
    assert "units.set_base(units.ATOMISTIC)" in str(warned[0].message)
    # the gate is precision, not one dtype name: bfloat16 loses 0.2% here
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with units.base(units.SI):
            _ = arr(1e4, dtype=jnp.bfloat16) * u.hartree
    warned = _units_warnings(caught)
    assert len(warned) == 1 and "2 decimal digits" in str(warned[0].message)
    with warnings.catch_warnings(record=True) as caught:  # in the atomistic base
        warnings.simplefilter("always")  # the hazard is overflow instead
        _ = arr(1.0, dtype=jnp.float32) * u.km
    warned = _units_warnings(caught)
    assert len(warned) == 1 and "overflow" in str(warned[0].message)
    assert "units.set_base(units.SI)" in str(warned[0].message)
    assert float((q**6).value) == 0.0  # the warning does not prevent the underflow
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with units.base(units.SI):
            with pytest.raises(units.DimensionError, match="flushes 1 of 1"):
                _ = jnp.float32(1e-30) * u.angstrom
            big = jnp.float32(3.4) * u.angstrom
        assert float((jnp.float32(3.4) * u.angstrom).value) == pytest.approx(3.4)
        with pytest.raises(units.DimensionError, match="overflows 1 of 1"):
            _ = jnp.float32(1e35) * u.m
        with pytest.raises(units.DimensionError, match="not a normal float32"):
            _ = (jnp.float32(1.0) * u.angstrom) ** 6 + big**6


def test_foreign_warnings_ignored():
    # the counts select kups.core.units' warnings, so jax's own warnings cannot fail a test
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(
            "Error reading persistent compilation cache entry",
            UserWarning,
            stacklevel=1,
        )
        _ = jnp.float32(1.0) * u.km  # 1e13 in the atomistic base, so it warns
    assert len(caught) == 2 and len(_units_warnings(caught)) == 1
