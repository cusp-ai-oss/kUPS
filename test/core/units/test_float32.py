# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Float32 tests (x64 off, jax's default): rescales must be right or raise."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core import units
from kups.core.units import E_A, E_K, E_KG, E_M, E_S, Unit
from kups.core.units import catalog as u
from kups.core.units import testing as ut

from ._helpers import arr, x64_mode


@pytest.fixture(autouse=True)
def float32():
    with x64_mode(False):
        yield


def _roundtrip(unit: Unit[E_M, E_KG, E_S, E_A, E_K]) -> float:
    return float((1.0 * unit).to(unit))


def test_to_rescales_once():
    # two-factor conversions can hit a subnormal intermediate that flushes to zero
    for got in (
        _roundtrip(u.fs**3),
        _roundtrip(u.angstrom**3 * u.fs),
        _roundtrip(u.amu * u.angstrom**3),
        _roundtrip(u.nm**6),
        _roundtrip(1 / u.fs**3),
    ):
        assert jnp.isfinite(got) and abs(got - 1.0) < 1e-6, got
    assert float((12.5 * u.amu * u.angstrom**2).to(u.amu * u.nm**2)) == pytest.approx(
        0.125, rel=1e-6
    )
    assert float((1.0 * u.angstrom**4).to(u.nm**4)) == pytest.approx(1e-4, rel=1e-6)


def test_unit_operand_range_guard():
    # without the guard, in SI 63.6 * u.eV * u.angstrom**6 silently evaluates to 0.0
    with units.base(units.SI):
        for build in (
            lambda: 63.6 * u.eV * u.angstrom**6,
            lambda: 1.0 * u.angstrom**4,  # 1e-40: subnormal, not below subnormals
            lambda: (1e20 * u.m) * (u.fs**3),
            lambda: (1.0 * u.eV) / u.angstrom**6,
            lambda: units.full_like(arr([1.0, 1.0]) * u.m**6, u.angstrom**6),
            lambda: units.full_like(
                arr([1.0, 1.0]) * u.J / u.m**6, u.eV / u.angstrom**6
            ),
            # pyrefly: ignore[no-matching-overload]
            lambda: units.minimum(1.0 * u.m**4, u.angstrom**4),
            lambda: jax.jit(lambda q: q * u.angstrom**6)(1.0 * u.eV),
            lambda: units.ones((2,), u.angstrom**6),
            lambda: units.full((2,), u.angstrom**6),
        ):
            with pytest.raises(units.DimensionError):
                build()
        # suggests x64 when that would give the needed range
        with pytest.raises(units.DimensionError, match=r"enable jax\.config\.update"):
            units.full_like(arr([1.0, 1.0]) * u.m**6, u.angstrom**6)
        # values float32 can hold still work
        assert units.full_like(arr([1.0, 1.0]) * u.m, u.angstrom).value.shape == (2,)
    # a requested dtype is checked, not the operands'
    with x64_mode(True), units.base(units.SI):
        with pytest.raises(units.DimensionError, match="not a normal float32"):
            units.full_like(arr([1.0, 1.0]) * u.m**6, u.angstrom**6, dtype=jnp.float32)
    fourth = arr([1.0, 1.0]) * u.angstrom**4
    with pytest.raises(units.DimensionError):
        units.full_like(fourth, u.m**4)
    assert float(((3.0 * u.angstrom) * u.angstrom).to(u.angstrom**2)) == pytest.approx(
        3.0, rel=1e-6
    )
    assert float((5.0 * u.kcal / u.mol + 1.0 * u.eV).to(u.eV)) == pytest.approx(
        1.2168205, rel=1e-6
    )


def test_unrepresentable_factor_refused():
    # refused up front: 0 * inf would be nan
    with pytest.raises(units.DimensionError, match="not a normal float32"):
        jnp.zeros((2, 3)) * u.kg**2
    with pytest.raises(units.DimensionError, match="not a normal float32"):
        0.0 * u.m**12
    # Å^6 is 1e-60 in SI; the check reads only the factor, so jit refuses it too
    with units.base(units.SI):
        q = arr([1.0, 2.0]) * u.m**6
        with pytest.raises(units.DimensionError, match="not a normal float32"):
            q.to(u.angstrom**6)
        with pytest.raises(units.DimensionError, match="not a normal float32"):
            jax.jit(lambda x: x.to(u.angstrom**6))(q)


def test_unit_operand_and_rebase_factors_refused():
    # Å^6 is 1e-60 in SI, below float32's smallest normal
    with units.base(units.SI):
        q = arr([1.0, 2.0]) * u.m**6
        with pytest.raises(units.DimensionError, match="one Å\\^6 is .* in base SI"):
            q * u.angstrom**6
    with pytest.raises(
        units.DimensionError, match="re-expressing 'm\\^6' from base SI"
    ):
        q + arr([1.0, 2.0]) * u.angstrom**6  # rebased into atomistic
    # a comparison, and a Unit fill, are checked against the operand's float32
    r = units.from_dim(arr([1.0, 2.0]), units.dim_of(u.kg**2))
    with pytest.raises(units.DimensionError, match="one kg\\^2 is"):
        _ = r < u.kg**2
    with pytest.raises(units.DimensionError, match="one kg\\^2 is"):
        r.take(arr([0, 5]), mode="fill", fill_value=u.kg**2)


def test_amu_angstrom2_in_si_is_refused():
    # one amu.Å^2 is 1.7e-47 kg.m^2, below float32's smallest subnormal
    with units.base(units.SI):
        with pytest.raises(units.DimensionError, match="not a normal float32"):
            _ = 12.5 * (u.amu * u.angstrom**2)
    inertia = 12.5 * u.amu * u.angstrom**2
    with pytest.raises(units.DimensionError, match="not a normal float32"):
        inertia.rebase(units.SI)


def test_subnormal_unit_size_refused():
    # one debye.nm is 3.3e-39 in SI: dividing by it needs a subnormal float32,
    # though its reciprocal 3e38 is normal
    unit = u.debye * u.nm
    q = units.from_dim(arr([1.0, 2.0]), units.dim_of(unit), units.SI)
    with pytest.raises(units.DimensionError, match="not a normal float32"):
        q.to(unit)


def test_astype_refuses_unavailable_dtypes():
    q = arr([1.0, 2.0]) * u.eV
    for dtype in (jnp.float64, np.dtype("float64")):
        with (
            pytest.warns(UserWarning, match="not available"),
            pytest.raises(units.DimensionError, match="magnitudes are unavailable"),
        ):
            q.astype(dtype)
    assert q.astype(float).dtype == jnp.float32  # float means the default dtype


def test_advice_suggests_x64():
    # float32 is the default without x64, so enabling it gives the range
    with pytest.raises(units.DimensionError, match=r"enable jax\.config\.update"):
        jnp.zeros(2) * u.kg**2


def test_advice_matches_the_dtype():
    # with x64 on, float16 needs a wider dtype, not x64
    with x64_mode(True):
        with units.base(units.SI):
            q = jnp.ones(2, jnp.float16) * u.m
        for convert in (
            lambda: jnp.ones(2, jnp.float16) * u.km,
            lambda: q.to(u.angstrom),
            lambda: q + jnp.ones(2, jnp.float16) * u.angstrom,  # rebased
            lambda: q * u.angstrom,  # the Unit operand, read in q's float16
        ):
            with pytest.raises(units.DimensionError) as err:
                convert()
            assert "x64" not in str(err.value)
            assert "wider than float16" in str(err.value)


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


def test_grad_across_bases_in_float32():
    # the cotangent rescale from the output's base to the primal's runs in float32
    k = 2.0 * u.eV / u.angstrom**2

    def energy(x: units.Length) -> units.Energy:
        return (0.5 * k * x**2).sum()

    with units.base(units.SI):
        r = arr([1.0, 2.0]) * u.angstrom
    g = units.grad(energy)(r)
    assert units.base_of(g) == units.SI and g.dtype == jnp.float32
    ut.assert_allclose(g, arr([2.0, 4.0]) * (u.eV / u.angstrom), u.eV / u.angstrom)
