# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Physical units for JAX, checked at runtime and in the type system.

    >>> from kups.core import units
    >>> from kups.core.units import catalog as u
    >>> 1 * u.nm + 5 * u.angstrom
    15.0 Å
    >>> with units.base(units.SI):
    ...     1 * u.km + 5 * u.m
    1005.0 m

Model:
  A ``Unit`` is a named multiple of a dimension, five rational exponents over
  ``(m, kg, s, A, K)``; ``mol`` is dimensionless with scale ``N_A``. Multiplying
  magnitudes by a unit gives a ``Quantity`` whose magnitudes are stored in the
  active base: ``ATOMISTIC`` (Å, amu, eV, e, K) unless ``set_base`` or a
  ``with base(...)`` scope chooses another, such as ``SI``. A dimensionless result
  decays to a plain ``jax.Array``.

  A quantity is a pytree whose only leaf is its magnitudes; the dimension and base
  live in the treedef, so ``jax.jit``, ``jax.vmap`` and ``jax.lax.scan`` take
  quantities unchanged and lower to the same HLO as plain arrays. ``Unit`` and
  ``Quantity`` are generic over symbolic integer exponents, so pyrefly tracks
  dimensions through arithmetic; at runtime any mismatch raises ``DimensionError``.
  ``E_M``, ``E_KG``, ``E_S``, ``E_A`` and ``E_K`` are those exponents as type
  variables, for signatures generic over the dimension, such as
  ``Quantity[E_M, E_KG, E_S, E_A, E_K]``; ``F_M`` to ``F_K`` name a second
  operand's.

  ``grad``, ``value_and_grad``, ``hessian`` and ``vjp`` label derivatives
  ``dim(out) / dim(arg)``. The array and linear algebra functions
  here cover the cases ``jnp`` rejects; ``apply`` wraps any other function and
  ``relabel`` states a dimension the type algebra cannot compute. ``Log`` holds
  ``log(x / ref)`` with a static reference unit, for values spanning many decades.

Caveats:
  * ``jax.grad`` gives a gradient the primal's dimension. This is silently wrong
    rather than an error; use ``units.grad``.
  * ``jax.tree.map`` over a single tree treats a quantity as a node, so a nonlinear
    function mislabels it: ``jax.tree.map(jnp.sqrt, 4.0 * u.angstrom**2)`` gives
    ``2.0 Å^2``. Use ``units.tree_map``, or pass ``is_leaf=units.is_leaf``.
  * A plain number never stands in for a dimensioned operand, zero included:
    ``q + 0``, ``q > 0``, ``units.maximum(q, 0.0)`` and ``q.at[i].set(0.0)`` all
    raise. Write the zero with its unit, ``0.0 * u.eV``, or use
    ``units.zeros_like(q)``.
  * Quantities never become arrays implicitly, and ``==`` raises. Use
    ``q.to(unit)``, ``q.value``, ``float(q)`` or ``q.item()``, and compare with
    ``kups.core.units.testing``. ``jnp.zeros_like`` drops the unit;
    ``units.zeros_like`` keeps it.
"""

from kups.core.units import catalog as catalog
from kups.core.units._array import (
    absolute,
    clip,
    concatenate,
    full,
    full_like,
    linspace,
    max,
    maximum,
    mean,
    min,
    minimum,
    norm,
    ones,
    ones_like,
    prod,
    select_n,
    sqrt,
    stack,
    sum,
    where,
    zeros,
    zeros_like,
)
from kups.core.units._base import (
    ATOMISTIC,
    SI,
    BaseSystem,
    active_base,
    base,
    set_base,
)
from kups.core.units._core import (
    Acceleration,
    Action,
    Area,
    Charge,
    Compressibility,
    CoulombConstant,
    Current,
    Density,
    DipoleMoment,
    Energy,
    Entropy,
    Force,
    Frequency,
    Length,
    Mass,
    Measured,
    Momentum,
    NumberDensity,
    OddExponentRoot,
    Permittivity,
    Power,
    Pressure,
    Quantity,
    Stiffness,
    Temperature,
    Time,
    Velocity,
    Voltage,
    Volume,
    Wavenumber,
    apply,
    base_of,
    dim_of,
    dim_str,
    from_dim,
    is_leaf,
    is_quantity,
    magnitudes,
    magnitudes_of,
    tree_flatten,
    tree_leaves,
    tree_map,
)
from kups.core.units._linalg import (
    cross,
    det,
    eigh,
    einsum,
    inv,
    qr,
)
from kups.core.units._log import Log
from kups.core.units._transforms import (
    grad,
    hessian,
    relabel,
    value_and_grad,
    vectorize,
    vjp,
)
from kups.core.units._typed import _E_A as E_A
from kups.core.units._typed import _E_K as E_K
from kups.core.units._typed import _E_KG as E_KG
from kups.core.units._typed import _E_M as E_M
from kups.core.units._typed import _E_S as E_S
from kups.core.units._typed import _F_A as F_A
from kups.core.units._typed import _F_K as F_K
from kups.core.units._typed import _F_KG as F_KG
from kups.core.units._typed import _F_M as F_M
from kups.core.units._typed import _F_S as F_S
from kups.core.units._unit import (
    ONE,
    AccelerationUnit,
    ActionUnit,
    AreaUnit,
    ChargeUnit,
    CompressibilityUnit,
    CoulombConstantUnit,
    CurrentUnit,
    DensityUnit,
    Dim,
    DimensionError,
    Dimensionless,
    DipoleMomentUnit,
    EnergyUnit,
    EntropyUnit,
    ForceUnit,
    FrequencyUnit,
    LengthUnit,
    MassUnit,
    MomentumUnit,
    NumberDensityUnit,
    PermittivityUnit,
    PowerUnit,
    PressureUnit,
    StiffnessUnit,
    TemperatureUnit,
    TimeUnit,
    Unit,
    VelocityUnit,
    VoltageUnit,
    VolumeUnit,
    WavenumberUnit,
    dmul,
)

__all__ = [
    # submodules
    "catalog",
    # core types
    "Quantity",
    "Measured",
    "Unit",
    "Log",
    "Dimensionless",
    "DimensionError",
    "OddExponentRoot",
    # dimension aliases
    "Acceleration",
    "Action",
    "Area",
    "Charge",
    "Compressibility",
    "CoulombConstant",
    "Current",
    "Density",
    "DipoleMoment",
    "Energy",
    "Entropy",
    "Force",
    "Frequency",
    "Length",
    "Mass",
    "Momentum",
    "NumberDensity",
    "Permittivity",
    "Power",
    "Pressure",
    "Stiffness",
    "Temperature",
    "Time",
    "Velocity",
    "Voltage",
    "Volume",
    "Wavenumber",
    # unit aliases
    "AccelerationUnit",
    "ActionUnit",
    "AreaUnit",
    "ChargeUnit",
    "CompressibilityUnit",
    "CoulombConstantUnit",
    "CurrentUnit",
    "DensityUnit",
    "DipoleMomentUnit",
    "EnergyUnit",
    "EntropyUnit",
    "ForceUnit",
    "FrequencyUnit",
    "LengthUnit",
    "MassUnit",
    "MomentumUnit",
    "NumberDensityUnit",
    "PermittivityUnit",
    "PowerUnit",
    "PressureUnit",
    "StiffnessUnit",
    "TemperatureUnit",
    "TimeUnit",
    "VelocityUnit",
    "VoltageUnit",
    "VolumeUnit",
    "WavenumberUnit",
    # base systems
    "BaseSystem",
    "SI",
    "ATOMISTIC",
    "base",
    "set_base",
    "active_base",
    # pytree helpers
    "is_quantity",
    "is_leaf",
    "magnitudes",
    "magnitudes_of",
    "tree_map",
    "tree_leaves",
    "tree_flatten",
    # dimensions
    "dim_of",
    "base_of",
    "dim_str",
    "from_dim",
    "ONE",
    "Dim",
    "dmul",
    # exponent type variables
    "E_M",
    "E_KG",
    "E_S",
    "E_A",
    "E_K",
    "F_M",
    "F_KG",
    "F_S",
    "F_A",
    "F_K",
    # derivatives
    "grad",
    "value_and_grad",
    "hessian",
    "vjp",
    "relabel",
    # transforms
    "vectorize",
    "apply",
    # array functions
    "sqrt",
    "norm",
    "minimum",
    "maximum",
    "where",
    "clip",
    "select_n",
    "absolute",
    # reductions
    "sum",
    "mean",
    "min",
    "max",
    "prod",
    # linear algebra
    "det",
    "inv",
    "cross",
    "einsum",
    "qr",
    "eigh",
    # joining
    "stack",
    "concatenate",
    # constructors
    "linspace",
    "zeros",
    "ones",
    "full",
    "zeros_like",
    "ones_like",
    "full_like",
]
