# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Element reference data for the D3 dispersion correction.

The raw tables live in [_tables.py][kups.potential.dispersion.data._tables] and in
``d3_reference_c6.npz``, each in the unit its own upstream quotes -- Angstrom for
the covalent radii, atomic units for the rest. This module is the only
place that converts them: everything
[load_d3_reference][kups.potential.dispersion.data.load_d3_reference] returns is
already in kUPS units (Angstrom, electronvolt), so the jitted D3 kernels never
perform a unit conversion.

Provenance and licensing of the underlying values are documented in ``README.md``
next to this file.
"""

from __future__ import annotations

import functools
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.constants import BOHR, HARTREE
from kups.core.utils.jax import dataclass
from kups.potential.dispersion.data._tables import (
    COVALENT_RADII_2009,
    MAX_ATOMIC_NUMBER,
    MAX_REFERENCES,
    R4_OVER_R2,
    REFERENCE_CN,
)

__all__ = [
    "MAX_ATOMIC_NUMBER",
    "MAX_REFERENCES",
    "D3Reference",
    "load_d3_reference",
]

_C6_FILE = Path(__file__).parent / "d3_reference_c6.npz"

_CN_RADII_SCALE = 4.0 / 3.0
"""Grimme's scaling of the Pyykko covalent radii for the D3 counting function."""

_UNUSED_REFERENCE = -1.0
"""Sentinel in ``reference_cn`` marking a reference slot an element does not use."""


@dataclass
class D3Reference:
    """D3 element tables, indexed by atomic number, in kUPS units.

    Row 0 of every table is an unused placeholder so that atomic numbers index
    directly. Reference slots an element does not use are masked by
    ``reference_mask``; their ``reference_cn`` entries hold ``-1`` and their
    ``reference_c6`` entries hold zero.

    Attributes:
        covalent_radii: D3 coordination-number radii [Å], shape ``(n_elements,)``.
        r4r2: ``sqrt(0.5 * sqrt(Z) * <r^4>/<r^2>)`` [Å], shape ``(n_elements,)``.
            Enters ``C8`` as ``3 * r4r2_i * r4r2_j * C6_ij``.
        reference_cn: Reference coordination numbers [dimensionless],
            shape ``(n_elements, n_references)``.
        reference_c6: Reference dispersion coefficients [eV·Å⁶],
            shape ``(n_elements, n_elements, n_references, n_references)``.
        reference_mask: ``True`` where a reference slot is in use,
            shape ``(n_elements, n_references)``.
    """

    covalent_radii: Array  # (n_elements,) float
    r4r2: Array  # (n_elements,) float
    reference_cn: Array  # (n_elements, n_references) float
    reference_c6: Array  # (n_elements, n_elements, n_references, n_references) float
    reference_mask: Array  # (n_elements, n_references) bool

    @property
    def n_elements(self) -> int:
        """Number of rows, i.e. ``MAX_ATOMIC_NUMBER + 1``."""
        return self.covalent_radii.shape[0]

    @property
    def n_references(self) -> int:
        """Number of reference-system slots per element."""
        return self.reference_cn.shape[-1]


def _load_reference_c6() -> np.ndarray:
    """Unpack the symmetric reference-C6 table from its upper-triangular store."""
    n = MAX_ATOMIC_NUMBER + 1
    with np.load(_C6_FILE) as handle:
        upper = handle["c6_upper"]
    rows, cols = np.triu_indices(n)
    dense = np.zeros((n, n, MAX_REFERENCES, MAX_REFERENCES), dtype=np.float64)
    dense[rows, cols] = upper
    dense[cols, rows] = upper.transpose(0, 2, 1)
    return dense


@functools.cache
def _load_tables() -> tuple[np.ndarray, ...]:
    """Read and convert the tables once, in float64, independent of JAX config."""
    radii_2009 = np.asarray(COVALENT_RADII_2009, dtype=np.float64)
    covalent_radii = _CN_RADII_SCALE * radii_2009  # already Å upstream

    atomic_numbers = np.arange(len(R4_OVER_R2), dtype=np.float64)
    r4_over_r2 = np.asarray(R4_OVER_R2, dtype=np.float64)
    # sqrt of a squared length in Bohr -> a length in Bohr -> Å
    r4r2 = np.sqrt(0.5 * r4_over_r2 * np.sqrt(atomic_numbers)) * BOHR

    reference_cn = np.asarray(REFERENCE_CN, dtype=np.float64)
    reference_c6 = _load_reference_c6() * HARTREE * BOHR**6

    n = MAX_ATOMIC_NUMBER + 1
    return (
        covalent_radii[:n],
        r4r2[:n],
        reference_cn[:n],
        reference_c6,
        reference_cn[:n] > _UNUSED_REFERENCE,
    )


def load_d3_reference() -> D3Reference:
    """Load the D3 element tables, converted to Å and eV.

    The underlying float64 tables are read and unpacked at most once per process;
    the JAX arrays are rebuilt on each call so their dtype tracks the current
    ``jax_enable_x64`` setting rather than whichever setting happened to be active
    at first use.

    Returns:
        Reference tables ready to be embedded in ``D3Parameters``.
    """
    covalent_radii, r4r2, reference_cn, reference_c6, reference_mask = _load_tables()
    return D3Reference(
        covalent_radii=jnp.asarray(covalent_radii),
        r4r2=jnp.asarray(r4r2),
        reference_cn=jnp.asarray(reference_cn),
        reference_c6=jnp.asarray(reference_c6),
        reference_mask=jnp.asarray(reference_mask),
    )
