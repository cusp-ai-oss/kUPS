# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""A deliberately slow, explicit NumPy reference for pairwise D3(BJ).

This module exists to separate two questions that are easy to conflate when a
vectorized JAX kernel disagrees with an external code: *is the physics right*
and *is the vectorization right*. It implements the equations directly, with
plain Python loops over atoms and lattice translations and no clever masking, so
it can be read against the literature line by line. It is validated against
``simple-dftd3`` in ``test_d3_reference.py``; the JAX kernels are then validated
against it.

Everything here is in kUPS units (Å, eV).
"""

from __future__ import annotations

import itertools

import numpy as np

K_CN = 16.0
"""Steepness of the D3 coordination-number counting function."""

K_WEIGHT = 4.0
"""Gaussian weighting exponent for reference-C6 interpolation."""


def _translations(cell: np.ndarray, cutoff: float) -> np.ndarray:
    """Every lattice translation that can bring a pair within ``cutoff``.

    Deliberately generous by one shell: this is a reference implementation and
    the extra images only cost time.
    """
    if cell is None:
        return np.zeros((1, 3))
    inverse = np.linalg.inv(cell)
    # perpendicular width along axis k is 1 / |column k of the inverse|
    perpendicular = 1.0 / np.linalg.norm(inverse, axis=0)
    reps = np.ceil(cutoff / perpendicular).astype(int) + 1
    ranges = [range(-r, r + 1) for r in reps]
    return np.array([np.array(n) @ cell for n in itertools.product(*ranges)])


def coordination_numbers(
    numbers: np.ndarray,
    positions: np.ndarray,
    covalent_radii: np.ndarray,
    *,
    cell: np.ndarray | None = None,
    cn_cutoff: float,
) -> np.ndarray:
    """D3 coordination numbers, summed over neighbours and periodic images."""
    trans = _translations(cell, cn_cutoff)
    n = len(numbers)
    cn = np.zeros(n)
    for i in range(n):
        r_cov_i = covalent_radii[numbers[i]]
        for j in range(n):
            r_cov_j = covalent_radii[numbers[j]]
            r_c = r_cov_i + r_cov_j
            for t in trans:
                r = np.linalg.norm(positions[j] + t - positions[i])
                # excludes the i == j, T == 0 self pair but keeps self images
                if r > cn_cutoff or r < 1e-12:
                    continue
                cn[i] += 1.0 / (1.0 + np.exp(-K_CN * (r_c / r - 1.0)))
    return cn


def reference_weights(
    numbers: np.ndarray,
    cn: np.ndarray,
    reference_cn: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    """Per-atom normalized Gaussian weights over the reference systems.

    The joint pair weight ``exp(-k3[(dCN_i)^2 + (dCN_j)^2])`` factorizes, so the
    joint normalization equals the product of these per-atom normalizations.
    Computed with the maximum exponent subtracted, which is exact and removes any
    possibility of the normalization underflowing to zero.
    """
    n_ref = reference_cn.shape[1]
    weights = np.zeros((len(numbers), n_ref))
    for i, z in enumerate(numbers):
        mask = reference_mask[z]
        exponent = np.where(mask, -K_WEIGHT * (cn[i] - reference_cn[z]) ** 2, -np.inf)
        exponent -= exponent.max()
        w = np.where(mask, np.exp(exponent), 0.0)
        weights[i] = w / w.sum()
    return weights


def dispersion_energy(
    numbers: np.ndarray,
    positions: np.ndarray,
    reference: dict[str, np.ndarray],
    *,
    cell: np.ndarray | None = None,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
    cutoff: float,
    cn_cutoff: float,
) -> float:
    """Total pairwise D3(BJ) dispersion energy [eV].

    ``reference`` carries the ``covalent_radii``, ``r4r2``, ``reference_cn``,
    ``reference_c6`` and ``reference_mask`` tables in kUPS units.
    """
    cn = coordination_numbers(
        numbers, positions, reference["covalent_radii"], cell=cell, cn_cutoff=cn_cutoff
    )
    weights = reference_weights(
        numbers, cn, reference["reference_cn"], reference["reference_mask"]
    )
    r4r2 = reference["r4r2"]
    c6_ref = reference["reference_c6"]

    trans = _translations(cell, cutoff)
    energy = 0.0
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            z_i, z_j = numbers[i], numbers[j]
            c6 = weights[i] @ c6_ref[z_i, z_j] @ weights[j]
            rr = 3.0 * r4r2[z_i] * r4r2[z_j]
            c8 = rr * c6
            r0 = a1 * np.sqrt(rr) + a2
            for t in trans:
                r = np.linalg.norm(positions[j] + t - positions[i])
                if r > cutoff or r < 1e-12:
                    continue
                energy -= 0.5 * (s6 * c6 / (r**6 + r0**6) + s8 * c8 / (r**8 + r0**8))
    return float(energy)


def reference_tables() -> dict[str, np.ndarray]:
    """The embedded kUPS tables as plain NumPy, for use by this oracle."""
    from kups.potential.dispersion.data import load_d3_reference

    ref = load_d3_reference()
    return {
        "covalent_radii": np.asarray(ref.covalent_radii, dtype=np.float64),
        "r4r2": np.asarray(ref.r4r2, dtype=np.float64),
        "reference_cn": np.asarray(ref.reference_cn, dtype=np.float64),
        "reference_c6": np.asarray(ref.reference_c6, dtype=np.float64),
        "reference_mask": np.asarray(ref.reference_mask),
    }
