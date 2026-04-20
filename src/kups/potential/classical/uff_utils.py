# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""UFF (Universal Force Field) utility functions.

Shared calculations used by UFF-style potentials (bond length, force constant formulas).

Reference: Rappé et al. (1992) "UFF, a Full Periodic Table Force Field"
J. Am. Chem. Soc. 114, 10024-10035. DOI: 10.1021/ja00051a040
"""

import jax.numpy as jnp
from jax import Array


def compute_uff_bond_length(
    bond_radius: Array,
    electronegativity: Array,
) -> Array:
    r"""Compute UFF bond lengths with electronegativity correction.

    Implements UFF Equations 2 and 4:
    - $r_{ij} = r_i + r_j - r_{EN}$ (Eq. 2)
    - $r_{EN} = \frac{r_i r_j (\sqrt{\chi_i} - \sqrt{\chi_j})^2}{\chi_i r_i + \chi_j r_j}$ (Eq. 4)

    Args:
        bond_radius: Valence bond radii [Å], shape `(n_species,)`
        electronegativity: GMP electronegativity, shape `(n_species,)`

    Returns:
        Bond length matrix, shape `(n_species, n_species)`
    """
    r_i, r_j = bond_radius[:, None], bond_radius[None, :]
    chi_i, chi_j = electronegativity[:, None], electronegativity[None, :]

    sqrt_chi_diff = jnp.sqrt(chi_i) - jnp.sqrt(chi_j)
    denom = chi_i * r_i + chi_j * r_j
    r_EN = jnp.where(jnp.abs(denom) > 1e-10, r_i * r_j * sqrt_chi_diff**2 / denom, 0.0)
    return r_i + r_j - r_EN
