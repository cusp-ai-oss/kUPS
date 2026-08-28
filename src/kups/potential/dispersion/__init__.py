# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dispersion corrections for density functionals and MLIPs.

Semi-local DFT — and machine-learned potentials trained on it — miss long-range
London dispersion. This package holds the reference data and the fitted damping
parameters such a correction needs; the corrections themselves are built on top
of them.

## Contents

- **[data][kups.potential.dispersion.data]**: Grimme's D3 element tables, loaded
  and converted to kUPS units. Provenance and licensing are documented in
  ``data/README.md``.
- **[damping][kups.potential.dispersion.damping]**: Becke-Johnson damping
  parameters, fitted per density functional.
"""

from kups.potential.dispersion.damping import (
    BECKE_JOHNSON_PARAMETERS,
    BeckeJohnsonDamping,
    available_functionals,
    damping_for_functional,
)
from kups.potential.dispersion.data import D3Reference, load_d3_reference

__all__ = [
    "BECKE_JOHNSON_PARAMETERS",
    "BeckeJohnsonDamping",
    "D3Reference",
    "available_functionals",
    "damping_for_functional",
    "load_d3_reference",
]
