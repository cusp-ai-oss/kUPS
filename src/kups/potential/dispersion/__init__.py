# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dispersion corrections for density functionals and MLIPs.

Semi-local DFT — and machine-learned potentials trained on it — miss long-range
London dispersion. These potentials add it back, and are designed to be composed
with an existing potential rather than used alone:

```python
from kups.core.potential import sum_potentials
from kups.application.potential.dispersion.d3 import make_d3_from_state
from kups.potential.dispersion import D3Parameters

d3 = make_d3_from_state(
    state_lens, parameters=D3Parameters.from_functional("pbe"), gradient=gradient
)
total = sum_potentials(mlip, d3)
```

## Available corrections

- **[D3][kups.potential.dispersion.d3]**: Grimme's D3 with Becke-Johnson damping,
  pairwise (two-body) term.
"""

from kups.potential.dispersion.d3 import (
    D3_DEFAULT_CN_CUTOFF,
    D3_DEFAULT_CUTOFF,
    D3_REFERENCE_CN_CUTOFF,
    D3_REFERENCE_CUTOFF,
    D3Parameters,
    d3_c6_coefficients,
    d3_coordination_numbers,
    d3_energy,
    make_d3_potential,
)
from kups.potential.dispersion.damping import (
    BECKE_JOHNSON_PARAMETERS,
    BeckeJohnsonDamping,
    available_functionals,
    damping_for_functional,
)
from kups.potential.dispersion.data import D3Reference, load_d3_reference

__all__ = [
    "BECKE_JOHNSON_PARAMETERS",
    "D3_DEFAULT_CN_CUTOFF",
    "D3_DEFAULT_CUTOFF",
    "D3_REFERENCE_CN_CUTOFF",
    "D3_REFERENCE_CUTOFF",
    "BeckeJohnsonDamping",
    "D3Parameters",
    "D3Reference",
    "available_functionals",
    "d3_c6_coefficients",
    "d3_coordination_numbers",
    "d3_energy",
    "damping_for_functional",
    "load_d3_reference",
    "make_d3_potential",
]
