# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Monte Carlo simulation components for canonical and grand canonical ensembles.

This package provides Monte Carlo move proposals and acceptance criteria for NVT
(canonical) and µVT (grand canonical) ensemble simulations. All components are
designed for use with [MCMCPropagator][kups.core.propagator.MCMCPropagator] and
support batched parallel systems.

## Module Organization

- **[moves][kups.mcmc.moves]**: MC move proposals (translations, rotations, insertions/deletions)
- **[probability][kups.mcmc.probability]**: Acceptance criteria (Boltzmann, fugacity, combined µVT)
- **[fugacity][kups.mcmc.fugacity]**: Equation of state calculations for real gas mixtures

## Typical Usage

```python
from kups.mcmc.moves import ParticleTranslationMove, ExchangeMove
from kups.mcmc.probability import BoltzmannLogProbabilityRatio, MuVTLogProbabilityRatio
from kups.core.propagator import MCMCPropagator, compose_propagators

# NVT simulation
translation_move = ParticleTranslationMove(...)
boltzmann_ratio = BoltzmannLogProbabilityRatio(...)
nvt_propagator = MCMCPropagator(
    proposal_fn=translation_move,
    log_probability_ratio_fn=boltzmann_ratio
)

# GCMC simulation
exchange_move = ExchangeMove(...)
muvt_ratio = MuVTLogProbabilityRatio(...)
gcmc_propagator = MCMCPropagator(
    proposal_fn=exchange_move,
    log_probability_ratio_fn=muvt_ratio
)

# Combine moves
combined = compose_propagators(nvt_propagator, gcmc_propagator)
```

See individual module documentation for detailed APIs.
"""

from kups.mcmc import fugacity, moves, probability

__all__ = ["moves", "probability", "fugacity"]
