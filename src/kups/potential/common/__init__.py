# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for potential energy calculations.

This module provides common infrastructure used by both classical and machine learning
potentials, including graph construction from atomic coordinates and energy composition
patterns for efficient incremental updates.

## Components

- **[graph][kups.potential.common.graph]**: Graph construction from atomic positions (point clouds, hypergraphs)
- **[energy][kups.potential.common.energy]**: Energy computation patterns with incremental updates
- **[direct][kups.potential.common.direct]**: Direct potential for models providing precomputed gradients
- **[evaluation][kups.potential.common.evaluation]**: One-shot potential evaluation with assertion retry logic

These utilities enable efficient neighbor list construction, batched graph representations,
and optimized energy evaluations that avoid redundant calculations during Monte Carlo moves.
"""
