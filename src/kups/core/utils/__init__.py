# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Core utilities for JAX-based scientific computing.

This package provides essential utilities for numerical computing with JAX,
organized into the following modules:

- **[ema][kups.core.utils.ema]**: Exponential moving average with numerical stability
- **[functools][kups.core.utils.functools]**: Functional programming combinators (compose, curry, etc.)
- **[jax][kups.core.utils.jax]**: JAX-specific utilities (jit, vectorize, dataclass, etc.)
- **[math][kups.core.utils.math]**: Mathematical functions (cubic roots, matrix operations)
- **[ops][kups.core.utils.ops]**: Array operations with broadcasting utilities
- **[position][kups.core.utils.position]**: Particle position utilities for periodic systems
- **[quaternion][kups.core.utils.quaternion]**: 3D rotation representation and operations

Most commonly used items can be imported directly from submodules.
"""
