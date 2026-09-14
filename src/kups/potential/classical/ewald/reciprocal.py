# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Ewald summation for long-range electrostatics in periodic systems.

Splits the Coulomb potential into short-range (real-space), long-range
(reciprocal-space), and self-interaction terms. Supports incremental
updates via cached structure factors for efficient Monte Carlo.
"""

from __future__ import annotations

import einops
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index
from kups.core.typing import (
    SystemId,
)
from kups.core.utils.segment import segment_sum


def _frequency_response(
    positions: Array,
    charges: Array,
    kvecs: Array,
    batch_mask: Index[SystemId],
) -> Array:
    """Per-particle response in reciprocal space.

    Math: ``rho_i(k) = q_i * [cos(k . r_i), sin(k . r_i)]``.

    Returns:
        Response array, shape ``(n_particles, n_kvecs, 2)`` for cos/sin.
    """
    exponent = einops.einsum(
        kvecs[batch_mask.indices],
        positions,
        "particles shifts dim, particles dim->particles shifts",
    )
    # particles x shifts x 2
    response = charges[:, None, None] * jnp.stack(
        [jnp.cos(exponent), jnp.sin(exponent)], axis=-1
    )
    return response


def _structure_factor_full(
    positions: Array,
    charges: Array,
    kvecs: Array,
    batch_mask: Index[SystemId],
) -> Array:
    """Full structure factor computation.

    Math: ``S(k) = sum_i rho_i(k)`` summed per system via segment_sum.

    Returns:
        Structure factor, shape ``(n_systems, n_kvecs, 2)``.
    """
    response = _frequency_response(positions, charges, kvecs, batch_mask)
    structure_factor = segment_sum(
        response,
        batch_mask.indices,
        batch_mask.num_labels,
        mode="drop",
    )
    return structure_factor
