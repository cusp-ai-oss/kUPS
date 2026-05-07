# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system max-step-size clipping transform.

Unlike :func:`kups.relaxation.optax.max_step_size`, this version takes an
``index_prefix`` pytree (analogous to ``in_axes`` in :func:`jax.vmap`) whose
leaves are :class:`Index` objects mapping each parameter element to a system.
The maximum displacement is enforced *per system*, so batching independent
systems through one optimizer is bit-identical to running them one at a time.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp

from kups.core.data.index import Index
from kups.core.typing import PyTree
from kups.core.utils.jax import dataclass, field, tree_copy
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.transforms._segmented_tree import (
    tree_scale_per_row,
    tree_segment_max,
)


@dataclass
class MaxStepSizeState:
    """Optimizer state holding the ``index_prefix`` captured at init time.

    Attributes:
        index_prefix: Tree prefix of the parameter pytree whose leaves are
            :class:`Index` objects, or ``None`` for global (cross-system)
            clipping.
    """

    index_prefix: PyTree | None


@dataclass
class MaxStepSize[Params](Optimizer[Params, MaxStepSizeState]):
    """Clip updates so no element of any system moves more than ``max_step_size``.

    Per-element norms are computed along the last axis. For every system, the
    maximum norm across all elements assigned to that system (across every leaf
    of ``updates``) is found, and updates for those elements are uniformly
    scaled so the worst-case norm does not exceed ``max_step_size``. Different
    systems are scaled independently.

    Attributes:
        max_step_size: Maximum allowed per-element displacement norm.
    """

    max_step_size: float = field(static=True)

    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> MaxStepSizeState:
        del parameters
        return MaxStepSizeState(index_prefix=tree_copy(index_prefix))

    def update(
        self,
        updates: Params,
        state: MaxStepSizeState,
        params: Params | None = None,
        **kwargs: Any,
    ) -> tuple[Params, MaxStepSizeState]:
        del params, kwargs
        index_prefix = state.index_prefix
        if index_prefix is None:
            index_prefix = jax.tree.map(lambda x: Index.new((0,) * len(x)), updates)
        per_particle_size = jax.tree.map(
            functools.partial(jnp.linalg.norm, axis=-1), updates
        )
        max_size = tree_segment_max(per_particle_size, index_prefix)
        scale = max_size.map_data(
            lambda x: jnp.minimum(1.0, self.max_step_size / (x + 1e-12))
        )
        updates = tree_scale_per_row(updates, scale, index_prefix)
        return updates, state
