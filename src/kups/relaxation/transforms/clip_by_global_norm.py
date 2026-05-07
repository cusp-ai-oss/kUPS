# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system L2-norm clipping transform.

Per-system analogue of :func:`optax.clip_by_global_norm`. For every system,
the L2 norm of every update entry assigned to that system (across all
leaves of the parameter pytree) is computed; entries are then uniformly
rescaled so that per-system L2 norm does not exceed ``max_norm``.
Different systems are clipped independently, so a batched run is
bit-identical to running each system one at a time.

For per-particle (rather than per-system) caps, see
:class:`kups.relaxation.transforms.MaxStepSize` — that constrains the
*largest single particle's* displacement, not the system's total.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from kups.core.data.index import Index
from kups.core.typing import PyTree
from kups.core.utils.jax import dataclass, field, tree_copy
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.transforms._segmented_tree import (
    tree_scale_per_row,
    tree_segment_norm,
)


@dataclass
class ClipByGlobalNormState:
    """State carrying the ``index_prefix`` captured at init time.

    Attributes:
        index_prefix: Tree prefix of the parameter pytree whose leaves are
            ``Index[K]`` objects, or ``None`` to clip with a single global
            (cross-system) L2 norm.
    """

    index_prefix: PyTree | None


@dataclass
class ClipByGlobalNorm[Params](Optimizer[Params, ClipByGlobalNormState]):
    """Clip the per-system L2 norm of updates to ``max_norm``.

    With ``index_prefix=None`` this reduces to the standard
    :func:`optax.clip_by_global_norm` (a single tree-global L2 norm).

    Attributes:
        max_norm: Maximum allowed per-system L2 norm of the update.
    """

    max_norm: float = field(static=True)

    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> ClipByGlobalNormState:
        del parameters
        return ClipByGlobalNormState(index_prefix=tree_copy(index_prefix))

    def update(
        self,
        updates: Params,
        state: ClipByGlobalNormState,
        params: Params | None = None,
        **kwargs: Any,
    ) -> tuple[Params, ClipByGlobalNormState]:
        del params, kwargs
        index_prefix = state.index_prefix
        if index_prefix is None:
            index_prefix = jax.tree.map(lambda x: Index.new((0,) * len(x)), updates)
        norm = tree_segment_norm(updates, index_prefix)
        scale = norm.map_data(lambda x: jnp.minimum(1.0, self.max_norm / (x + 1e-12)))
        return tree_scale_per_row(updates, scale, index_prefix), state
