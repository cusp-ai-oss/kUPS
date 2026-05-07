# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system relaxation transforms compatible with the
:class:`kups.relaxation.optimizer.Optimizer` protocol.

These are batch-aware versions of the transforms in
:mod:`kups.relaxation.optax`: they accept an ``index_prefix`` pytree at
``init`` time identifying which system each element belongs to, so batched
systems are clipped or scaled independently.
"""

from kups.relaxation.transforms.clip_by_global_norm import (
    ClipByGlobalNorm,
    ClipByGlobalNormState,
)
from kups.relaxation.transforms.fire import ScaleByFire, ScaleByFireState
from kups.relaxation.transforms.fire2 import ScaleByFire2, ScaleByFire2State
from kups.relaxation.transforms.lbfgs import (
    ScaleByAseLbfgs,
    ScaleByAseLbfgsState,
)
from kups.relaxation.transforms.max_step_size import (
    MaxStepSize,
    MaxStepSizeState,
)

__all__ = [
    "ClipByGlobalNorm",
    "ClipByGlobalNormState",
    "MaxStepSize",
    "MaxStepSizeState",
    "ScaleByAseLbfgs",
    "ScaleByAseLbfgsState",
    "ScaleByFire",
    "ScaleByFire2",
    "ScaleByFire2State",
    "ScaleByFireState",
]
