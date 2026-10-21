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
from kups.relaxation.transforms.fire import (
    FireReset,
    FireResetData,
    FireResetIndices,
    ScaleByFire,
    ScaleByFireState,
    fire_reset_layout,
)
from kups.relaxation.transforms.fire2 import (
    Fire2Reset,
    Fire2ResetData,
    Fire2ResetIndices,
    ScaleByFire2,
    ScaleByFire2State,
    fire2_reset_layout,
)
from kups.relaxation.transforms.lbfgs import (
    LbfgsReset,
    LbfgsResetData,
    LbfgsResetIndices,
    ScaleByAseLbfgs,
    ScaleByAseLbfgsState,
    lbfgs_reset_layout,
)
from kups.relaxation.transforms.linesearch import (
    LineSearchState,
    ScaleByBacktrackingLinesearch,
    ScaleByMoreThuenteLinesearch,
    linesearch_reset_layout,
)
from kups.relaxation.transforms.max_step_size import (
    MaxStepSize,
    MaxStepSizeState,
)

__all__ = [
    "ClipByGlobalNorm",
    "ClipByGlobalNormState",
    "Fire2Reset",
    "Fire2ResetData",
    "Fire2ResetIndices",
    "FireReset",
    "FireResetData",
    "FireResetIndices",
    "LbfgsReset",
    "LbfgsResetData",
    "LbfgsResetIndices",
    "LineSearchState",
    "MaxStepSize",
    "MaxStepSizeState",
    "ScaleByAseLbfgs",
    "ScaleByAseLbfgsState",
    "ScaleByBacktrackingLinesearch",
    "ScaleByFire",
    "ScaleByFire2",
    "ScaleByFire2State",
    "ScaleByFireState",
    "ScaleByMoreThuenteLinesearch",
    "fire2_reset_layout",
    "fire_reset_layout",
    "lbfgs_reset_layout",
    "linesearch_reset_layout",
]
