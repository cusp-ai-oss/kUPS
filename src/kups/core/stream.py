# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Host refill requests using the existing assertion/fix boundary."""

from jax import Array

from kups.core.assertion import Fix, runtime_assert
from kups.core.data.table import Table
from kups.core.lens import View
from kups.core.propagator import Propagator
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field


@dataclass
class RefillPropagator[State](Propagator[State]):
    """Request host service for selected systems before advancing a batch.

    Compose this BEFORE a ResetOnErrorPropagator wrapping the numerical step.
    A pending request prevents that step from committing; propagate_and_fix
    calls refill on the returned state, then retries. The fix must clear the
    serviced requests, including slots left empty at end of input.

    Payloads stay in state. Only an empty tuple travels through fix_args,
    because assertions in compiled loops reduce their arguments by maximum.
    Use one such composition per host cycle: refill is ordinary host work,
    not an unbounded stream inside propagate_and_fix's bounded retry loop.
    The numerical work may be a bounded LoopPropagator block. Set its repetition
    count to zero while requests are pending to avoid work on repair-only attempts.
    """

    requested: View[State, Table[SystemId, Array]] = field(static=True)
    refill: Fix[State, Table[SystemId, Array]] = field(static=True)

    def _fix(self, state: State, unused: tuple[()]) -> State:
        return self.refill(state, self.requested(state))

    def __call__(self, key: Array, state: State) -> State:
        del key
        runtime_assert(
            ~self.requested(state).data.any(),
            "Batch slots require refill.",
            fix_fn=self._fix,
            fix_args=(),
        )
        return state
