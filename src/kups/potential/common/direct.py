# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Direct potential for models providing precomputed gradients.

Wraps model functions that directly produce energy and gradients (e.g., PyTorch
force fields) into the kUPS Potential protocol. Unlike PotentialFromEnergy which
uses autodiff, this passes through whatever gradients/Hessians the model provides.
"""

from typing import Any, Literal, Protocol, overload

from kups.core.lens import Lens, View
from kups.core.patch import ComposedPatch, IdPatch, IndexLensPatch, Patch, WithPatch
from kups.core.potential import (
    CompensatedPotentialResult,
    Potential,
    PotentialOut,
    PotentialResult,
)
from kups.core.utils.jax import dataclass, field, jit
from kups.core.utils.kahan import KahanSummand
from kups.potential.common.energy import SumComposer


class DirectPotentialFunction[State, Input, Gradients, Hessians](Protocol):
    """Protocol for functions returning PotentialOut directly."""

    def __call__(
        self, inp: Input
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]: ...


@dataclass
class DirectPotential[
    State,
    Input,
    Gradients,
    Hessians,
    StatePatch: Patch[Any],
](Potential[State, Gradients, Hessians, StatePatch]):
    """Potential wrapping models that directly produce gradients.

    For models providing precomputed gradients (e.g., PyTorch force fields).
    Passes through whatever gradients and Hessians the model provides
    (typically Hessians=EmptyType).
    """

    direct_potential_fn: DirectPotentialFunction[State, Input, Gradients, Hessians] = (
        field(static=True)
    )
    composer: SumComposer[State, Input, StatePatch] = field(static=True)
    cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]] | None = (
        field(static=True)
    )
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = field(
        static=True
    )

    @overload
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
        *,
        include_compensate: Literal[False] = False,
    ) -> PotentialResult[State, Gradients, Hessians]: ...
    @overload
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
        *,
        include_compensate: Literal[True],
    ) -> CompensatedPotentialResult[State, Gradients, Hessians]: ...
    @jit(static_argnames="include_compensate")
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
        *,
        include_compensate: bool = False,
    ) -> (
        PotentialResult[State, Gradients, Hessians]
        | CompensatedPotentialResult[State, Gradients, Hessians]
    ):
        """Evaluate the model and accumulate the composed inputs.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates
            include_compensate: Return the accumulator rather than its
                compensated total

        Returns:
            Potential output with the cache update patch composed
        """
        dp_plan = self.composer(state, patch)
        assert len(dp_plan) > 0, "At least one configuration must be added."

        outs: list[PotentialOut[Gradients, Hessians]] = []
        patches: list[Patch[State]] = []

        for inp, weight in dp_plan:
            result = self.direct_potential_fn(inp)
            outs.append(weight * result.data)
            patches.append(result.patch)

        # Fold the weighted deltas into the cached running summand, carrying the
        # Kahan compensation across calls. A full recompute starts a fresh summand.
        if dp_plan.add_previous_total:
            assert self.cache_lens is not None
            summand = self.cache_lens.get(state)
            for out in outs:
                summand = summand + out
        else:
            summand = KahanSummand.init(outs[0])
            for out in outs[1:]:
                summand = summand + out

        if self.cache_lens is not None:
            assert self.patch_idx_view is not None
            idx = self.patch_idx_view(state)
            cache_patch = IndexLensPatch(
                summand, KahanSummand(idx, idx), self.cache_lens
            )
        else:
            cache_patch = IdPatch[State]()

        out_patch = ComposedPatch((cache_patch, *patches))
        if include_compensate:
            return WithPatch(summand, out_patch)
        return WithPatch(summand.total, out_patch)
