# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Direct potential for models providing precomputed gradients.

Wraps model functions that directly produce energy and gradients (e.g., PyTorch
force fields) into the kUPS Potential protocol. Unlike PotentialFromEnergy which
uses autodiff, this passes through whatever gradients/Hessians the model provides.
"""

from typing import Protocol

from kups.core.lens import Lens, View
from kups.core.patch import ComposedPatch, IdPatch, IndexLensPatch, Patch, WithPatch
from kups.core.potential import Potential, PotentialOut
from kups.core.utils.jax import dataclass, field, jit, kahan_summation
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
    StatePatch: Patch,
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
    cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = field(
        static=True
    )
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = field(
        static=True
    )

    @jit
    def __call__(
        self, state: State, patch: StatePatch | None = None
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]:
        dp_plan = self.composer(state, patch)
        assert len(dp_plan) > 0, "At least one configuration must be added."

        outs: list[PotentialOut[Gradients, Hessians]] = []
        patches: list[Patch[State]] = []

        for inp, weight in dp_plan:
            result = self.direct_potential_fn(inp)
            outs.append(weight * result.data)
            patches.append(result.patch)

        if dp_plan.add_previous_total:
            assert self.cache_lens is not None
            outs.append(self.cache_lens.get(state))

        total = kahan_summation(*outs)[0]

        if self.cache_lens is not None:
            assert self.patch_idx_view is not None
            cache_patch = IndexLensPatch(
                total, self.patch_idx_view(state), self.cache_lens
            )
        else:
            cache_patch = IdPatch()

        out_patch = ComposedPatch((cache_patch, *patches))
        return WithPatch(total, out_patch)
