# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Energy computation patterns with incremental updates.

This module provides the infrastructure for converting energy functions into full
potentials with gradients, Hessians, and efficient incremental updates. The key
abstraction is the composition pattern that enables reusing previous computations
during Monte Carlo moves.

Key components:

- **[EnergyFunction][kups.potential.common.energy.EnergyFunction]**: Protocol for simple energy functions
- **[PotentialFromEnergy][kups.potential.common.energy.PotentialFromEnergy]**: Converts energy functions to full potentials
- **[SumComposer][kups.potential.common.energy.SumComposer]**: Plans incremental energy updates
The composition pattern allows efficient Monte Carlo by computing only energy
differences rather than full recomputation (e.g., subtract old particle contribution,
add new particle contribution, reuse rest).
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, overload

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.patch import ComposedPatch, IdPatch, IndexLensPatch, Patch, WithPatch
from kups.core.potential import (
    CompensatedPotentialResult,
    Energy,
    Potential,
    PotentialOut,
    PotentialResult,
)
from kups.core.typing import SystemId
from kups.core.utils.jax import (
    dataclass,
    field,
    jit,
    linearize,
    tree_structure,
)
from kups.core.utils.kahan import KahanSummand


class Summand[Input](NamedTuple):
    """Weighted input configuration for energy computation.

    Attributes:
        inp: Input configuration
        weight: Multiplicative weight (typically 1 or -1 for add/subtract)
    """

    inp: Input
    weight: float = 1


class Sum[Input](list[Summand[Input]]):
    """Sequence of weighted configurations for incremental energy updates.

    Represents a plan for computing energy changes efficiently. For example,
    when moving a particle: subtract old contribution (weight=-1), add new
    contribution (weight=1), optionally reuse cached total.

    The `add_previous_total` flag enables reusing previous full energy calculations,
    crucial for incremental updates.

    Attributes:
        add_previous_total: Whether to include previous total energy in plan
    """

    def __init__(self, *args: Summand[Input], add_previous_total: bool = False) -> None:
        super().__init__(args)
        self.add_previous_total = add_previous_total


class SumComposer[State, Input, StatePatch: Patch[Any]](Protocol):
    """Protocol for generating incremental energy update plans.

    Given a state and proposed patch, returns a sum of weighted configurations
    to compute efficiently. Enables O(k) updates instead of O(N) recomputation
    for Monte Carlo moves affecting k particles.

    Example plan for moving a particle:
    1. Subtract energy of old configuration (weight=-1)
    2. Add energy of new configuration (weight=1)
    3. Reuse previous total (add_previous_total=True)

    Type Parameters:
        State: Simulation state type
        Input: Energy function input type
        StatePatch: Patch type for state updates
    """

    def __call__(self, state: State, patch: StatePatch | None) -> Sum[Input]:
        """Generate incremental update plan.

        Args:
            state: Current simulation state
            patch: Proposed changes (or None for full computation)

        Returns:
            Sum of weighted configurations to evaluate
        """
        ...


class InputConstructor[State, Input, StatePatch: Patch[Any]](Protocol):
    """Construct a full input, or an old/new input affected by a proposal."""

    def __call__(
        self,
        state: State,
        patch: StatePatch | None,
        old_input: bool = False,
    ) -> Input: ...


@dataclass
class LocalSumComposer[State, Input, StatePatch: Patch[Any]]:
    """Shared full/old/new plan for local energies, regardless of representation.

    ``difference`` can combine previous and proposed inputs into one input
    whose energy is proposed minus previous, sharing a single evaluator call.
    """

    constructor: InputConstructor[State, Input, StatePatch] = field(static=True)
    difference: Callable[[Input, Input], Input] | None = field(
        static=True, default=None, kw_only=True
    )

    def __call__(self, state: State, patch: StatePatch | None) -> Sum[Input]:
        if patch is None:
            return Sum(Summand(self.constructor(state, None)))
        previous = self.constructor(state, patch, old_input=True)
        proposed = self.constructor(state, patch)
        if self.difference is not None:
            return Sum(
                Summand(self.difference(previous, proposed)), add_previous_total=True
            )
        return Sum(
            Summand(previous, -1),
            Summand(proposed, 1),
            add_previous_total=True,
        )


@dataclass
class FullSumComposer[State, Input, StatePatch: Patch[Any]]:
    """Construct one full input for the proposed state."""

    constructor: InputConstructor[State, Input, StatePatch] = field(static=True)

    def __call__(self, state: State, patch: StatePatch | None) -> Sum[Input]:
        return Sum(Summand(self.constructor(state, patch)))


@dataclass
class IdentityComposer[Input](SumComposer[Input, Input, Patch[Any]]):
    """Simple composer that always returns input state unchanged.

    Used for potentials without incremental update support (always full recomputation).
    """

    def __call__(self, state: Input, patch: Patch[Any] | None) -> Sum[Input]:
        if patch is not None:
            raise ValueError("IdentityComposer does not support patches.")
        return Sum(Summand(state))


# We need this union since while IdPatch is assignable to Patch[State], pyright does not
# correctly infer this when provided as a generic argument.
type EnergyAndCachePatch[State] = (
    WithPatch[Table[SystemId, Energy], Patch[State]]
    | WithPatch[Table[SystemId, Energy], IdPatch[State]]
)


class EnergyFunction[State, Input](Protocol):
    """Protocol for functions computing energy from graph inputs.

    Type Parameters:
        State: Simulation state type
        Input: Graph input type (e.g., GraphPotentialInput)
    """

    def __call__(self, inp: Input) -> EnergyAndCachePatch[State]:
        """Compute energy from input.

        Args:
            inp: Graph potential input

        Returns:
            Energy and optional state patch
        """
        ...


@dataclass
class PotentialFromEnergy[
    State,
    Input,
    Gradients,
    Hessians,
    StatePatch: Patch[Any],
]:
    """Converts energy functions to full potentials with gradients and Hessians.

    Wraps an energy function with incremental updates via
    [SumComposer][kups.potential.common.energy.SumComposer], gradients of the
    total energy by autodiff, Hessian rows by forward-on-backward
    differentiation, and optional caching with state patches.

    Hessian indices have shape `(num_calls, 2, num_entries_per_call)`, holding
    row and column indices per vectorized call. One call may cover rows of
    different systems, but each call must contain a single row per system;
    mixing rows of one system in a call differentiates their summed gradient.
    For two systems of 3 particles (0-2 and 6-8), the first call is
    ``[[0, 0, 0, 6, 6, 6], [0, 1, 2, 6, 7, 8]]``.

    Attributes:
        energy_fn: Energy function to wrap
        composer: Plans incremental updates from state and patch
        gradient_lens: Selects tensors for gradient computation
        hessian_lens: Selects gradients for Hessian computation
        hessian_idx_view: Extracts Hessian row/column indices from state
        cache_lens: Lens to cached potential output (optional)
        patch_idx_view: Index structure for cache updates (optional)
    """

    energy_fn: EnergyFunction[State, Input] = field(static=True)
    composer: SumComposer[State, Input, StatePatch] = field(static=True)
    gradient_lens: Lens[Input, Gradients] = field(static=True)
    hessian_lens: Lens[Gradients, Hessians] = field(static=True)
    hessian_idx_view: View[State, Hessians] = field(static=True)
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
        """Compute energy, gradients and Hessians for the composed inputs.

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
        h_idx = self.hessian_idx_view(state)
        patches: list[Patch[State]] = []
        for inp, weight in dp_plan:
            # Prepare inputs
            g_inp = self.gradient_lens.get(inp)
            if not jax.tree.leaves(g_inp):
                # No differentiation targets: evaluate the energy directly and
                # skip the linearize/VJP scaffolding entirely.
                energy_result = self.energy_fn(inp)
                out = PotentialOut(
                    energy_result.data, g_inp, self.hessian_lens.get(g_inp)
                )
                outs.append(weight * out)
                patches.append(energy_result.patch)
                continue
            h_inp = self.hessian_lens.get(g_inp)
            h_tree = tree_structure(h_inp)
            h_inp_list = h_tree.flatten_up_to(h_inp)

            # We need two nested functions here, the outer one is for Hessians
            # which must be a subset of the gradients.
            # We compute the Hessian row by row via forward on backward differentiation.
            def _potential_and_grad(
                h_inp_list: Sequence[Array],
            ) -> tuple[list[Array], tuple[Gradients, EnergyAndCachePatch[State]]]:
                # The inner function is for computing the potential and gradients.
                def _potential(
                    g_inp: Gradients,
                ) -> tuple[Table[SystemId, Energy], EnergyAndCachePatch[State]]:
                    patched_input = self.gradient_lens.set(inp, g_inp)
                    energy_out = self.energy_fn(patched_input)
                    return energy_out.data, energy_out

                h_inp = h_tree.unflatten(h_inp_list)
                patched_g_inp = self.hessian_lens.set(g_inp, h_inp)
                energies, vjp_fn, energy_result = jax.vjp(
                    _potential, patched_g_inp, has_aux=True
                )
                gradients = vjp_fn(jax.tree.map(jnp.ones_like, energies))[0]
                flat_out = h_tree.flatten_up_to(self.hessian_lens.get(gradients))
                return flat_out, (gradients, energy_result)

            # Energy and gradients computation + Hessian linearization
            _, hessian_row_fn, (gradients, energy_result) = linearize(
                _potential_and_grad, h_inp_list, has_aux=True
            )

            # Hessian computation
            h_vals: list[Array] = []
            for i, idx in enumerate(h_tree.flatten_up_to(h_idx)):

                @jax.vmap
                def hessian_vec_fn(idx: Array) -> Array:
                    a, b = idx
                    tangent = list(map(jnp.zeros_like, h_inp_list))
                    # Set a 1 to the specific element in the flattened input
                    tangent[i] = (
                        tangent[i]
                        .ravel()
                        .at[a]
                        .set(1, mode="drop")
                        .reshape(tangent[i].shape)
                    )
                    out = hessian_row_fn(tangent)[i]
                    return out.ravel().at[b].get(mode="fill", fill_value=0)

                h_vals.append(hessian_vec_fn(idx))
            hessians = h_tree.unflatten(h_vals)

            # Output
            out = PotentialOut(energy_result.data, gradients, hessians)
            outs.append(weight * out)
            patches.append(energy_result.patch)

        # Fold the weighted deltas into the cached running summand, carrying the
        # Kahan compensation across calls. A full recompute starts a fresh summand.
        if dp_plan.add_previous_total:
            assert self.cache_lens is not None, (
                "Cache lens must be set for caching previous total potential."
            )
            summand = self.cache_lens.get(state)
            for out in outs:
                summand = summand + out
        else:
            summand = KahanSummand.init(outs[0])
            for out in outs[1:]:
                summand = summand + out

        if self.cache_lens is not None:
            assert self.patch_idx_view is not None, (
                "Patch index view must be set when cache lens is set."
            )
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


if TYPE_CHECKING:

    def __(a: PotentialFromEnergy[Any, Any, Any, Any, Patch[Any]]) -> None:
        _: Potential[Any, Any, Any, Patch[Any]] = a
