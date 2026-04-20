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

from typing import TYPE_CHECKING, NamedTuple, Protocol, Sequence, no_type_check

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.lens import Lens, View
from kups.core.patch import ComposedPatch, IdPatch, IndexLensPatch, Patch, WithPatch
from kups.core.potential import (
    EMPTY,
    EmptyType,
    Energy,
    Potential,
    PotentialOut,
    empty_patch_idx_view,
)
from kups.core.typing import HasPositionsAndSystemIndex, ParticleId, SystemId
from kups.core.unitcell import UnitCell
from kups.core.utils.jax import (
    dataclass,
    field,
    jit,
    kahan_summation,
    linearize,
    tree_structure,
)


class IsStateWithParticlesAndUnitCell(Protocol):
    @property
    def particles(self) -> Table[ParticleId, HasPositionsAndSystemIndex]: ...
    @property
    def systems(self) -> Table[SystemId, UnitCell]: ...


class PositionAndUnitCell(NamedTuple):
    positions: Table[ParticleId, Array]
    unitcell: Table[SystemId, UnitCell]


@no_type_check
def position_and_unitcell_idx_view(
    state: IsStateWithParticlesAndUnitCell,
) -> PotentialOut[PositionAndUnitCell, EmptyType]:
    return PotentialOut(
        empty_patch_idx_view(state).total_energies,
        PositionAndUnitCell(state.particles.data.system, Index.new(state.systems.keys)),
        EMPTY,
    )


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

    def __init__(self, *args: Summand[Input], add_previous_total: bool = False):
        super().__init__(args)
        self.add_previous_total = add_previous_total


class SumComposer[State, Input, StatePatch: Patch](Protocol):
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


@dataclass
class IdentityComposer[Input](SumComposer[Input, Input, Patch]):
    """Simple composer that always returns input state unchanged.

    Used for potentials without incremental update support (always full recomputation).
    """

    def __call__(self, state: Input, patch: Patch | None) -> Sum[Input]:
        if patch is not None:
            raise ValueError("IdentityComposer does not support patches.")
        return Sum(Summand(state))


# We need this union since while IdPatch is assignable to Patch[State], pyright does not
# correctly infer this when provided as a generic argument.
type EnergyAndCachePatch[State] = (
    WithPatch[Table[SystemId, Energy], Patch[State]]
    | WithPatch[Table[SystemId, Energy], IdPatch]
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
    StatePatch: Patch,
]:
    """Converts energy functions to full potentials with gradients and Hessians.

    The core building block for all potential implementations. Takes a simple
    energy function and automatically adds:

    1. Incremental updates via [SumComposer][kups.potential.common.energy.SumComposer]
    2. Gradients via automatic differentiation
    3. Hessians via forward-on-backward differentiation
    4. Caching and state patches

    Type Parameters:
        State: Simulation state type
        Input: Energy function input type
        Gradients: Gradient structure type
        Hessians: Hessian structure type
        StatePatch: Patch type for state updates

    Attributes:
        energy_fn: Energy function to wrap
        composer: Plans incremental updates from state and patch
        gradient_lens: Selects tensors for gradient computation
        hessian_lens: Selects gradients for Hessian computation
        hessian_idx_view: Extracts Hessian row/column indices from state
        cache_lens: Lens to cached potential output (optional)
        patch_idx_view: Index structure for cache updates (optional)

    Gradients are computed of total energy with respect to tensors specified by
    `gradient_lens`. Hessians are computed row-by-row via forward-on-backward mode,
    with indices specified by `hessian_idx_view`.

    Hessian indices have shape `(num_calls, 2, num_entries_per_call)`:
    - First dimension: vectorized over (parallel rows)
    - Second dimension: [row_indices, column_indices]

    Example - Computing 3×3 Hessian for first 3 particles:
    ```python
    hessian_indices = [
        [[0, 0, 0], [0, 1, 2]],  # ∂²E/∂x₀∂(x₀,x₁,x₂)
        [[1, 1, 1], [0, 1, 2]],  # ∂²E/∂x₁∂(x₀,x₁,x₂)
        [[2, 2, 2], [0, 1, 2]],  # ∂²E/∂x₂∂(x₀,x₁,x₂)
    ]
    ```

    For batched systems, parallelize over both batches and rows:
    ```python
    # Two systems with 3 particles each (particles 0-2 and 6-8)
    hessian_indices = [
        [[0, 0, 0, 6, 6, 6], [0, 1, 2, 6, 7, 8]],  # Rows 0 and 0
        [[1, 1, 1, 7, 7, 7], [0, 1, 2, 6, 7, 8]],  # Rows 1 and 1
        [[2, 2, 2, 8, 8, 8], [0, 1, 2, 6, 7, 8]],  # Rows 2 and 2
    ]
    ```

    Warning - Potentially incorrect parallelization example:
    ```python
    # WRONG: This computes d(df/dx₀ + df/dx₁)/dx instead of separate Hessian rows
    hessian_indices = [
        [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
    ]
    # Gives: d(df/dx₀ + df/dx₁)/dx₀, d(df/dx₀ + df/dx₁)/dx₁, d(df/dx₀ + df/dx₁)/dx₂
    # Expected: d²f/dx₀², d²f/dx₀dx₁, d²f/dx₀dx₂, d²f/dx₁², d²f/dx₁dx₂, d²f/dx₂²
    ```

    Each vectorized call must compute independent Hessian rows. Mixing rows
    from different particles in the same call causes gradient interference.
    """

    energy_fn: EnergyFunction[State, Input] = field(static=True)
    composer: SumComposer[State, Input, StatePatch] = field(static=True)
    gradient_lens: Lens[Input, Gradients] = field(static=True)
    hessian_lens: Lens[Gradients, Hessians] = field(static=True)
    hessian_idx_view: View[State, Hessians] = field(static=True)
    cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = field(
        static=True
    )
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = field(
        static=True
    )

    @jit
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]:
        dp_plan = self.composer(state, patch)
        assert len(dp_plan) > 0, "At least one configuration must be added."
        outs: list[PotentialOut[Gradients, Hessians]] = []
        h_idx = self.hessian_idx_view(state)
        patches: list[Patch[State]] = []
        for inp, weight in dp_plan:
            # Prepare inputs
            g_inp = self.gradient_lens.get(inp)
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
                def hessian_vec_fn(idx: Array):
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

        # If the dispatcher demands it, we add the previous total potential
        if dp_plan.add_previous_total:
            assert self.cache_lens is not None, (
                "Cache lens must be set for caching previous total potential."
            )
            outs.append(self.cache_lens.get(state))

        # Aggregate the result with Kahan summation
        total = kahan_summation(*outs)[0]
        if self.cache_lens is not None:
            assert self.patch_idx_view is not None, (
                "Patch index view must be set when cache lens is set."
            )
            cache_patch = IndexLensPatch(
                total, self.patch_idx_view(state), self.cache_lens
            )
        else:
            cache_patch = IdPatch()
        out_patch = ComposedPatch((cache_patch, *patches))
        return WithPatch(total, out_patch)


if TYPE_CHECKING:

    def __(a: PotentialFromEnergy):
        _: Potential = a
