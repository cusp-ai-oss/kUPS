# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Potential energy calculations with gradients and Hessians.

This module provides the infrastructure for computing potential energies and their
derivatives in molecular simulations. Potentials are composable and can be cached
for efficient evaluation.

Key components:
- **[PotentialOut][kups.core.potential.PotentialOut]**: Container for energy, gradients, and Hessians
- **[Potential][kups.core.potential.Potential]**: Protocol for energy computation with optional state patches
- **[SummedPotential][kups.core.potential.SummedPotential]**: Compose multiple potentials by summation
- **[CachedPotential][kups.core.potential.CachedPotential]**: Cache potential outputs
- **[ScaledPotential][kups.core.potential.ScaledPotential]**: Scale a potential by a constant factor

Potentials support linearity: energies, gradients, and Hessians can be summed,
enabling modular force field composition (e.g., bonded + non-bonded + Coulomb).

Every potential can also report the Kahan compensation of its accumulated total
via ``include_compensate=True``, see
[Potential][kups.core.potential.Potential].
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple, Protocol, overload

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.lens import Lens, View, bind, const_lens
from kups.core.patch import ComposedPatch, IndexLensPatch, Patch, WithPatch
from kups.core.propagator import Propagator
from kups.core.sharding import shard_axis
from kups.core.typing import (
    HasParticles,
    HasPositionsAndSystemIndex,
    OriginDeviceId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field, tree_map
from kups.core.utils.kahan import KahanSummand

type Energy = Array
"""Type alias for energy arrays, typically shape (n_systems,)."""


@dataclass
class EmptyType:
    """Sentinel type indicating empty gradients or Hessians.

    Use this when a potential does not compute gradients or Hessians,
    rather than None, to maintain type safety.
    """


EMPTY: EmptyType = EmptyType()
"""Singleton instance of EmptyType.

Use this instead of constructing EmptyType() directly.
"""

EMPTY_LENS: Lens[Any, EmptyType] = const_lens(EMPTY)
"""Lens that always returns EMPTY, ignoring input. Set is a no-op.

This is useful for potentials that don't compute gradients or Hessians.
"""


type IsStateWithParticles = HasParticles[HasPositionsAndSystemIndex]


def empty_patch_idx_view(
    state: IsStateWithParticles,
) -> PotentialOut[EmptyType, EmptyType]:
    """Default patch index view covering all systems with no gradient/Hessian outputs."""
    system_keys = state.particles.data.system.keys
    return PotentialOut(Index.new(system_keys), EMPTY, EMPTY)  # type: ignore


@dataclass
class PotentialOut[Gradients, Hessians]:
    """Output of a potential energy calculation.

    Contains the total energy per system, gradients with respect to specified
    tensors (e.g., positions, charges), and optionally Hessians (second derivatives).

    Assumes **linearity**: energies, gradients, and Hessians can be summed,
    enabling composition of multiple potentials via [SummedPotential][kups.core.potential.SummedPotential]
    (e.g., U_total = U_bonded + U_vdw + U_elec).

    Type Parameters:
        Gradients: PyTree structure containing first derivatives
        Hessians: PyTree structure containing second derivatives (subset of gradients)

    Attributes:
        total_energies: Total energy per system as a `Table[SystemId, Energy]`
        gradients: First derivatives (e.g., forces = -∇U)
        hessians: Second derivatives (e.g., for normal mode analysis)

    Example:
        ```python
        # Simple potential output with position gradients only
        out = PotentialOut(
            total_energies=jnp.array([10.5, 12.3]),  # 2 systems
            gradients={"positions": force_array},     # Forces on particles
            hessians=EMPTY                            # No Hessians computed
        )

        # Combine potentials
        total = lj_out + coulomb_out  # Element-wise addition
        ```
    """

    total_energies: Table[SystemId, Energy]
    gradients: Gradients
    hessians: Hessians

    def __add__(
        self, other: PotentialOut[Gradients, Hessians]
    ) -> PotentialOut[Gradients, Hessians]:
        return jax.tree.map(jnp.add, self, other)

    def __sub__(
        self, other: PotentialOut[Gradients, Hessians]
    ) -> PotentialOut[Gradients, Hessians]:
        return jax.tree.map(jnp.subtract, self, other)

    def __mul__(self, other: float) -> PotentialOut[Gradients, Hessians]:
        return jax.tree.map(lambda x: other * x, self)

    def __rmul__(self, other: float) -> PotentialOut[Gradients, Hessians]:
        return jax.tree.map(lambda x: other * x, self)

    @property
    def as_tuple(self) -> tuple[Table[SystemId, Energy], Gradients, Hessians]:
        """Convert to tuple form (energies, gradients, hessians).

        Returns:
            Tuple of (total_energies, gradients, hessians)
        """
        return self.total_energies, self.gradients, self.hessians


type PotentialResult[State, Gradients, Hessians] = WithPatch[
    PotentialOut[Gradients, Hessians], Patch[State]
]
"""Potential output paired with the patch that commits it to the state."""

type CompensatedPotentialResult[State, Gradients, Hessians] = WithPatch[
    KahanSummand[PotentialOut[Gradients, Hessians]], Patch[State]
]
"""Potential output as an accumulator carrying the compensation of its total."""


class Potential[
    State,
    Gradients,
    Hessians,
    StatePatch: Patch[Any],
](Protocol):
    """Protocol for potential energy functions.

    A potential computes energy, gradients, and optionally Hessians for a given
    simulation state. Potentials can optionally accept a state patch describing
    recent changes, enabling efficient incremental updates.

    Type Parameters:
        State: Simulation state type
        Gradients: Structure of first derivatives
        Hessians: Structure of second derivatives (subset of gradients)
        StatePatch: Type of state modification patches

    The `patch` argument enables incremental computation:
        - Monte Carlo: Only recompute for moved particles
        - Molecular dynamics: Reuse neighbor lists
        - General: Avoid redundant calculations

    Incremental updates accumulate small energy changes onto a large cached total,
    which loses low-order bits in single precision. Potentials therefore keep a
    [KahanSummand][kups.core.utils.kahan.KahanSummand] internally and expose its
    compensation via ``include_compensate=True``. Differencing two compensated
    totals (see
    [KahanSummand.difference][kups.core.utils.kahan.KahanSummand.difference])
    recovers the energy change exactly, which matters for Monte Carlo acceptance
    where the change is far smaller than the total.

    Example:
        ```python
        class LennardJonesPotential:
            def __call__(self, state, patch=None, *, include_compensate=False):
                # Compute LJ energy and forces
                energy = compute_lj_energy(state.positions)
                forces = compute_lj_forces(state.positions)

                out = PotentialOut(energy, {"positions": forces}, EMPTY)
                if include_compensate:
                    out = KahanSummand.init(out)  # Nothing accumulated, so zero
                return WithPatch(out, IdPatch())  # No state caching needed

        # Use in simulation
        potential = LennardJonesPotential()
        result = potential(state)
        energy = result.data.total_energies
        forces = result.data.gradients.positions
        ```
    """

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
        """Compute potential energy and derivatives.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates
            include_compensate: Return the output as a
                [KahanSummand][kups.core.utils.kahan.KahanSummand] carrying the
                rounding error accumulated so far. The compensation is zero
                unless the potential accumulates onto a cached total.

        Returns:
            Potential output and state patch
        """
        ...


@dataclass
class SummedPotential[State, Gradients, Hessians, StatePatch: Patch[Any]](
    Potential[State, Gradients, Hessians, StatePatch]
):
    """Compose multiple potentials by summing their outputs.

    Enables modular force field composition where total energy is the sum of
    individual contributions (e.g., bonded + Lennard-Jones + Coulomb).

    Type Parameters:
        State: Simulation state type
        Gradients: Gradient structure type
        Hessians: Hessian structure type
        StatePatch: State patch type

    Attributes:
        potentials: Tuple of potentials to sum (must have at least one)

    Example:
        ```python
        # Compose a force field
        total_potential = sum_potentials(
            bonded_potential,
            lennard_jones_potential,
            coulomb_potential
        )

        # Compute total energy and forces
        result = total_potential(state)
        # result.data.total_energies = E_bonded + E_lj + E_coul
        # result.data.gradients = ∇E_bonded + ∇E_lj + ∇E_coul
        ```
    """

    potentials: tuple[Potential[State, Gradients, Hessians, StatePatch], ...] = field(
        static=True
    )

    def __post_init__(self) -> None:
        if len(self.potentials) == 0:
            raise ValueError("At least one potential must be provided")

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
        """Evaluate all potentials and sum their outputs.

        Calls each potential in sequence with the same state and patch, then
        sums the resulting energies, gradients, and Hessians element-wise.
        Patches are composed in order. The summands are always compensated so
        that the terms are added with Kahan summation.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates
            include_compensate: Return the accumulator rather than its
                compensated total

        Returns:
            Combined potential output with composed patches
        """
        outs = [s(state, patch, include_compensate=True) for s in self.potentials]
        # Sum using WithPatch.__add__ (adds data and composes patches)
        total = sum(outs[1:], outs[0])
        if include_compensate:
            return total
        return total.map_data(lambda x: x.total)


def sum_potentials[State, Gradients, Hessians, StatePatch: Patch[Any]](
    *potentials: Potential[State, Gradients, Hessians, StatePatch],
) -> Potential[State, Gradients, Hessians, StatePatch]:
    """Compose multiple potentials by summing their outputs.

    Args:
        potentials: Potentials to sum.

    Returns:
        A single potential producing the summed output.

    Raises:
        ValueError: If no potentials are provided.
    """
    return SummedPotential(potentials)


@dataclass
class ScaledPotential[State, Gradients, Hessians, StatePatch: Patch[Any]](
    Potential[State, Gradients, Hessians, StatePatch]
):
    """Scale a potential's output by a constant factor.

    Multiplies energies, gradients, and Hessians by a scalar. Useful for
    thermodynamic integration, replica exchange, or applying coupling parameters.

    Attributes:
        potential: Base potential to scale
        scale: Multiplicative factor (lambda in thermodynamic integration)

    Example:
        ```python
        # Thermodynamic integration: lambda = 0 (non-interacting) to lambda = 1 (full)
        scaled_lj = ScaledPotential(lj_potential, scale=0.5)

        # Energy is scaled: E_scaled = 0.5 * E_lj
        result = scaled_lj(state)
        ```
    """

    potential: Potential[State, Gradients, Hessians, StatePatch] = field(static=True)
    scale: float = field(static=True)

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
        """Evaluate potential and scale the output.

        Computes the base potential then multiplies energies, gradients, and
        Hessians by the scale factor. The compensation is scaled along with the
        value. The patch is passed through unchanged.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates
            include_compensate: Return the accumulator rather than its
                compensated total

        Returns:
            Scaled potential output with original patch
        """
        out = self.potential(state, patch, include_compensate=True)
        out = bind(out).focus(lambda x: x.data).apply(lambda x: x * self.scale)
        if include_compensate:
            return out
        return out.map_data(lambda x: x.total)


@dataclass
class CachedPotential[State, Gradients, Hessians, StatePatch: Patch[Any]](
    Potential[State, Gradients, Hessians, StatePatch]
):
    """Wrap a potential with caching for efficient incremental updates.

    Caches the potential output in the state and updates it via patches. Crucial
    for Monte Carlo simulations where only small perturbations are made and you
    want to avoid recomputing the entire potential.

    Attributes:
        potential: Base potential to wrap
        cache: Lens to the cache location in state
        patch_idx_view: Maps acceptance mask indices to cached structure.
            If ``None``, all-zero indices are used.
        compensate_cache: Lens to a second location holding the Kahan
            compensation of the cached output. Required for potentials that
            accumulate onto the cached total, so that differencing two cached
            totals stays exact; ``None`` for potentials that recompute from
            scratch, where the compensation is never read back.

    The patch_idx_view provides the indexing structure matching the potential
    output, used to selectively update cached values based on acceptance masks.

    Example:
        ```python
        # Cache LJ potential for MC simulation
        cached_lj = CachedPotential(
            potential=lj_potential,
            cache=lens(lambda s: s.lj_cache),
            patch_idx_view=lambda s: s.particle_indices
        )

        # First call computes and caches
        result = cached_lj(state, patch=None)
        state = result.patch(state, accept_mask)

        # The previous value can be easily accessed
        result = cached_lj.cached_value(state)
        ```
    """

    potential: Potential[State, Gradients, Hessians, StatePatch] = field(static=True)
    cache: Lens[State, PotentialOut[Any, Any]] = field(static=True)
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = field(
        static=True, default=None
    )
    compensate_cache: Lens[State, PotentialOut[Any, Any]] | None = field(
        static=True, default=None
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
        """Evaluate potential and update cache.

        Computes the base potential, then creates a patch that will update the
        cached value when applied with an acceptance mask. The cache update uses
        the patch_idx_view to determine which cached entries to modify. When
        ``compensate_cache`` is set, the compensation is committed alongside the
        value so that the next call can difference against a compensated total.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates
            include_compensate: Return the accumulator rather than its
                compensated total

        Returns:
            Potential output with cache update patch composed
        """
        result = self.potential(state, patch, include_compensate=True)
        summand = result.data
        if self.patch_idx_view is not None:
            patch_idx = self.patch_idx_view(state)
        else:
            assert len(summand.value.total_energies) == 1, (
                "patch_idx_view must be provided for multi-system potentials"
            )
            sys_keys = summand.value.total_energies.keys
            patch_idx = tree_map(
                lambda x: Index(sys_keys, jnp.zeros(x.shape, dtype=int)), summand.value
            )
        patches: list[Patch[State]] = [
            result.patch,
            IndexLensPatch(summand.value, patch_idx, self.cache),
        ]
        if self.compensate_cache is not None:
            patches.append(
                IndexLensPatch(summand.compensate, patch_idx, self.compensate_cache)
            )
        out_patch = ComposedPatch(tuple(patches))
        if include_compensate:
            return WithPatch(summand, out_patch)
        return WithPatch(summand.total, out_patch)

    def cached_value(
        self, state: State
    ) -> KahanSummand[PotentialOut[Gradients, Hessians]]:
        """Retrieve the cached potential output from state.

        Args:
            state: Simulation state containing cached values

        Returns:
            Previously computed and cached potential output. The compensation is
            the one accumulated up to that point, or zero when no
            ``compensate_cache`` is configured.
        """
        value = self.cache.get(state)
        if self.compensate_cache is None:
            return KahanSummand.init(value)
        return KahanSummand(value, self.compensate_cache.get(state))


class LinearMappedPotentialInput[State, InGrad, InHess](NamedTuple):
    state: State
    potential_out: PotentialOut[InGrad, InHess]


@dataclass
class LinearMappedPotential[
    State,
    InGrad,
    OutGrad,
    InHess,
    OutHess,
    StatePatch: Patch[Any],
](Potential[State, OutGrad, OutHess, StatePatch]):
    """Wrap a potential and transform its gradient and hessian outputs.

    Applies mapping functions to gradients and hessians returned by the inner
    potential, enabling projection (e.g., extracting position gradients from a
    combined position+lattice gradient structure).

    The mapping must be linear: it is applied to the running sum and to the
    compensation separately, so a cached accumulator survives the mapping.
    Projections and weighted sums of the output qualify, anything nonlinear does
    not.

    Attributes:
        potential: Base potential to wrap
        gradient_map: Function to transform gradients from InGrad to OutGrad
        hessian_map: Function to transform hessians from InHess to OutHess

    Example:
        ```python
        # Extract position gradients from VirialTheoremGradients
        position_potential = LinearMappedPotential(
            potential=full_potential,  # Returns VirialTheoremGradients
            gradient_map=lambda g: g.positions,
            hessian_map=lambda h: h,  # Pass through hessians unchanged
        )

        result = position_potential(state)
        # result.data.gradients is now just the position array
        ```
    """

    potential: Potential[State, InGrad, InHess, StatePatch] = field(static=True)
    mapping: View[
        LinearMappedPotentialInput[State, InGrad, InHess],
        PotentialOut[OutGrad, OutHess],
    ] = field(static=True)

    @overload
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
        *,
        include_compensate: Literal[False] = False,
    ) -> PotentialResult[State, OutGrad, OutHess]: ...
    @overload
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
        *,
        include_compensate: Literal[True],
    ) -> CompensatedPotentialResult[State, OutGrad, OutHess]: ...
    def __call__(
        self,
        state: State,
        patch: StatePatch | None = None,
        *,
        include_compensate: bool = False,
    ) -> (
        PotentialResult[State, OutGrad, OutHess]
        | CompensatedPotentialResult[State, OutGrad, OutHess]
    ):
        """Evaluate the wrapped potential and map its output.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates
            include_compensate: Return the accumulator rather than its
                compensated total

        Returns:
            Mapped potential output with the original patch
        """
        result = self.potential(state, patch, include_compensate=True)
        mapped = KahanSummand(
            self.mapping(LinearMappedPotentialInput(state, result.data.value)),
            self.mapping(LinearMappedPotentialInput(state, result.data.compensate)),
        )
        if include_compensate:
            return WithPatch(mapped, result.patch)
        return WithPatch(mapped.total, result.patch)


@dataclass
class PotentialAsPropagator[State, Gradients, Hessians, StatePatch: Patch[Any]](
    Propagator[State]
):
    """Adapt a potential to the [Propagator][kups.core.propagator.Propagator] interface.

    Converts a potential into a propagator that computes energies and applies
    the resulting patch to the state. Useful for integrating potential evaluations
    into propagator pipelines.

    Attributes:
        potential: Potential to wrap as a propagator

    Note:
        The propagator accepts all patches (acceptance mask all True). This is
        typically used for energy/force evaluations rather than Monte Carlo moves.

    Example:
        ```python
        # Use potential in a propagator chain
        potential_prop = PotentialAsPropagator(lj_potential)

        # Propagate state (computes energy and applies patch)
        new_state = potential_prop(rng_key, state)
        ```
    """

    potential: Potential[State, Gradients, Hessians, StatePatch] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        """Evaluate potential and apply patch to state.

        Computes the potential energy and applies the resulting patch with all
        acceptance flags set to True (all updates accepted). Ignores the random key.

        Args:
            key: JAX PRNG key (unused)
            state: Current simulation state

        Returns:
            Updated state after applying potential patch
        """
        del key
        out = self.potential(state)
        energies = out.data.total_energies
        patch_result = out.patch(
            state, energies.set_data(jnp.ones(len(energies), dtype=bool))
        )
        return patch_result


@dataclass
class ShardedPotential[State, Gradients, Hessians, StatePatch: Patch[Any]](
    Potential[State, Gradients, Hessians, StatePatch]
):
    """Combine a domain-decomposed potential's per-device output across the mesh.

    Each device evaluates ``potential`` on its owned shard, yielding a
    per-device PARTIAL per-system energy; combining across the mesh makes the
    returned total the global value, replicated. ONLY the energy is combined —
    gradients w.r.t. the replicated positions/cell are already mesh-summed by
    ``shard_map``'s transpose, and summing them again would multiply forces by
    the device count. Valid only inside ``shard_map`` over the
    ``OriginDeviceId`` axis.

    The per-device outputs are compensated accumulators
    ([KahanSummand][kups.core.utils.kahan.KahanSummand]), and a compensation
    does NOT combine as a naive mesh sum: each device's compensation carries the
    low-order bits ITS OWN value dropped, while the cross-device additions drop
    bits of their own that a plain ``psum`` would lose. The accumulators are
    therefore all-gathered and folded in mesh order with the same exact-2Sum
    ``KahanSummand.__add__`` that ``SummedPotential`` uses to combine summands
    on one device. Every device folds the same gathered rows, so the fold is
    device-invariant; the closing ``pmax`` of identical values only re-types it
    as replicated for the varying-manual-axes check.

    The wrapped potential must not cache its own output (``cache_lens=None``):
    a state patch produced below this wrapper would write the pre-combine
    per-device partial into the state. Wrap in ``CachedPotential`` ABOVE this
    combiner instead, so the cached value is the global one.

    Attributes:
        potential: Domain-decomposed base potential returning per-device
            partial energies.
    """

    potential: Potential[State, Gradients, Hessians, StatePatch] = field(static=True)

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
        """Evaluate the shard's partial and combine the energy across the mesh.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates
            include_compensate: Return the accumulator rather than its
                compensated total

        Returns:
            Globally combined potential output with the shard's patch
        """
        out = self.potential(state, patch, include_compensate=True)
        summand = out.data
        energies = summand.value.total_energies
        compensations = summand.compensate.total_energies
        axis = shard_axis(OriginDeviceId)
        values = jax.lax.all_gather(energies.data, axis)  # (n_devices, n_systems)
        errors = jax.lax.all_gather(compensations.data, axis)
        total = KahanSummand(values[0], errors[0])
        for device in range(1, values.shape[0]):
            total = total + KahanSummand(values[device], errors[device])
        total = tree_map(lambda x: jax.lax.pmax(x, axis), total)
        combined = KahanSummand(
            PotentialOut(
                energies.set_data(total.value),
                summand.value.gradients,
                summand.value.hessians,
            ),
            PotentialOut(
                compensations.set_data(total.compensate),
                summand.compensate.gradients,
                summand.compensate.hessians,
            ),
        )
        if include_compensate:
            return WithPatch(combined, out.patch)
        return WithPatch(combined.total, out.patch)
