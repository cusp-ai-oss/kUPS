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
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.lens import Lens, View, bind, const_lens
from kups.core.patch import ComposedPatch, IndexLensPatch, Patch, WithPatch
from kups.core.propagator import Propagator
from kups.core.typing import HasPositionsAndSystemIndex, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, tree_map

type Energy = Array
"""Type alias for energy arrays, typically shape (n_systems,)."""


@dataclass
class EmptyType:
    """Sentinel type indicating empty gradients or Hessians.

    Use this when a potential does not compute gradients or Hessians,
    rather than None, to maintain type safety.
    """

    ...


EMPTY: EmptyType = EmptyType()
"""Singleton instance of EmptyType.

Use this instead of constructing EmptyType() directly.
"""

EMPTY_LENS: Lens[Any, EmptyType] = const_lens(EMPTY)
"""Lens that always returns EMPTY, ignoring input. Set is a no-op.

This is useful for potentials that don't compute gradients or Hessians.
"""


class IsStateWithParticles(Protocol):
    @property
    def particles(self) -> Table[ParticleId, HasPositionsAndSystemIndex]: ...


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


class Potential[
    State,
    Gradients,
    Hessians,
    StatePatch: Patch,
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

    Example:
        ```python
        class LennardJonesPotential:
            def __call__(self, state, patch=None):
                # Compute LJ energy and forces
                energy = compute_lj_energy(state.positions)
                forces = compute_lj_forces(state.positions)

                return WithPatch(
                    PotentialOut(energy, {"positions": forces}, EMPTY),
                    IdPatch()  # No state caching needed
                )

        # Use in simulation
        potential = LennardJonesPotential()
        result = potential(state)
        energy = result.data.total_energies
        forces = result.data.gradients.positions
        ```
    """

    def __call__(
        self, state: State, patch: StatePatch | None = None
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]:
        """Compute potential energy and derivatives.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates

        Returns:
            Potential output and state patch
        """
        ...


@dataclass
class SummedPotential[State, Gradients, Hessians, StatePatch: Patch](
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

    def __post_init__(self):
        if len(self.potentials) == 0:
            raise ValueError("At least one potential must be provided")

    def __call__(
        self, state: State, patch: StatePatch | None = None
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]:
        """Evaluate all potentials and sum their outputs.

        Calls each potential in sequence with the same state and patch, then
        sums the resulting energies, gradients, and Hessians element-wise.
        Patches are composed in order.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates

        Returns:
            Combined potential output with composed patches
        """
        outs = [s(state, patch) for s in self.potentials]
        # Sum using WithPatch.__add__ (adds data and composes patches)
        return sum(outs[1:], outs[0])


def sum_potentials[State, Gradients, Hessians, StatePatch: Patch](
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
class ScaledPotential[State, Gradients, Hessians, StatePatch: Patch](
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

    def __call__(
        self, state: State, patch: StatePatch | None = None
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]:
        """Evaluate potential and scale the output.

        Computes the base potential then multiplies energies, gradients, and
        Hessians by the scale factor. The patch is passed through unchanged.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates

        Returns:
            Scaled potential output with original patch
        """
        out = self.potential(state, patch)
        out = bind(out).focus(lambda x: x.data).apply(lambda x: x * self.scale)
        return out


@dataclass
class CachedPotential[State, Gradients, Hessians, StatePatch: Patch](
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
    cache: Lens[State, PotentialOut[Gradients, Hessians]] = field(static=True)
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = field(
        static=True, default=None
    )

    def __call__(
        self, state: State, patch: StatePatch | None = None
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]:
        """Evaluate potential and update cache.

        Computes the base potential, then creates a patch that will update the
        cached value when applied with an acceptance mask. The cache update uses
        the patch_idx_view to determine which cached entries to modify.

        Args:
            state: Current simulation state
            patch: Optional state patch for incremental updates

        Returns:
            Potential output with cache update patch composed
        """
        result = self.potential(state, patch)
        if self.patch_idx_view is not None:
            patch_idx = self.patch_idx_view(state)
        else:
            assert len(result.data.total_energies) == 1, (
                "patch_idx_view must be provided for multi-system potentials"
            )
            sys_keys = result.data.total_energies.keys
            patch_idx = tree_map(
                lambda x: Index(sys_keys, jnp.zeros(x.shape, dtype=int)), result.data
            )
        cache_patch = IndexLensPatch(result.data, patch_idx, self.cache)
        return WithPatch(result.data, ComposedPatch((result.patch, cache_patch)))

    def cached_value(self, state: State) -> PotentialOut[Gradients, Hessians]:
        """Retrieve the cached potential output from state.

        Args:
            state: Simulation state containing cached values

        Returns:
            Previously computed and cached potential output
        """
        return self.cache.get(state)


@dataclass
class MappedPotential[State, InGrad, OutGrad, InHess, OutHess, StatePatch: Patch](
    Potential[State, OutGrad, OutHess, StatePatch]
):
    """Wrap a potential and transform its gradient and hessian outputs.

    Applies mapping functions to gradients and hessians returned by the inner
    potential, enabling projection (e.g., extracting position gradients from a
    combined position+lattice gradient structure).

    Attributes:
        potential: Base potential to wrap
        gradient_map: Function to transform gradients from InGrad to OutGrad
        hessian_map: Function to transform hessians from InHess to OutHess

    Example:
        ```python
        # Extract position gradients from VirialTheoremGradients
        position_potential = MappedPotential(
            potential=full_potential,  # Returns VirialTheoremGradients
            gradient_map=lambda g: g.positions,
            hessian_map=lambda h: h,  # Pass through hessians unchanged
        )

        result = position_potential(state)
        # result.data.gradients is now just the position array
        ```
    """

    potential: Potential[State, InGrad, InHess, StatePatch] = field(static=True)
    gradient_map: View[InGrad, OutGrad] = field(static=True)
    hessian_map: View[InHess, OutHess] = field(static=True)

    def __call__(
        self, state: State, patch: StatePatch | None = None
    ) -> WithPatch[PotentialOut[OutGrad, OutHess], Patch[State]]:
        result = self.potential(state, patch)
        mapped_out = PotentialOut(
            total_energies=result.data.total_energies,
            gradients=self.gradient_map(result.data.gradients),
            hessians=self.hessian_map(result.data.hessians),
        )
        return WithPatch(mapped_out, result.patch)


@dataclass
class PotentialAsPropagator[State, Gradients, Hessians, StatePatch: Patch](
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
