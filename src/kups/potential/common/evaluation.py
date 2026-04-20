# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for evaluating potentials outside of JIT-compiled simulation loops.

These functions handle the boilerplate of constructing a one-shot potential call
with assertion-based retry logic, making it straightforward to evaluate energies,
forces, and Hessians for analysis or initialisation purposes.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import jax
import jax.core
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.lens import Lens, View, lens
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    Edges,
    NearestNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.patch import Patch, WithPatch
from kups.core.potential import EMPTY_LENS, Potential, PotentialOut
from kups.core.result import Result, as_result_function
from kups.core.typing import HasUnitCell, ParticleId, SystemId
from kups.core.utils.functools import constant
from kups.core.utils.jax import dataclass, no_jax_tracing
from kups.potential.classical.ewald import (
    EwaldLongRangeInput,
    EwaldParameters,
    IsEwaldPointData,
    ewald_long_range_energy,
    ewald_self_interaction_energy,
    ewald_short_range_energy,
)
from kups.potential.common.energy import (
    EnergyFunction,
    IdentityComposer,
    PotentialFromEnergy,
)
from kups.potential.common.graph import (
    FullGraphSumComposer,
    GraphPotentialInput,
    HyperGraph,
    IsRadiusGraphPoints,
    PointCloud,
    RadiusGraphConstructor,
)


def potential_with_assertions[State, G, H, P: Patch](
    potential: Potential[State, G, H, P],
) -> Callable[
    [State, P | None], Result[State, WithPatch[PotentialOut[G, H], Patch[State]]]
]:
    """Wrap a potential so its output is lifted into a ``Result`` carrying assertions.

    Args:
        potential: The potential to wrap.

    Returns:
        A callable with the same signature that returns a ``Result`` instead of
        a raw ``WithPatch``.
    """
    return as_result_function(potential)


@no_jax_tracing
def evaluate_potential_and_fix[State, Gradients, Hessians, P: Patch](
    potential: Callable[
        [State, P | None],
        Result[State, WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]],
    ],
    state: State,
    patch: P | None = None,
    /,
    max_tries: int = 10,
) -> tuple[State, WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]]:
    """Evaluate a potential, retrying with assertion fixes until it succeeds.

    On each attempt, failed assertions are fixed via their ``fix`` functions.
    Assertions without a fix function will raise immediately.

    Args:
        potential: Assertion-aware potential (e.g. from ``potential_with_assertions``).
        state: Current simulation state.
        patch: Optional patch passed through to the potential.
        max_tries: Maximum number of retry attempts before raising.

    Returns:
        ``(fixed_state, output)`` where ``output`` is the first successful result.

    Raises:
        ValueError: If called inside a JAX-traced context.
        RuntimeError: If the potential still fails after ``max_tries`` attempts.
    """
    is_traced = any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(state))
    if is_traced:
        raise ValueError("potential_and_fix cannot be jax transformed.")

    for _ in range(max_tries):
        out = potential(state, patch)
        if not out.failed_assertions:
            return state, out.value
        state = out.fix_or_raise(state)
    raise RuntimeError("Failed to resolve potential after multiple attempts")


@no_jax_tracing
def evaluate_potential[Input, Gradients, Hessians](
    input: Input,
    *,
    energy_fn: EnergyFunction[Input, Input],
    gradient_lens: Lens[Input, Gradients] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[Input, Hessians] = EMPTY_LENS,
) -> PotentialOut[Gradients, Hessians]:
    """Evaluate an energy function on a plain input struct (no graph construction).

    Useful for potentials whose input is already fully constructed (e.g. Ewald
    long-range, self-interaction).

    Args:
        input: Input passed directly to ``energy_fn``.
        energy_fn: Energy function to evaluate.
        gradient_lens: Lens selecting the output to differentiate with respect to.
        hessian_lens: Lens selecting the gradient output for the Hessian.
        hessian_idx_view: View used to index into the Hessian output.

    Returns:
        ``PotentialOut`` containing energy, gradients, and Hessians.
    """
    potential = PotentialFromEnergy(
        energy_fn,
        IdentityComposer[Input](),
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=None,
        patch_idx_view=None,
    )
    pot_w_assertions = potential_with_assertions(potential)
    return evaluate_potential_and_fix(pot_w_assertions, input)[1].data


@dataclass
class _RadiusGraphEvalState:
    neighborlist_params: UniversalNeighborlistParameters

    @property
    def neighborlist(self) -> NearestNeighborList:
        return DenseNearestNeighborList.from_state(self)


@no_jax_tracing
def evaluate_radius_graph_potential[
    Parameters,
    Gradients,
    Hessians,
    P: IsRadiusGraphPoints,
    S: HasUnitCell,
](
    point_cloud: PointCloud[P, S],
    parameters: Parameters,
    *,
    cutoffs: Table[SystemId, Array] | None = None,
    energy_fn: EnergyFunction[
        Any,
        GraphPotentialInput[Parameters, P, S, Literal[2]],
    ],
    gradient_lens: Lens[
        GraphPotentialInput[Parameters, P, S, Literal[2]],
        Gradients,
    ] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[Any, Hessians] = EMPTY_LENS,
) -> PotentialOut[Gradients, Hessians]:
    """Build a radius graph and evaluate an edge-based energy function on it.

    Uses a pessimistic ``DenseNearestNeighborList`` sized to the largest system,
    growing as needed via assertion retries.

    Args:
        point_cloud: Particles and systems (unit cell).
        parameters: Parameters forwarded to ``energy_fn``.
        cutoffs: Indexed cutoff data per system. If None, tries to extract
            from ``parameters.cutoff``.
        energy_fn: Edge-based energy function.
        gradient_lens: Lens selecting the differentiation target.
        hessian_lens: Lens selecting the gradient for Hessian computation.
        hessian_idx_view: View used to index into the Hessian output.

    Returns:
        ``PotentialOut`` containing energy, gradients, and Hessians.
    """
    if cutoffs is None:
        cutoffs = Table(
            point_cloud.systems.keys,
            parameters.cutoff,  # type: ignore[union-attr]
        )
    neighborlist_params = UniversalNeighborlistParameters.estimate(
        point_cloud.particles.data.system.counts, point_cloud.systems, cutoffs
    )
    state = _RadiusGraphEvalState(neighborlist_params=neighborlist_params)
    potential = PotentialFromEnergy(
        energy_fn,
        FullGraphSumComposer(
            graph_constructor=RadiusGraphConstructor(
                particles=constant(point_cloud.particles),
                systems=constant(point_cloud.systems),
                cutoffs=constant(cutoffs),
                neighborlist=lambda s: s.neighborlist,
                probe=None,
            ),
            parameter_view=constant(parameters),
        ),
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=None,
        patch_idx_view=None,
    )
    pot_w_assertions = potential_with_assertions(potential)
    return evaluate_potential_and_fix(pot_w_assertions, state)[1].data


@dataclass
class _EwaldRadiusGraphPointsImpl:
    """Wraps IsEwaldPointData with inclusion/exclusion for short-range evaluation."""

    positions: Array
    system: Index[SystemId]
    inclusion: Index[SystemId]
    exclusion: Index[ParticleId]
    charges: Array


@no_jax_tracing
def evaluate_ewald_potential[
    Gradients,
    Hessians,
](
    point_cloud: PointCloud[IsEwaldPointData, HasUnitCell],
    parameters: EwaldParameters,
    *,
    gradient_lens: Lens[
        PointCloud[IsEwaldPointData, HasUnitCell], Gradients
    ] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[
        PointCloud[IsEwaldPointData, HasUnitCell], Hessians
    ] = EMPTY_LENS,
) -> PotentialOut[Gradients, Hessians]:
    """Evaluate the full Ewald potential: long-range + short-range + self-interaction.

    Args:
        point_cloud: Particles and systems for the Ewald sum.
        parameters: Ewald parameters (alpha, cutoff, k-vectors, cache).
        gradient_lens: Lens selecting the differentiation target on ``point_cloud``.
        hessian_lens: Lens selecting the gradient for Hessian computation.
        hessian_idx_view: View used to index into the Hessian output.

    Returns:
        ``PotentialOut`` summing long-range, short-range, and self-interaction terms.
    """
    lr_out = evaluate_potential(
        EwaldLongRangeInput(point_cloud, parameters),
        energy_fn=ewald_long_range_energy,
        gradient_lens=lens(lambda x: x.point_cloud, cls=EwaldLongRangeInput).focus(
            gradient_lens.get
        ),
        hessian_lens=hessian_lens,
        hessian_idx_view=lambda state: hessian_idx_view(state.point_cloud),
    )

    # Build particles with per-particle exclusion for short-range
    n_particles = point_cloud.particles.size
    pi_keys = point_cloud.particles.keys
    sr_particles = Table(
        pi_keys,
        _EwaldRadiusGraphPointsImpl(
            positions=point_cloud.particles.data.positions,
            system=point_cloud.particles.data.system,
            inclusion=point_cloud.particles.data.system,
            exclusion=Index(pi_keys, jnp.arange(n_particles)),
            charges=point_cloud.particles.data.charges,
        ),
    )

    sr_cloud: PointCloud = PointCloud(sr_particles, point_cloud.systems)
    sr_cutoffs = Table(
        point_cloud.systems.keys,
        jnp.broadcast_to(
            jnp.array(parameters.cutoff),
            (point_cloud.systems.size,),
        ),
    )
    sr_out = evaluate_radius_graph_potential(
        point_cloud=sr_cloud,
        parameters=parameters,
        cutoffs=sr_cutoffs,
        energy_fn=ewald_short_range_energy,
        gradient_lens=lens(lambda x: x.graph, cls=GraphPotentialInput).focus(
            gradient_lens.get
        ),
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
    )
    self_out = evaluate_potential(
        GraphPotentialInput(
            parameters,
            HyperGraph(
                point_cloud.particles,
                point_cloud.systems,
                Edges(
                    Index(pi_keys, jnp.zeros((0, 0), dtype=int)),
                    jnp.zeros((0, 0, 3), dtype=int),
                ),
            ),
        ),
        energy_fn=ewald_self_interaction_energy,
        gradient_lens=lens(lambda x: x.graph, cls=GraphPotentialInput).focus(
            gradient_lens.get
        ),
        hessian_lens=hessian_lens,
        hessian_idx_view=lambda state: hessian_idx_view(state.graph),
    )
    return lr_out + sr_out + self_out
