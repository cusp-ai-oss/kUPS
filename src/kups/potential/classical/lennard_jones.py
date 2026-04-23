# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Lennard-Jones potential implementations with tail corrections.

This module provides the Lennard-Jones 12-6 potential for van der Waals interactions:

$$
U(r) = 4\\epsilon\\left[\\left(\\frac{\\sigma}{r}\\right)^{12} - \\left(\\frac{\\sigma}{r}\\right)^6\\right]
$$

Includes variants with smooth tail corrections and analytical long-range corrections for
periodic systems. Supports Lorentz-Berthelot mixing rules for multi-component systems.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    assert_never,
    cast,
    overload,
    runtime_checkable,
)

import jax.numpy as jnp
from jax import Array

from kups.core.data import Table
from kups.core.lens import Lens, SimpleLens, View, identity_lens
from kups.core.neighborlist import NearestNeighborList
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Energy,
    Potential,
    PotentialOut,
    empty_patch_idx_view,
)
from kups.core.propagator import StateProperty
from kups.core.typing import (
    HasCache,
    HasExclusionIndex,
    HasInclusionIndex,
    HasLabels,
    HasPositions,
    HasSystemIndex,
    HasUnitCell,
    Label,
    MaybeCached,
    ParticleId,
    SystemId,
)
from kups.core.utils.functools import pipe
from kups.core.utils.jax import dataclass, field, jit
from kups.potential.common.energy import (
    EnergyFunction,
    PositionAndUnitCell,
    PotentialFromEnergy,
    position_and_unitcell_idx_view,
)
from kups.potential.common.graph import (
    FullGraphSumComposer,
    GraphPotentialInput,
    IsRadiusGraphProbe,
    LocalGraphSumComposer,
    PointCloudConstructor,
    RadiusGraphConstructor,
)

type MixingRule = Literal["lorentz_berthelot"]


@runtime_checkable
class IsLennardJonesParticles(HasPositions, HasLabels, HasSystemIndex, Protocol): ...


@dataclass
class LennardJonesParameters:
    """Lennard-Jones potential parameters.

    Attributes:
        labels: Species labels as ``Index``.
        sigma: Length scale parameters [Å], shape ``(n_species, n_species)``.
        epsilon: Energy well depths [energy units], shape ``(n_species, n_species)``.
        cutoff: Cutoff radius [Å], shape ``(n_systems,)``.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    sigma: Array  # (n_species, n_species) float
    epsilon: Array  # (n_species, n_species) float
    cutoff: Table[SystemId, Array]  # (n_graphs,) float

    @classmethod
    def from_dict(
        cls,
        cutoff: float | Array,
        parameters: dict[str, tuple[float | None, float | None]],
        mixing_rule: MixingRule,
    ) -> LennardJonesParameters:
        """Create parameters from a dict of per-species values.

        Args:
            cutoff: Cutoff radius [Angstrom].
            parameters: Map from species label to ``(sigma, epsilon)`` pair.
                ``None`` values default to ``sigma=1.0``, ``epsilon=0.0``.
            mixing_rule: Combining rule for cross-species interactions.
        """
        labels = tuple(parameters.keys())
        raw = [(s or 1.0, e or 0.0) for s, e in parameters.values()]
        sigma, epsilon = jnp.asarray(raw).T
        cutoff_indexed = Table((SystemId(0),), jnp.array([cutoff]))
        match mixing_rule:
            case "lorentz_berthelot":
                return cls.from_lorentz_berthelot_mixing(
                    labels, sigma, epsilon, cutoff_indexed
                )
            case _ as unreachable:
                assert_never(unreachable)

    @classmethod
    def from_lorentz_berthelot_mixing(
        cls,
        labels: tuple[str, ...],
        sigma: Array,
        epsilon: Array,
        cutoff: Table[SystemId, Array],
    ) -> LennardJonesParameters:
        """Create parameters using Lorentz-Berthelot mixing rules.

        - σᵢⱼ = (σᵢ + σⱼ) / 2 (arithmetic mean)
        - εᵢⱼ = √(εᵢ × εⱼ) (geometric mean)
        """
        assert sigma.ndim == epsilon.ndim == 1
        sigma_matrix = (sigma[:, None] + sigma) / 2
        epsilon_matrix = jnp.sqrt(epsilon[:, None] * epsilon)
        return cls(tuple(map(Label, labels)), sigma_matrix, epsilon_matrix, cutoff)


type LennardJonesInput = GraphPotentialInput[
    LennardJonesParameters, IsLennardJonesParticles, HasUnitCell, Literal[2]
]


@jit
def lennard_jones_edge_energy(inp: LennardJonesInput) -> Array:
    """Compute Lennard-Jones energy per edge."""
    graph = inp.graph
    assert graph.edges.indices.shape[1] == 2
    sigma = inp.parameters.sigma
    epsilon = inp.parameters.epsilon
    assert sigma.ndim == 2 and sigma.shape[0] == sigma.shape[1]
    assert epsilon.ndim == 2 and epsilon.shape[0] == epsilon.shape[1]
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    epsilon = epsilon[edg_species[:, 0], edg_species[:, 1]]
    sigma = sigma[edg_species[:, 0], edg_species[:, 1]]
    r2 = jnp.sum(graph.edge_shifts[:, 0] ** 2, axis=-1)
    c6 = (sigma**2 / r2) ** 3
    edge_energy = 4 * epsilon * (c6**2 - c6)
    batch = graph.edge_batch_mask.indices
    mask = r2 < jnp.pow(inp.parameters.cutoff.data, 2)[batch]
    return edge_energy * mask


def lennard_jones_energy(
    inp: LennardJonesInput,
) -> WithPatch[Table[SystemId, Energy], Patch]:
    """Compute total Lennard-Jones energy per system."""
    graph = inp.graph
    edge_energy = lennard_jones_edge_energy(inp)
    total_energies = graph.edge_batch_mask.sum_over(edge_energy) / 2
    return WithPatch(total_energies, IdPatch())


@dataclass
class PairTailCorrectedLennardJonesParameters(LennardJonesParameters):
    """Lennard-Jones parameters with smooth pairwise tail correction.

    Attributes:
        truncation_radius: Radius where smoothing begins [Å], shape ``(n_systems,)``.
    """

    truncation_radius: Table[SystemId, Array]  # (n_graphs,)


type PairTailCorrectedLennardJonesInput = GraphPotentialInput[
    PairTailCorrectedLennardJonesParameters,
    IsLennardJonesParticles,
    HasUnitCell,
    Literal[2],
]


@jit
def pair_tail_corrected_lennard_jones_energy(
    inp: PairTailCorrectedLennardJonesInput,
) -> WithPatch[Table[SystemId, Energy], Patch[Any]]:
    """Compute Lennard-Jones energy with smooth pairwise tail correction."""
    graph = inp.graph
    r: Array = jnp.linalg.norm(graph.edge_shifts, axis=(-2, -1))
    edge_energy = lennard_jones_edge_energy(cast(LennardJonesInput, inp))

    batch = graph.edge_batch_mask
    r_tr = inp.parameters.truncation_radius[batch]
    r_cut = inp.parameters.cutoff[batch]
    mask = r > r_tr
    remove = r >= r_cut
    factor1 = ((r_cut**2) - r**2) ** 2
    factor2 = 2 * r**2 + (r_cut**2 - 3 * r_tr**2)
    div = (r_cut**2 - r_tr**2) ** 3
    corrected_edge_energy = jnp.where(
        mask, edge_energy * factor1 * factor2 / div, edge_energy
    )
    corrected_edge_energy = jnp.where(remove, 0.0, corrected_edge_energy)
    total_energies = batch.sum_over(corrected_edge_energy) / 2
    return WithPatch(total_energies, IdPatch())


@dataclass
class GlobalTailCorrectedLennardJonesParameters(LennardJonesParameters):
    """Lennard-Jones parameters with analytical long-range correction.

    Attributes:
        tail_corrected: Enable correction per species pair, shape ``(n_species, n_species)``.
    """

    tail_corrected: Array  # (n_species, n_species) bool

    @classmethod
    def from_dict(
        cls,
        cutoff: float | Array,
        parameters: dict[str, tuple[float | None, float | None]],
        mixing_rule: MixingRule,
        tail_correction: bool = True,
    ) -> GlobalTailCorrectedLennardJonesParameters:
        """Create tail-corrected parameters from a dict of per-species values.

        Args:
            cutoff: Cutoff radius [Angstrom].
            parameters: Map from species label to ``(sigma, epsilon)`` pair.
                ``None`` values default to ``sigma=1.0``, ``epsilon=0.0``.
            mixing_rule: Combining rule for cross-species interactions.
            tail_correction: Whether to enable tail corrections for all
                non-zero epsilon pairs.
        """
        base = LennardJonesParameters.from_dict(cutoff, parameters, mixing_rule)
        mask = (
            base.epsilon > 0
            if tail_correction
            else jnp.zeros_like(base.epsilon, dtype=bool)
        )
        return cls(
            labels=base.labels,
            sigma=base.sigma,
            epsilon=base.epsilon,
            cutoff=base.cutoff,
            tail_corrected=mask,
        )


type GlobalTailCorrectedLennardJonesInput = GraphPotentialInput[
    GlobalTailCorrectedLennardJonesParameters,
    IsLennardJonesParticles,
    HasUnitCell,
    Literal[0],
]


def _global_tail_correction_common(
    inp: GlobalTailCorrectedLennardJonesInput,
) -> tuple[Array, Array, Array, Array, Array, int]:
    """Extract shared quantities for global LJ tail correction energy/pressure."""
    n_species = inp.parameters.sigma.shape[0]
    assert inp.parameters.sigma.shape == (n_species, n_species)
    assert inp.parameters.epsilon.shape == (n_species, n_species)
    assert inp.parameters.tail_corrected.shape == (n_species, n_species)
    n_graphs = inp.graph.batch_size
    system_ids = inp.graph.particles.data.system.indices
    species_ids = inp.graph.particles.data.labels.indices_in(inp.parameters.labels)
    counts = (
        jnp.zeros((n_graphs, n_species), dtype=int)
        .at[system_ids, species_ids]
        .add(1, mode="drop")
    )
    volume = inp.graph.systems.data.unitcell.volume[:, None, None]
    density = (counts[:, :, None] * counts[:, None, :]) / volume
    sigma = inp.parameters.sigma
    cutoff = inp.parameters.cutoff.data[:, None, None]
    term1 = (sigma / cutoff) ** 3
    term2 = term1**3
    return density, volume, term1, term2, inp.parameters.tail_corrected, n_graphs


@jit
def global_lennard_jones_tail_correction_energy(
    inp: GlobalTailCorrectedLennardJonesInput,
) -> WithPatch[Table[SystemId, Energy], Patch]:
    """Compute analytical long-range tail correction energy."""
    density, _volume, term1, term2, tail_mask, n_graphs = (
        _global_tail_correction_common(inp)
    )
    sigma = inp.parameters.sigma
    epsilon = inp.parameters.epsilon
    result = (8 / 3) * jnp.pi * density * epsilon * sigma**3 * (term2 / 3 - term1)
    result *= tail_mask
    total_energies = Table.arange(result.sum(axis=(1, 2)), label=SystemId)
    assert len(total_energies) == n_graphs
    return WithPatch(total_energies, IdPatch())


@jit
def global_lennard_jones_tail_correction_pressure(
    inp: GlobalTailCorrectedLennardJonesInput,
) -> WithPatch[Table[SystemId, Energy], Patch]:
    """Compute analytical long-range tail correction for pressure."""
    density, volume, term1, term2, tail_mask, n_graphs = _global_tail_correction_common(
        inp
    )
    sigma = inp.parameters.sigma
    epsilon = inp.parameters.epsilon
    result = (
        (16 / 3)
        * jnp.pi
        * density
        / volume
        * epsilon
        * sigma**3
        * (term2 / 3 * 2 - term1)
    )
    result *= tail_mask
    total_pressure = Table.arange(result.sum(axis=(1, 2)), label=SystemId)
    assert len(total_pressure) == n_graphs
    return WithPatch(total_pressure, IdPatch())


# --- Factory functions ---
@runtime_checkable
class IsLJGraphParticles(
    IsLennardJonesParticles, HasInclusionIndex, HasExclusionIndex, Protocol
): ...


type LJRadiusInp = GraphPotentialInput[
    LennardJonesParameters, IsLJGraphParticles, HasUnitCell, Literal[2]
]


def make_lennard_jones_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsLJGraphParticles]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    neighborlist_view: View[State, NearestNeighborList],
    parameter_view: View[State, LennardJonesParameters],
    probe: Probe[State, Ptch, IsRadiusGraphProbe[IsLJGraphParticles]] | None,
    gradient_lens: Lens[LJRadiusInp, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create a standard Lennard-Jones potential with sharp cutoff."""
    graph_fn = RadiusGraphConstructor(
        particles=particles_view,
        systems=systems_view,
        cutoffs=pipe(parameter_view, lambda p: p.cutoff),
        neighborlist=neighborlist_view,
        probe=probe,
    )
    composer = LocalGraphSumComposer(
        graph_constructor=graph_fn,
        parameter_view=parameter_view,
    )
    return PotentialFromEnergy(
        composer=composer,
        energy_fn=lennard_jones_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


class HasLJParticlesAndSystems(Protocol):
    """Protocol for states with indexed particles and systems."""

    @property
    def particles(self) -> Table[ParticleId, IsLJGraphParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...


class IsLJState[Params](HasLJParticlesAndSystems, Protocol):
    """State with particles, systems, neighbor list, and LJ parameters."""

    @property
    def neighborlist(self) -> NearestNeighborList: ...
    @property
    def lj_parameters(self) -> Params: ...


@overload
def make_lennard_jones_from_state[State](
    state: Lens[State, IsLJState[MaybeCached[LennardJonesParameters, Any]]],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch]: ...


@overload
def make_lennard_jones_from_state[State](
    state: Lens[State, IsLJState[MaybeCached[LennardJonesParameters, Any]]],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, Patch]: ...


@overload
def make_lennard_jones_from_state[State, P: Patch](
    state: Lens[
        State,
        IsLJState[HasCache[LennardJonesParameters, PotentialOut[EmptyType, EmptyType]]],
    ],
    probe: Probe[State, P, IsRadiusGraphProbe],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_lennard_jones_from_state[State, P: Patch](
    state: Lens[
        State,
        IsLJState[
            HasCache[
                LennardJonesParameters, PotentialOut[PositionAndUnitCell, EmptyType]
            ]
        ],
    ],
    probe: Probe[State, P, IsRadiusGraphProbe],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, P]: ...


def make_lennard_jones_from_state(
    state: Any,
    probe: Any = None,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create a LJ potential from a typed state, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, systems, neighbor list,
            and LJ parameters.
        probe: Detects which particles changed since the last step.
            ``None`` for full recomputation.
        compute_position_and_unitcell_gradients: When ``True``, the returned
            potential computes gradients w.r.t. particle positions and lattice
            vectors. Gradient type becomes ``PositionAndUnitCell``.

    Returns:
        Configured Lennard-Jones potential.
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[GraphPotentialInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.unitcell),
            )
        )
        patch_idx_view = position_and_unitcell_idx_view
    param_view = state.focus(
        lambda x: (
            x.lj_parameters.data
            if isinstance(x.lj_parameters, HasCache)
            else x.lj_parameters
        )
    )
    cache_view = None
    if probe is not None:
        param_view = state.focus(lambda x: x.lj_parameters.data)
        cache_view = state.focus(lambda x: x.lj_parameters.cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_lennard_jones_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        state.focus(lambda x: x.neighborlist),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
        cache_view,
    )


type PCLJInp = GraphPotentialInput[
    PairTailCorrectedLennardJonesParameters, IsLJGraphParticles, HasUnitCell, Literal[2]
]


def make_pair_tail_corrected_lennard_jones_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsLJGraphParticles]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    neighborlist_view: View[State, NearestNeighborList],
    parameter_view: View[State, PairTailCorrectedLennardJonesParameters],
    probe: Probe[State, Ptch, IsRadiusGraphProbe[IsLJGraphParticles]] | None,
    gradient_lens: Lens[PCLJInp, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create a Lennard-Jones potential with smooth pairwise tail correction."""
    radius_graph_fn = RadiusGraphConstructor(
        particles=particles_view,
        systems=systems_view,
        cutoffs=pipe(parameter_view, lambda p: p.cutoff),
        neighborlist=neighborlist_view,
        probe=probe,
    )
    composer = LocalGraphSumComposer(
        graph_constructor=radius_graph_fn,
        parameter_view=parameter_view,
    )
    return PotentialFromEnergy(
        composer=composer,
        energy_fn=pair_tail_corrected_lennard_jones_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


type GCLJInp = GraphPotentialInput[
    GlobalTailCorrectedLennardJonesParameters,
    IsLJGraphParticles,
    HasUnitCell,
    Literal[0],
]


def make_global_lennard_jones_tail_correction_potential[State, Gradients, Hessians](
    particles_view: View[State, Table[ParticleId, IsLJGraphParticles]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    parameter_view: View[State, GlobalTailCorrectedLennardJonesParameters],
    gradient_lens: Lens[GCLJInp, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Patch[State]]:
    """Create analytical long-range tail correction for Lennard-Jones potential."""
    return PotentialFromEnergy(
        energy_fn=global_lennard_jones_tail_correction_energy,
        composer=FullGraphSumComposer(
            PointCloudConstructor(
                particles=particles_view,
                systems=systems_view,
                probe_particles=None,
            ),
            parameter_view=parameter_view,
        ),
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


type IsGlobalTailCorrectedIsLJState = IsLJState[
    MaybeCached[GlobalTailCorrectedLennardJonesParameters, PotentialOut]
]


@overload
def make_lennard_jones_tail_correction_from_state[
    InState,
    State: IsGlobalTailCorrectedIsLJState,
](
    state: Lens[InState, State],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[InState, EmptyType, EmptyType, Patch]: ...


@overload
def make_lennard_jones_tail_correction_from_state[
    InState,
    State: IsGlobalTailCorrectedIsLJState,
](
    state: Lens[InState, State],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[InState, PositionAndUnitCell, EmptyType, Patch]: ...


def make_lennard_jones_tail_correction_from_state(
    state: Any,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create a global tail-corrected LJ potential from a typed state.

    Args:
        state: Lens into the sub-state providing particles, systems, and
            LJ tail correction parameters.
        compute_position_and_unitcell_gradients: When ``True``, the returned
            potential computes gradients w.r.t. particle positions and lattice
            vectors. Gradient type becomes ``PositionAndUnitCell``.

    Returns:
        Configured tail-corrected Lennard-Jones potential.
    """
    gradient_lens: Any = EMPTY_LENS
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[GraphPotentialInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.unitcell),
            )
        )
    return make_global_lennard_jones_tail_correction_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        cast(
            Any,
            state.focus(
                lambda x: (
                    x.lj_parameters.data
                    if isinstance(x.lj_parameters, HasCache)
                    else x.lj_parameters
                )
            ),
        ),
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
    )


def make_global_lennard_jones_tail_correction_pressure[State](
    particles_view: View[State, Table[ParticleId, IsLJGraphParticles]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    parameter_view: View[State, GlobalTailCorrectedLennardJonesParameters],
) -> StateProperty[State, Table[SystemId, Array]]:
    """Create long-range pressure correction for Lennard-Jones systems."""
    graph_constructor = PointCloudConstructor(
        particles=particles_view,
        systems=systems_view,
        probe_particles=None,
    )

    def pressure(key: Array, state: State) -> Table[SystemId, Array]:
        del key
        params = parameter_view(state)
        graph = graph_constructor(state, None)
        return global_lennard_jones_tail_correction_pressure(
            GraphPotentialInput(params, graph)
        ).data

    return pressure


def global_lennard_jones_tail_correction_pressure_from_state(
    key: Array, state: IsGlobalTailCorrectedIsLJState
) -> Table[SystemId, Array]:
    """Create long-range pressure correction from a typed state."""
    state_lens = identity_lens(type(state))
    return make_global_lennard_jones_tail_correction_pressure(
        state_lens.focus(lambda x: x.particles),
        state_lens.focus(lambda x: x.systems),
        cast(
            Any,
            state_lens.focus(
                lambda x: (
                    x.lj_parameters.data
                    if isinstance(x.lj_parameters, HasCache)
                    else x.lj_parameters
                )
            ),
        ),
    )(key, state)


if TYPE_CHECKING:
    _lj: EnergyFunction[Any, LennardJonesInput] = lennard_jones_energy
    _ptc: EnergyFunction[Any, PairTailCorrectedLennardJonesInput] = (
        pair_tail_corrected_lennard_jones_energy
    )
