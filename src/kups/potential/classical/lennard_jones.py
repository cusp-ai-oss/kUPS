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
    override,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import (
    EmptyNeighborList,
    NeighborList,
)
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    Energy,
    Potential,
    PotentialOut,
)
from kups.core.propagator import StateProperty
from kups.core.typing import (
    HasCell,
    HasExclusionIndex,
    HasInclusionIndex,
    HasLabels,
    HasPositions,
    HasSystemIndex,
    Label,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field, jit
from kups.core.utils.kahan import KahanSummand
from kups.potential.common.energy import (
    EnergyFunction,
    FullSumComposer,
    LocalSumComposer,
    PotentialFromEnergy,
)
from kups.potential.common.graph import (
    GraphConstructor,
    GraphInputConstructor,
    GraphPotentialInput,
    IsGraphProbe,
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
    LennardJonesParameters, IsLennardJonesParticles, HasCell[AnyPeriodicity], Literal[2]
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
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Compute total Lennard-Jones energy per system."""
    graph = inp.graph
    edge_energy = lennard_jones_edge_energy(inp)
    total_energies = graph.edge_batch_mask.sum_over(edge_energy) / 2
    return WithPatch(total_energies, IdPatch[Any]())


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
    HasCell[AnyPeriodicity],
    Literal[2],
]


@jit
def pair_tail_corrected_lennard_jones_energy(
    inp: PairTailCorrectedLennardJonesInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
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
    return WithPatch(total_energies, IdPatch[Any]())


@dataclass
class GlobalTailCorrectedLennardJonesParameters(LennardJonesParameters):
    """Lennard-Jones parameters with analytical long-range correction.

    Attributes:
        tail_corrected: Enable correction per species pair, shape ``(n_species, n_species)``.
    """

    tail_corrected: Array  # (n_species, n_species) bool

    @property
    def tail_coefficients(self) -> tuple[Array, Array]:
        """Two cutoff-independent coefficient matrices over species pairs."""
        sigma6 = self.sigma**6
        base = self.tail_corrected * self.epsilon
        return base * sigma6, base * sigma6**2

    @property
    def tail_weights(self) -> Array:
        """Current pair weights, shaped ``(2, n_systems, n_species, n_species)``."""
        term1 = (self.sigma / self.cutoff.data[:, None, None]) ** 3
        base = self.tail_corrected * self.epsilon * self.sigma**3
        return jnp.stack([base * term1, base * term1**3])

    @classmethod
    @override
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
    HasCell[AnyPeriodicity],
    Literal[0],
]


def _global_tail_correction_common(
    inp: GlobalTailCorrectedLennardJonesInput,
) -> tuple[Array, Array, Array, int]:
    """Shared quantities for the global LJ tail correction energy/pressure.

    Contracts the species counts with the pair weight matrices as bilinear
    forms, ``q_i = n^T W_i n``, instead of materializing the density matrix.

    Returns:
        ``(q1 / V, q2 / V, V, n_graphs)`` with
        ``W1 = mask*eps*sigma^3*(sigma/r_c)^3`` and ``W2 = mask*eps*sigma^3*(sigma/r_c)^9``.
    """
    n_species = inp.parameters.sigma.shape[0]
    assert inp.parameters.sigma.shape == (n_species, n_species)
    assert inp.parameters.epsilon.shape == (n_species, n_species)
    assert inp.parameters.tail_corrected.shape == (n_species, n_species)
    n_graphs = inp.graph.batch_size
    system_ids = inp.graph.particles.data.system.indices
    species_ids = inp.graph.particles.data.labels.indices_in(inp.parameters.labels)
    counts = (
        jnp.zeros((n_graphs, n_species), dtype=inp.parameters.sigma.dtype)
        .at[system_ids, species_ids]
        .add(1, mode="drop")
    )
    volume = inp.graph.systems.data.cell.volume
    if (
        jax.default_backend() in {"gpu", "cuda", "rocm"}
        and n_graphs * n_species**2 <= 65_536
    ):
        # The expanded contraction is faster for small GPU workloads.
        w1, w2 = inp.parameters.tail_weights
        q1 = jnp.einsum("gs,gst,gt->g", counts, w1, counts) / volume
        q2 = jnp.einsum("gs,gst,gt->g", counts, w2, counts) / volume
    else:
        # Factor out cutoffs to share the species matrices across systems.
        w1, w2 = inp.parameters.tail_coefficients
        cutoff3 = inp.parameters.cutoff.data**3
        q1 = jnp.einsum("gs,st,gt->g", counts, w1, counts)
        q2 = jnp.einsum("gs,st,gt->g", counts, w2, counts)
        q1 /= volume * cutoff3
        q2 /= volume * cutoff3**3
    return q1, q2, volume, n_graphs


@jit
def global_lennard_jones_tail_correction_energy(
    inp: GlobalTailCorrectedLennardJonesInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Compute analytical long-range tail correction energy."""
    q1, q2, _volume, n_graphs = _global_tail_correction_common(inp)
    result = (8 / 3) * jnp.pi * (q2 / 3 - q1)
    total_energies = Table.arange(result, label=SystemId)
    assert len(total_energies) == n_graphs
    return WithPatch(total_energies, IdPatch[Any]())


@jit
def global_lennard_jones_tail_correction_pressure(
    inp: GlobalTailCorrectedLennardJonesInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Compute analytical long-range tail correction for pressure."""
    q1, q2, volume, n_graphs = _global_tail_correction_common(inp)
    result = (16 / 3) * jnp.pi / volume * (q2 / 3 * 2 - q1)
    total_pressure = Table.arange(result, label=SystemId)
    assert len(total_pressure) == n_graphs
    return WithPatch(total_pressure, IdPatch[Any]())


# --- Factory functions ---
@runtime_checkable
class IsLJGraphParticles(
    IsLennardJonesParticles, HasInclusionIndex, HasExclusionIndex, Protocol
): ...


type LJRadiusInp = GraphPotentialInput[
    LennardJonesParameters, IsLJGraphParticles, HasCell[AnyPeriodicity], Literal[2]
]


def make_lennard_jones_potential[
    State,
    Ptch: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsLJGraphParticles]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    parameter_view: View[State, LennardJonesParameters],
    probe: Probe[State, Ptch, IsGraphProbe[IsLJGraphParticles, Literal[2]]] | None,
    gradient_lens: Lens[LJRadiusInp, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create a standard Lennard-Jones potential with sharp cutoff."""
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=neighborlist_view,
        probe=probe,
    )
    composer = LocalSumComposer(
        GraphInputConstructor(
            graph_constructor=graph_fn,
            parameter_view=parameter_view,
        )
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


type PCLJInp = GraphPotentialInput[
    PairTailCorrectedLennardJonesParameters,
    IsLJGraphParticles,
    HasCell[AnyPeriodicity],
    Literal[2],
]


def make_pair_tail_corrected_lennard_jones_potential[
    State,
    Ptch: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsLJGraphParticles]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    parameter_view: View[State, PairTailCorrectedLennardJonesParameters],
    probe: Probe[State, Ptch, IsGraphProbe[IsLJGraphParticles, Literal[2]]] | None,
    gradient_lens: Lens[PCLJInp, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create a Lennard-Jones potential with smooth pairwise tail correction."""
    radius_graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=neighborlist_view,
        probe=probe,
    )
    composer = LocalSumComposer(
        GraphInputConstructor(
            graph_constructor=radius_graph_fn,
            parameter_view=parameter_view,
        )
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
    HasCell[AnyPeriodicity],
    Literal[0],
]


def make_global_lennard_jones_tail_correction_potential[State, Gradients, Hessians](
    particles_view: View[State, Table[ParticleId, IsLJGraphParticles]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, GlobalTailCorrectedLennardJonesParameters],
    gradient_lens: Lens[GCLJInp, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, Patch[State]]:
    """Create analytical long-range tail correction for Lennard-Jones potential."""
    return PotentialFromEnergy(
        energy_fn=global_lennard_jones_tail_correction_energy,
        composer=FullSumComposer(
            GraphInputConstructor(
                GraphConstructor(
                    particles=particles_view,
                    systems=systems_view,
                    neighborlist=lambda _: EmptyNeighborList[Literal[0]](),
                    probe=None,
                ),
                parameter_view=parameter_view,
            )
        ),
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )


def make_global_lennard_jones_tail_correction_pressure[State](
    particles_view: View[State, Table[ParticleId, IsLJGraphParticles]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, GlobalTailCorrectedLennardJonesParameters],
) -> StateProperty[State, Table[SystemId, Array]]:
    """Create long-range pressure correction for Lennard-Jones systems."""
    graph_constructor = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda _: EmptyNeighborList[Literal[0]](),
        probe=None,
    )

    def pressure(key: Array, state: State) -> Table[SystemId, Array]:
        del key
        params = parameter_view(state)
        graph = graph_constructor(state, None)
        return global_lennard_jones_tail_correction_pressure(
            GraphPotentialInput(params, graph)
        ).data

    return pressure


if TYPE_CHECKING:
    _lj: EnergyFunction[Any, LennardJonesInput] = lennard_jones_energy
    _ptc: EnergyFunction[Any, PairTailCorrectedLennardJonesInput] = (
        pair_tail_corrected_lennard_jones_energy
    )
