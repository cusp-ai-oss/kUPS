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
    override,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Table
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
    GraphPairEnergy,
    GraphPotentialInput,
    IsParticleProbe,
)
from kups.potential.common.pair import PairEnergy
from kups.potential.common.rigid_body_composition import RigidBodyComposition

type MixingRule = Literal["lorentz_berthelot"]


@runtime_checkable
class IsLennardJonesParticles(HasPositions, HasLabels, HasSystemIndex, Protocol): ...


@runtime_checkable
class IsLJGraphParticles(
    IsLennardJonesParticles, HasInclusionIndex, HasExclusionIndex, Protocol
): ...


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


@jit
def lennard_jones_pair_kernel(
    parameters: LennardJonesParameters,
    labels_i: Index[Label],
    labels_j: Index[Label],
    rij: Array,
    r2: Array,
    system: Index[SystemId],
    /,
) -> Array:
    """Lennard-Jones pair kernel shared by graph and fused evaluation.

    Callers apply the cutoff; pair inputs broadcast to a common shape.

    Args:
        parameters: LJ mixing tables.
        labels_i: Species labels of the left atoms.
        labels_j: Species labels of the right atoms.
        rij: Difference vectors (unused).
        r2: Squared distances.
        system: Left-side system ids (unused).

    Returns:
        Pair energies.
    """
    del rij, system
    species_i = labels_i.indices_in(parameters.labels)
    species_j = labels_j.indices_in(parameters.labels)
    epsilon = parameters.epsilon[species_i, species_j]
    sigma = parameters.sigma[species_i, species_j]
    c6 = (sigma**2 / r2) ** 3
    return 4 * epsilon * (c6**2 - c6)


lennard_jones_pair = PairEnergy[
    LennardJonesParameters, IsLennardJonesParticles, Index[Label]
](
    kernel=lennard_jones_pair_kernel,
    features=lambda p: p.labels,
    cutoffs=lambda p: p.cutoff,
)
"""LJ pair term; add terms before constructing a fused evaluator."""


type LennardJonesInput = GraphPotentialInput[
    LennardJonesParameters, IsLennardJonesParticles, HasCell[AnyPeriodicity], Literal[2]
]


lennard_jones_energy = GraphPairEnergy(lennard_jones_pair)
"""Lennard-Jones graph evaluator derived from the shared pair term."""


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
    edge_energy = lennard_jones_energy.edge_energies(inp)

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


def _tail_energy(q1: Array, q2: Array) -> Array:
    """Convert the two integrated LJ contributions to energy."""
    return (8 / 3) * jnp.pi * (q2 / 3 - q1)


@jit
def global_lennard_jones_tail_correction_energy(
    inp: GlobalTailCorrectedLennardJonesInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Compute analytical long-range tail correction energy."""
    q1, q2, _volume, n_graphs = _global_tail_correction_common(inp)
    result = _tail_energy(q1, q2)
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
    probe: Probe[State, Ptch, IsParticleProbe[IsLJGraphParticles]] | None,
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
    probe: Probe[State, Ptch, IsParticleProbe[IsLJGraphParticles]] | None,
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


def _molecular_tail_coefficients[State](
    composition: RigidBodyComposition[State, HasLabels],
    particles: HasLabels,
    parameters: GlobalTailCorrectedLennardJonesParameters,
    volume: Array,
) -> Array:
    """Project the ordinary LJ tail energy onto background and template counts."""
    labels = parameters.labels
    basis = composition.sum_particles(
        jax.nn.one_hot(particles.labels.indices_in(labels), len(labels)),
        jax.nn.one_hot(composition.templates.labels.indices_in(labels), len(labels)),
    )
    w1, w2 = parameters.tail_weights
    return _tail_energy(
        jnp.einsum("gmi,gij,gnj->gmn", basis, w1, basis) / volume[:, None, None],
        jnp.einsum("gmi,gij,gnj->gmn", basis, w2, basis) / volume[:, None, None],
    )


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
    *,
    composition: RigidBodyComposition[State, HasLabels] | None = None,
) -> Potential[State, Gradients, Hessians, Patch[State]]:
    """Create the LJ tail correction, optionally prepared for rigid bodies.

    ``composition`` projects the same tail formula onto rigid-body counts once
    at construction. It requires fixed parameters and cells and no derivatives.
    """
    constructor = GraphInputConstructor(
        GraphConstructor(
            particles=particles_view,
            systems=systems_view,
            neighborlist=lambda _: EmptyNeighborList[Literal[0]](),
            probe=None,
        ),
        parameter_view=parameter_view,
    )
    if composition is not None:
        initial = composition.initial_state
        coefficients = _molecular_tail_coefficients(
            composition,
            particles_view(initial).data,
            parameter_view(initial),
            systems_view(initial).data.cell.volume,
        )
        return composition.potential(
            coefficients,
            gradient_lens(constructor(initial, None)),
            hessian_lens,
            out_cache_lens,
            patch_idx_view,
        )
    return PotentialFromEnergy(
        energy_fn=global_lennard_jones_tail_correction_energy,
        composer=FullSumComposer(constructor),
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
