# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Morse bond stretching potential.

Reference: Rappé et al. (1992) "UFF, a Full Periodic Table Force Field"
J. Am. Chem. Soc. 114, 10024-10035. DOI: 10.1021/ja00051a040

Functional form:

$$
U(r) = D \\left[1 - e^{-\\alpha(r - r_0)}\\right]^2
$$

More accurate than harmonic for large displacements with proper
dissociation behavior. Near equilibrium, Morse approximates harmonic
with force constant $k = 2 D \\alpha^2$.
"""

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import FixedEdgesNeighborList
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    Energy,
    Potential,
    PotentialOut,
)
from kups.core.typing import (
    HasCell,
    HasPositionsAndLabels,
    Label,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field
from kups.core.utils.kahan import KahanSummand
from kups.potential.classical.uff_utils import compute_uff_bond_length
from kups.potential.common.energy import (
    EnergyFunction,
    PotentialFromEnergy,
)
from kups.potential.common.graph import (
    GraphConstructor,
    GraphPotentialInput,
    IsGraphProbe,
    IsRadiusGraphPoints,
    LocalGraphSumComposer,
)


@runtime_checkable
class IsBondedParticles(HasPositionsAndLabels, IsRadiusGraphPoints, Protocol):
    """Particle data with positions, labels, and system index."""

    ...


@dataclass
class MorseBondParameters:
    r"""Morse bond potential parameters.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        r0: Equilibrium bond lengths [Å], shape `(n_species, n_species)`.
        D: Bond dissociation energy (well depth), shape `(n_species, n_species)`.
        alpha: Width parameter [Å⁻¹], shape `(n_species, n_species)`.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    r0: Array  # (n_species, n_species)
    D: Array  # (n_species, n_species)
    alpha: Array  # (n_species, n_species)

    @classmethod
    def from_harmonic(
        cls, labels: tuple[str, ...], r0: Array, k: Array, D: Array
    ) -> "MorseBondParameters":
        r"""Create Morse parameters from harmonic force constant.

        Args:
            labels: Species labels, shape `(n_species,)`
            r0: Equilibrium bond lengths [Å], shape `(n_species, n_species)`
            k: Harmonic force constants [energy/Å²], shape `(n_species, n_species)`
            D: Bond dissociation energies, shape `(n_species, n_species)`

        Returns:
            MorseBondParameters with computed alpha values
        """
        alpha = jnp.sqrt(k / D)
        return cls(labels=tuple(map(Label, labels)), r0=r0, D=D, alpha=alpha)

    @classmethod
    def from_uff(
        cls,
        labels: tuple[str, ...],
        bond_radius: Array,
        electronegativity: Array,
        effective_charge: Array,
        dissociation_energy: Array,
    ) -> "MorseBondParameters":
        r"""Create Morse parameters using UFF bond length/force constant formulas.

        Args:
            labels: Species labels, shape `(n_species,)`
            bond_radius: Valence bond radii [Å], shape `(n_species,)`
            electronegativity: GMP electronegativity, shape `(n_species,)`
            effective_charge: Effective atomic charge Z*, shape `(n_species,)`
            dissociation_energy: Bond dissociation energy D, shape `(n_species, n_species)`

        Returns:
            MorseBondParameters with full interaction matrices
        """
        r0 = compute_uff_bond_length(bond_radius, electronegativity)
        Z_i, Z_j = effective_charge[:, None], effective_charge[None, :]

        # Force constant (Eq. 6): k = 664.12 * Z_i * Z_j / r_ij^3
        k = 664.12 * Z_i * Z_j / (r0**3)
        alpha = jnp.sqrt(k / (2.0 * dissociation_energy))

        return cls(
            labels=tuple(map(Label, labels)), r0=r0, D=dissociation_energy, alpha=alpha
        )


type MorseBondInput = GraphPotentialInput[
    MorseBondParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[2]
]


def morse_bond_energy(
    inp: MorseBondInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    r"""Compute Morse bond energy for all bonds.

    Calculates energy as $D [1 - e^{-\alpha(r - r_0)}]^2$ for each bond.

    Args:
        inp: Graph potential input with Morse bond parameters

    Returns:
        Total bond energy per system
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 2, (
        "Morse bond potential only supports pairwise interactions (order=2)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    r0 = inp.parameters.r0[edg_species[:, 0], edg_species[:, 1]]
    D = inp.parameters.D[edg_species[:, 0], edg_species[:, 1]]
    alpha = inp.parameters.alpha[edg_species[:, 0], edg_species[:, 1]]
    r = jnp.linalg.norm(graph.edge_shifts[:, 0], axis=-1)
    edge_energy = D * (1 - jnp.exp(-alpha * (r - r0))) ** 2
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


def make_morse_bond_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, MorseBondParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]] | None,
    gradient_lens: Lens[MorseBondInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create Morse bond potential for explicitly defined bonds.

    Args:
        particles_view: Extracts particle data (positions, species) with system index
        edge_indices_view: Extracts bond connectivity
        systems_view: Extracts indexed system data (cell)
        parameter_view: Extracts [MorseBondParameters][kups.potential.classical.morse.MorseBondParameters]
        probe: Graph probe for incremental particle and neighbor-list updates
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure
        out_cache_lens: Cache location lens

    Returns:
        Morse bond [Potential][kups.core.potential.Potential]
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[2]](
            edge_indices_view(state)
        ),
        probe=probe,
    )
    composer = LocalGraphSumComposer(
        graph_constructor=graph_fn,
        parameter_view=parameter_view,
    )
    potential = PotentialFromEnergy(
        composer=composer,
        energy_fn=morse_bond_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
    return potential


if TYPE_CHECKING:
    _: EnergyFunction[Any, MorseBondInput] = morse_bond_energy
