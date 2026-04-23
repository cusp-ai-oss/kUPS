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

from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

import jax.numpy as jnp
from jax import Array

from kups.core.data import Table
from kups.core.lens import Lens, SimpleLens, View
from kups.core.neighborlist import Edges
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Energy,
    Potential,
    PotentialOut,
    empty_patch_idx_view,
)
from kups.core.typing import (
    HasCache,
    HasPositionsAndLabels,
    HasSystemIndex,
    HasUnitCell,
    Label,
    MaybeCached,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field
from kups.potential.classical.uff_utils import compute_uff_bond_length
from kups.potential.common.energy import (
    EnergyFunction,
    PositionAndUnitCell,
    PotentialFromEnergy,
    position_and_unitcell_idx_view,
)
from kups.potential.common.graph import (
    EdgeSetGraphConstructor,
    GraphPotentialInput,
    IsEdgeSetGraphProbe,
    LocalGraphSumComposer,
)


@runtime_checkable
class IsBondedParticles(HasPositionsAndLabels, HasSystemIndex, Protocol):
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
    MorseBondParameters, IsBondedParticles, HasUnitCell, Literal[2]
]


def morse_bond_energy(
    inp: MorseBondInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
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
    return WithPatch(total_energies, IdPatch())


def make_morse_bond_potential[
    State,
    P: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edges_view: View[State, Edges[Literal[2]]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    parameter_view: View[State, MorseBondParameters],
    probe: Probe[State, P, IsEdgeSetGraphProbe[IsBondedParticles, Literal[2]]] | None,
    gradient_lens: Lens[MorseBondInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create Morse bond potential for explicitly defined bonds.

    Args:
        particles_view: Extracts particle data (positions, species) with system index
        edges_view: Extracts bond connectivity
        systems_view: Extracts indexed system data (unit cell)
        parameter_view: Extracts [MorseBondParameters][kups.potential.classical.morse.MorseBondParameters]
        probe: Grouped probe for incremental updates (particles, edges, capacity)
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure
        out_cache_lens: Cache location lens

    Returns:
        Morse bond [Potential][kups.core.potential.Potential]
    """
    graph_fn = EdgeSetGraphConstructor(
        particles=particles_view,
        edges=edges_view,
        systems=systems_view,
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


class IsMorseBondState[Params](Protocol):
    """Protocol for states providing all inputs for the Morse bond potential."""

    @property
    def particles(self) -> Table[ParticleId, IsBondedParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...
    @property
    def bond_edges(self) -> Edges[Literal[2]]: ...
    @property
    def morse_bond_parameters(self) -> Params: ...


@overload
def make_morse_bond_from_state[State](
    state: Lens[State, IsMorseBondState[MaybeCached[MorseBondParameters, Any]]],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch]: ...


@overload
def make_morse_bond_from_state[State](
    state: Lens[State, IsMorseBondState[MaybeCached[MorseBondParameters, Any]]],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, Patch]: ...


@overload
def make_morse_bond_from_state[State, P: Patch](
    state: Lens[
        State,
        IsMorseBondState[
            HasCache[MorseBondParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[
        State,
        P,
        IsEdgeSetGraphProbe[IsBondedParticles, Literal[2]],
    ],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_morse_bond_from_state[State, P: Patch](
    state: Lens[
        State,
        IsMorseBondState[
            HasCache[MorseBondParameters, PotentialOut[PositionAndUnitCell, EmptyType]]
        ],
    ],
    probe: Probe[
        State,
        P,
        IsEdgeSetGraphProbe[IsBondedParticles, Literal[2]],
    ],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, P]: ...


def make_morse_bond_from_state(
    state: Any,
    probe: Any = None,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create a Morse bond potential from a typed state, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, unit cell, edges,
            and Morse bond parameters.
        probe: Detects which particles and edges changed since the last step.
            If ``None``, no incremental updates are used.
        compute_position_and_unitcell_gradients: When ``True``, the returned
            potential computes gradients w.r.t. particle positions and lattice
            vectors (for forces / stress).

    Returns:
        Configured Morse bond [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[MorseBondInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.unitcell),
            )
        )
        patch_idx_view = position_and_unitcell_idx_view
    param_view = state.focus(
        lambda x: (
            x.morse_bond_parameters.data
            if isinstance(x.morse_bond_parameters, HasCache)
            else x.morse_bond_parameters
        )
    )
    cache_view = None
    if probe is not None:
        param_view = state.focus(lambda x: x.morse_bond_parameters.data)
        cache_view = state.focus(lambda x: x.morse_bond_parameters.cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_morse_bond_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.bond_edges),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
        cache_view,
    )


if TYPE_CHECKING:
    _: EnergyFunction = morse_bond_energy
