# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dihedral/torsion potential as defined in UFF.

Reference: Rappe et al. (1992) "UFF, a Full Periodic Table Force Field"
J. Am. Chem. Soc. 114, 10024-10035. DOI: 10.1021/ja00051a040

Functional form:

$$
U(\\phi) = \\frac{1}{2} V_\\phi \\left[1 - \\cos(n \\phi_0) \\cos(n \\phi)\\right]
$$

where:

- $\\phi$ is the dihedral angle (angle between planes i-j-k and j-k-l)
- $V_\\phi$ is the barrier height
- $n$ is the periodicity (typically 1, 2, 3, or 6)
- $\\phi_0$ is the equilibrium dihedral angle
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
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field
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
class DihedralParameters:
    """UFF dihedral/torsion potential parameters.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        V: Barrier heights [energy], shape `(n_species, n_species, n_species, n_species)`.
        n: Periodicities, shape `(n_species, n_species, n_species, n_species)`.
        phi0: Equilibrium dihedral angles [radians],
            shape `(n_species, n_species, n_species, n_species)`.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    V: Array  # (n_species, n_species, n_species, n_species)
    n: Array  # (n_species, n_species, n_species, n_species)
    phi0: Array  # (n_species, n_species, n_species, n_species)

    @classmethod
    def from_uff(
        cls,
        labels: tuple[str, ...],
        bond_angle: Array,
        torsion_sp3: Array,
        torsion_sp2: Array,
        group: Array,
        bond_order: Array | None = None,
        hybridization_tol: float = 5.0 * 3.141592653589793 / 180.0,
    ) -> "DihedralParameters":
        r"""Create dihedral parameters using UFF formulas.

        Computes torsion parameters based on central bond hybridization following
        the UFF paper (Rappe et al. 1992). For dihedral i-j-k-l, parameters depend
        on the central j-k bond and neighboring atoms.

        Args:
            labels: Species labels, shape `(n_species,)`.
            bond_angle: Natural valence angle [radians], shape `(n_species,)`.
            torsion_sp3: sp3 torsional barrier V [kcal/mol], shape `(n_species,)`.
            torsion_sp2: sp2 torsional parameter U [kcal/mol], shape `(n_species,)`.
            group: Periodic table group (1-18), shape `(n_species,)`.
            bond_order: Bond order for j-k bond, shape `(n_species, n_species)`.
            hybridization_tol: Tolerance for hybridization detection [radians].

        Returns:
            DihedralParameters with full 4D interaction matrices.
        """
        nt = bond_angle.shape[0]

        # Determine hybridization from bond angle
        is_sp2 = jnp.abs(bond_angle - jnp.radians(120.0)) < hybridization_tol
        is_sp3 = ~(jnp.abs(bond_angle - jnp.pi) < hybridization_tol) & ~is_sp2

        # Group 6 and main group detection
        is_group6 = group == 6
        is_main_group = group > 0

        # Broadcast to 4D for dihedral i-j-k-l
        is_sp2_i = is_sp2[:, None, None, None]
        is_sp3_j = is_sp3[None, :, None, None]
        is_sp3_k = is_sp3[None, None, :, None]
        is_sp2_j = is_sp2[None, :, None, None]
        is_sp2_k = is_sp2[None, None, :, None]
        is_sp2_l = is_sp2[None, None, None, :]

        is_group6_j = is_group6[None, :, None, None]
        is_group6_k = is_group6[None, None, :, None]
        is_main_j = is_main_group[None, :, None, None]
        is_main_k = is_main_group[None, None, :, None]

        # Torsional parameters broadcast to 4D
        V_j = torsion_sp3[None, :, None, None]
        V_k = torsion_sp3[None, None, :, None]
        U_j = torsion_sp2[None, :, None, None]
        U_k = torsion_sp2[None, None, :, None]

        # Bond order for central j-k bond
        if bond_order is None:
            BO_jk = jnp.ones((1, nt, nt, 1))
        else:
            BO_jk = bond_order[None, :, :, None]

        # V from Eq.17: V = 5 * sqrt(U_j * U_k) * (1 + 4.18 * ln(BO))
        V_eq17 = (
            5.0
            * jnp.sqrt(jnp.abs(U_j * U_k))
            * (1.0 + 4.18 * jnp.log(jnp.maximum(BO_jk, 1e-10)))
        )

        # Initialize arrays
        V_arr = jnp.zeros((nt, nt, nt, nt))
        n_arr = jnp.ones((nt, nt, nt, nt), dtype=jnp.int32)
        phi0_arr = jnp.zeros((nt, nt, nt, nt))

        # === Case 1: sp3-sp3 general ===
        sp3_sp3 = is_sp3_j & is_sp3_k
        V_arr = jnp.where(sp3_sp3, jnp.sqrt(jnp.abs(V_j * V_k)), V_arr)
        n_arr = jnp.where(sp3_sp3, 3, n_arr)
        phi0_arr = jnp.where(sp3_sp3, jnp.pi, phi0_arr)

        # === Case 2: sp3-sp3 Group 6 pair (override n and phi0) ===
        sp3_sp3_group6 = sp3_sp3 & is_group6_j & is_group6_k
        n_arr = jnp.where(sp3_sp3_group6, 2, n_arr)
        phi0_arr = jnp.where(sp3_sp3_group6, jnp.pi / 2, phi0_arr)

        # === Case 3: sp2-sp2 ===
        sp2_sp2 = is_sp2_j & is_sp2_k
        V_arr = jnp.where(sp2_sp2, V_eq17, V_arr)
        n_arr = jnp.where(sp2_sp2, 2, n_arr)
        phi0_arr = jnp.where(sp2_sp2, jnp.pi, phi0_arr)

        # === Case 4: sp3-sp2 general ===
        sp3_sp2 = (is_sp3_j & is_sp2_k) | (is_sp2_j & is_sp3_k)
        V_arr = jnp.where(sp3_sp2, 1.0, V_arr)
        n_arr = jnp.where(sp3_sp2, 6, n_arr)
        phi0_arr = jnp.where(sp3_sp2, 0.0, phi0_arr)

        # === Case 5: sp3-sp2 propene-like (sp2 bonded to another sp2) ===
        propene = (is_sp2_j & is_sp3_k & is_sp2_i) | (is_sp3_j & is_sp2_k & is_sp2_l)
        V_arr = jnp.where(propene, 2.0, V_arr)
        n_arr = jnp.where(propene, 3, n_arr)
        phi0_arr = jnp.where(propene, jnp.pi, phi0_arr)

        # === Case 6: sp3-sp2 with Group 6 sp3 ===
        group6_sp3_sp2 = (is_sp3_j & is_group6_j & is_sp2_k) | (
            is_sp2_j & is_sp3_k & is_group6_k
        )
        V_arr = jnp.where(group6_sp3_sp2, V_eq17, V_arr)
        n_arr = jnp.where(group6_sp3_sp2, 2, n_arr)
        phi0_arr = jnp.where(group6_sp3_sp2, jnp.pi / 2, phi0_arr)

        # === Case 7: Non-main-group (V=0) ===
        non_main = ~is_main_j | ~is_main_k
        V_arr = jnp.where(non_main, 0.0, V_arr)

        return cls(labels=tuple(map(Label, labels)), V=V_arr, n=n_arr, phi0=phi0_arr)


type DihedralInput = GraphPotentialInput[
    DihedralParameters, IsBondedParticles, HasUnitCell, Literal[4]
]


def dihedral_energy(
    inp: DihedralInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    r"""Compute UFF dihedral/torsion energy for all dihedrals.

    Args:
        inp: Graph potential input with dihedral parameters

    Returns:
        Total dihedral energy per system
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 4, (
        "Dihedral potential only supports quadruplet interactions (order=4)."
    )
    edg = graph.particles[graph.edges.indices]
    edg_species = edg.labels.indices_in(inp.parameters.labels)
    s0, s1, s2, s3 = (
        edg_species[:, 0],
        edg_species[:, 1],
        edg_species[:, 2],
        edg_species[:, 3],
    )

    V = inp.parameters.V[s0, s1, s2, s3]
    n = inp.parameters.n[s0, s1, s2, s3]
    phi0 = inp.parameters.phi0[s0, s1, s2, s3]

    # Edge shifts: all relative to first atom (i); convert to bond vectors
    r_ij = graph.edge_shifts[:, 0]  # j - i
    r_jk = graph.edge_shifts[:, 1] - r_ij  # k - j
    r_kl = graph.edge_shifts[:, 2] - graph.edge_shifts[:, 1]  # l - k

    # Compute normal vectors to planes
    n1 = jnp.cross(r_ij, r_jk)  # Normal to plane i-j-k
    n2 = jnp.cross(r_jk, r_kl)  # Normal to plane j-k-l

    # Normalize the normal vectors
    n1_norm = jnp.linalg.norm(n1, axis=-1, keepdims=True)
    n2_norm = jnp.linalg.norm(n2, axis=-1, keepdims=True)

    eps = 1e-10
    n1_normalized = n1 / (n1_norm + eps)
    n2_normalized = n2 / (n2_norm + eps)

    # Compute dihedral angle using atan2 for proper sign
    cos_phi = jnp.sum(n1_normalized * n2_normalized, axis=-1)
    r_jk_norm = jnp.linalg.norm(r_jk, axis=-1, keepdims=True)
    r_jk_normalized = r_jk / (r_jk_norm + eps)
    sin_phi = jnp.sum(
        jnp.cross(n1_normalized, n2_normalized) * r_jk_normalized, axis=-1
    )

    phi = jnp.arctan2(sin_phi, cos_phi)

    # UFF torsion energy: U(phi) = 1/2 V [1 - cos(n*phi0)*cos(n*phi)]
    edge_energy = 0.5 * V * (1.0 - jnp.cos(n * phi0) * jnp.cos(n * phi))

    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch())


def make_dihedral_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edges_view: View[State, Edges[Literal[4]]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    parameter_view: View[State, DihedralParameters],
    probe: Probe[State, Ptch, IsEdgeSetGraphProbe[IsBondedParticles, Literal[4]]]
    | None,
    gradient_lens: Lens[DihedralInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create UFF dihedral potential for explicitly defined dihedrals.

    Args:
        particles_view: Extracts particle data (positions, species) with system index
        edges_view: Extracts dihedral connectivity (quadruplets)
        systems_view: Extracts indexed system data (unit cell)
        parameter_view: Extracts [DihedralParameters][kups.potential.classical.dihedral.DihedralParameters]
        probe: Grouped probe for incremental updates (particles, edges, capacity)
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure
        out_cache_lens: Cache location lens

    Returns:
        UFF dihedral [Potential][kups.core.potential.Potential]
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
        energy_fn=dihedral_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
    return potential


class IsDihedralState[Params](Protocol):
    """Protocol for states providing all inputs for the dihedral potential."""

    @property
    def particles(self) -> Table[ParticleId, IsBondedParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...
    @property
    def dihedral_edges(self) -> Edges[Literal[4]]: ...
    @property
    def dihedral_parameters(self) -> Params: ...


@overload
def make_dihedral_from_state[
    State,
    InState: IsDihedralState[DihedralParameters | HasCache[DihedralParameters, Any]],
](
    state: Lens[State, InState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch]: ...


@overload
def make_dihedral_from_state[
    State,
    InState: IsDihedralState[DihedralParameters | HasCache[DihedralParameters, Any]],
](
    state: Lens[State, InState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, Patch]: ...


@overload
def make_dihedral_from_state[
    State,
    InState: IsDihedralState[
        HasCache[DihedralParameters, PotentialOut[EmptyType, EmptyType]]
    ],
    P: Patch,
](
    state: Lens[State, InState],
    probe: Probe[
        State,
        P,
        IsEdgeSetGraphProbe[IsBondedParticles, Literal[4]],
    ],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_dihedral_from_state[
    State,
    InState: IsDihedralState[
        HasCache[DihedralParameters, PotentialOut[PositionAndUnitCell, EmptyType]]
    ],
    P: Patch,
](
    state: Lens[State, InState],
    probe: Probe[
        State,
        P,
        IsEdgeSetGraphProbe[IsBondedParticles, Literal[4]],
    ],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, P]: ...


def make_dihedral_from_state(
    state: Any,
    probe: Any = None,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create a dihedral potential from a typed state, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, unit cell, edges,
            and dihedral parameters.
        probe: Detects which particles and edges changed since the last step.
            If None, no incremental updates are used.
        compute_position_and_unitcell_gradients: When True, computes gradients
            w.r.t. particle positions and lattice vectors.

    Returns:
        Configured dihedral [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[DihedralInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.unitcell),
            )
        )
        patch_idx_view = position_and_unitcell_idx_view
    param_view = state.focus(
        lambda x: (
            x.dihedral_parameters.data
            if isinstance(x.dihedral_parameters, HasCache)
            else x.dihedral_parameters
        )
    )
    cache_view = None
    if probe is not None:
        param_view = state.focus(lambda x: x.dihedral_parameters.data)
        cache_view = state.focus(lambda x: x.dihedral_parameters.cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_dihedral_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.dihedral_edges),
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
    _: EnergyFunction = dihedral_energy
