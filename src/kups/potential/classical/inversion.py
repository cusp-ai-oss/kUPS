# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""UFF-style inversion (out-of-plane/improper) potential.

Reference: Rappe et al. (1992) "UFF, a Full Periodic Table Force Field"
J. Am. Chem. Soc. 114, 10024-10035. DOI: 10.1021/ja00051a040

Functional form (Eq. 18):

$$
E_\\omega = K_{IJKL} \\left[C_0 + C_1 \\cos(\\omega) + C_2 \\cos(2\\omega)\\right]
$$

The inversion center is atom I (index 0) with three neighbors J, K, L. The angle
$\\omega$ is between the I-J bond and the I-K-L plane.

For sp2 centers ($\\omega_0 = 0$): $C_0=1$, $C_1=-1$, $C_2=0$, giving:

$$E = K(1 - \\cos\\omega)$$

where K is the force constant (6 kcal/mol for sp2 carbon, 50 for carbonyl).

For non-sp2 centers (e.g., Group 5 atoms like PH3), coefficients are fit such
that $E(\\omega_0)=0$ (minimum at equilibrium) and $E(0)=E_{barrier}$:

- $C_2 = \\frac{1}{4 \\sin^2(\\omega_0)}$
- $C_1 = -4 C_2 \\cos(\\omega_0)$
- $C_0 = C_2 (2 \\cos^2(\\omega_0) + 1)$
- K is scaled internally so that E(0) equals the barrier parameter

Note: The UFF paper specifies that when all 3 inversions per center are used
(IJ/IKL, IK/IJL, IL/IJK), each barrier should be divided by 3.
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
class InversionParameters:
    r"""UFF-style inversion potential parameters.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        omega0: Equilibrium out-of-plane angles [radians],
            shape `(n_species, n_species, n_species, n_species)`.
        k: Force constant or barrier [energy],
            shape `(n_species, n_species, n_species, n_species)`.
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    omega0: Array  # (n_species, n_species, n_species, n_species)
    k: Array  # (n_species, n_species, n_species, n_species)

    @classmethod
    def from_uff(
        cls,
        labels: tuple[str, ...],
        inversion_barrier: Array,
        omega0: Array | None = None,
    ) -> "InversionParameters":
        r"""Create inversion parameters using UFF formulas.

        Args:
            labels: Species labels, shape `(n_species,)`.
            inversion_barrier: Barrier/force constant for each type, shape `(n_species,)`.
            omega0: Equilibrium angle [radians], shape `(n_species,)`.

        Returns:
            InversionParameters with full interaction matrices
        """
        nt = inversion_barrier.shape[0]

        # Broadcast to 4D: center is first index
        k_arr = inversion_barrier[:, None, None, None] * jnp.ones((nt, nt, nt, nt))

        if omega0 is None:
            omega0_arr = jnp.zeros((nt, nt, nt, nt))
        else:
            omega0_arr = omega0[:, None, None, None] * jnp.ones((nt, nt, nt, nt))

        return cls(labels=tuple(map(Label, labels)), omega0=omega0_arr, k=k_arr)


type InversionInput = GraphPotentialInput[
    InversionParameters, IsBondedParticles, HasUnitCell, Literal[4]
]


def inversion_energy(
    inp: InversionInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    r"""Compute UFF-style inversion energy for all inversion centers.

    Args:
        inp: Graph potential input with inversion parameters

    Returns:
        Total inversion energy per system
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 4, (
        "Inversion potential requires 4-body interactions (order=4)."
    )
    edg = graph.particles[graph.edges.indices]
    edg_species = edg.labels.indices_in(inp.parameters.labels)
    # Edge indices: [:, 0] = center j, [:, 1] = i, [:, 2] = k, [:, 3] = l
    s_j, s_i, s_k, s_l = (
        edg_species[:, 0],
        edg_species[:, 1],
        edg_species[:, 2],
        edg_species[:, 3],
    )

    omega0 = inp.parameters.omega0[s_j, s_i, s_k, s_l]
    k = inp.parameters.k[s_j, s_i, s_k, s_l]

    # Get displacement vectors from edge_shifts
    r_ji = graph.edge_shifts[:, 0]  # (n_edges, 3)
    r_jk = graph.edge_shifts[:, 1]  # (n_edges, 3)
    r_jl = graph.edge_shifts[:, 2]  # (n_edges, 3)

    # Normal to the plane k-j-l
    n = jnp.cross(r_jk, r_jl)  # (n_edges, 3)
    n_norm = jnp.linalg.norm(n, axis=-1, keepdims=True)
    eps = 1e-10
    n_unit = n / (n_norm + eps)

    # Out-of-plane angle: sin(omega) = (r_ji . n) / |r_ji|
    r_ji_norm = jnp.linalg.norm(r_ji, axis=-1)
    sin_omega = jnp.einsum("ij,ij->i", r_ji, n_unit) / (r_ji_norm + eps)
    sin_omega = jnp.clip(sin_omega, -1.0, 1.0)
    omega = jnp.arcsin(sin_omega)  # radians

    # Compute UFF coefficients
    sin2_omega0 = jnp.sin(omega0) ** 2
    cos_omega0 = jnp.cos(omega0)

    # Handle sp2 case (omega0 ~ 0) separately to avoid division by zero
    is_sp2 = jnp.abs(omega0) < 1e-6
    c2 = jnp.where(
        is_sp2, 0.0, 1.0 / (4.0 * jnp.where(sin2_omega0 > 1e-10, sin2_omega0, 1.0))
    )
    c1 = jnp.where(is_sp2, -1.0, -4.0 * c2 * cos_omega0)
    c0 = jnp.where(is_sp2, 1.0, c2 * (2.0 * cos_omega0**2 + 1.0))

    # Energy: U = K * [C0 + C1*cos(omega) + C2*cos(2*omega)]
    cos_omega = jnp.cos(omega)
    cos_2omega = jnp.cos(2.0 * omega)

    # For non-sp2, k is the barrier E_barrier at omega=0
    barrier_factor = c0 + c1 + c2
    k_scaled = jnp.where(
        is_sp2,
        k,
        k / jnp.maximum(barrier_factor, 1e-10),
    )
    edge_energy = k_scaled * (c0 + c1 * cos_omega + c2 * cos_2omega)

    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch())


def make_inversion_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edges_view: View[State, Edges[Literal[4]]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    parameter_view: View[State, InversionParameters],
    probe: Probe[State, Ptch, IsEdgeSetGraphProbe[IsBondedParticles, Literal[4]]]
    | None,
    gradient_lens: Lens[InversionInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create UFF-style inversion potential for sp2/sp3 centers.

    Args:
        particles_view: Extracts particle data (positions, species) with system index
        edges_view: Extracts inversion connectivity (4-tuples: center + 3 neighbors)
        systems_view: Extracts indexed system data (unit cell)
        parameter_view: Extracts [InversionParameters][kups.potential.classical.inversion.InversionParameters]
        probe: Grouped probe for incremental updates (particles, edges, capacity)
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure
        out_cache_lens: Cache location lens

    Returns:
        Inversion [Potential][kups.core.potential.Potential]
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
        energy_fn=inversion_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
    return potential


class IsInversionState[Params](Protocol):
    """Protocol for states providing all inputs for the inversion potential."""

    @property
    def particles(self) -> Table[ParticleId, IsBondedParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...
    @property
    def inversion_edges(self) -> Edges[Literal[4]]: ...
    @property
    def inversion_parameters(self) -> Params: ...


@overload
def make_inversion_from_state[
    State,
    InState: IsInversionState[InversionParameters | HasCache[InversionParameters, Any]],
](
    state: Lens[State, InState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch]: ...


@overload
def make_inversion_from_state[
    State,
    InState: IsInversionState[InversionParameters | HasCache[InversionParameters, Any]],
](
    state: Lens[State, InState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, Patch]: ...


@overload
def make_inversion_from_state[
    State,
    InState: IsInversionState[
        HasCache[InversionParameters, PotentialOut[EmptyType, EmptyType]]
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
def make_inversion_from_state[
    State,
    InState: IsInversionState[
        HasCache[InversionParameters, PotentialOut[PositionAndUnitCell, EmptyType]]
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


def make_inversion_from_state(
    state: Any,
    probe: Any = None,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create an inversion potential, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, unit cell, edges,
            and inversion parameters.
        probe: Detects which particles and edges changed since the last step.
            None for no incremental updates.
        compute_position_and_unitcell_gradients: When True, computes gradients
            w.r.t. particle positions and lattice vectors.

    Returns:
        Configured inversion [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[InversionInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.unitcell),
            )
        )
        patch_idx_view = position_and_unitcell_idx_view
    param_view = state.focus(
        lambda x: (
            x.inversion_parameters.data
            if isinstance(x.inversion_parameters, HasCache)
            else x.inversion_parameters
        )
    )
    cache_view = None
    if probe is not None:
        param_view = state.focus(lambda x: x.inversion_parameters.data)
        cache_view = state.focus(lambda x: x.inversion_parameters.cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_inversion_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.inversion_edges),
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
    _: EnergyFunction = inversion_energy
