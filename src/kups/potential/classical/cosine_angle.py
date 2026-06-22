# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""UFF-style cosine angle bending potential.

Reference: Rappe et al. (1992) "UFF, a Full Periodic Table Force Field"
J. Am. Chem. Soc. 114, 10024-10035. DOI: 10.1021/ja00051a040

General form:

$$
U(\\theta) = K \\left[C_0 + C_1 \\cos(\\theta) + C_2 \\cos(2\\theta)\\right]
$$

where the coefficients are computed from the equilibrium angle $\\theta_0$:

- $C_2 = \\frac{1}{4 \\sin^2(\\theta_0)}$
- $C_1 = -4 C_2 \\cos(\\theta_0)$
- $C_0 = C_2 (2 \\cos^2(\\theta_0) + 1)$

For linear angles ($\\theta_0 = 180°$), the coefficients are singular, so a special
form is used: $U(\\theta) = K (1 + \\cos\\theta)$.
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
class CosineAngleParameters:
    """UFF-style cosine angle potential parameters.

    Attributes:
        labels: Species labels, shape `(n_species,)`.
        theta0: Equilibrium angles [radians], shape `(n_species, n_species, n_species)`.
        k: Force constants [energy], shape `(n_species, n_species, n_species)`.
        linear_tol: Tolerance for detecting linear angles [radians].
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    theta0: Array  # (n_species, n_species, n_species)
    k: Array  # (n_species, n_species, n_species)
    linear_tol: Array = field(default_factory=lambda: jnp.radians(5))

    @classmethod
    def from_uff(
        cls,
        labels: tuple[str, ...],
        bond_angle: Array,
        bond_radius: Array,
        electronegativity: Array,
        effective_charge: Array,
        linear_tol: Array = jnp.radians(5),
    ) -> "CosineAngleParameters":
        r"""Create angle parameters using UFF formulas.

        Computes angle parameters from per-species atomic properties:
        - $\theta_0$ from central atom's bond angle
        - $K$ from Eq. 13 using bond lengths and effective charges

        Args:
            labels: Species labels, shape `(n_species,)`
            bond_angle: Natural valence angle [radians], shape `(n_species,)`
            bond_radius: Valence bond radii [A], shape `(n_species,)`
            electronegativity: GMP electronegativity, shape `(n_species,)`
            effective_charge: Effective atomic charge Z*, shape `(n_species,)`
            linear_tol: Tolerance for detecting linear angles [radians]

        Returns:
            CosineAngleParameters with full interaction matrices
        """
        nt = bond_angle.shape[0]

        # theta0 from central atom (index j in i-j-k)
        theta0 = bond_angle[None, :, None] * jnp.ones((nt, nt, nt))

        # Compute bond lengths for all pairs
        r_bond = compute_uff_bond_length(bond_radius, electronegativity)

        # r_ij = r_bond[i, j], r_jk = r_bond[j, k]
        r_ij = r_bond[:, :, None]
        r_jk = r_bond[None, :, :]

        # r_ik using law of cosines
        cos_theta0 = jnp.cos(theta0)
        r_ik_sq = r_ij**2 + r_jk**2 - 2 * r_ij * r_jk * cos_theta0
        r_ik = jnp.sqrt(jnp.maximum(r_ik_sq, 1e-10))

        # Force constant (Eq. 13)
        Z_i = effective_charge[:, None, None]
        Z_k = effective_charge[None, None, :]
        prefactor = 664.12 / (r_ij * r_jk)
        term = 3 * r_ij * r_jk * (1 - cos_theta0**2) - r_ik_sq * cos_theta0
        K = jnp.abs(prefactor * Z_i * Z_k / (r_ik**5) * term)

        return cls(
            labels=tuple(map(Label, labels)), theta0=theta0, k=K, linear_tol=linear_tol
        )


type CosineAngleInput = GraphPotentialInput[
    CosineAngleParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[3]
]


def _compute_cosine_coefficients(theta0: Array) -> tuple[Array, Array, Array]:
    r"""Compute Fourier coefficients $C_0$, $C_1$, $C_2$ from equilibrium angle.

    Args:
        theta0: Equilibrium angle in radians.

    Returns:
        Tuple of ($C_0$, $C_1$, $C_2$) coefficients.
    """
    sin2 = jnp.sin(theta0) ** 2
    cos_theta0 = jnp.cos(theta0)
    # Add small epsilon to avoid division by zero for linear angles
    c2 = 1.0 / (4.0 * jnp.maximum(sin2, 1e-10))
    c1 = -4.0 * c2 * cos_theta0
    c0 = c2 * (2.0 * cos_theta0**2 + 1.0)
    return c0, c1, c2


def cosine_angle_energy(
    inp: CosineAngleInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    r"""Compute UFF-style cosine angle energy for all angles.

    Calculates energy using the general cosine form:

    $$U(\theta) = K [C_0 + C_1 \cos(\theta) + C_2 \cos(2\theta)]$$

    For near-linear angles ($\theta_0$ close to 180deg), uses:

    $$U(\theta) = K (1 + \cos\theta)$$

    Args:
        inp: Graph potential input with cosine angle parameters

    Returns:
        Total angle energy per system
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 3, (
        "Cosine angle potential only supports triplet interactions (order=3)."
    )
    edg = graph.particles[graph.edges.indices]
    edg_species = edg.labels.indices_in(inp.parameters.labels)
    s0, s1, s2 = edg_species[:, 0], edg_species[:, 1], edg_species[:, 2]
    theta0 = inp.parameters.theta0[s0, s1, s2]
    k = inp.parameters.k[s0, s1, s2]

    # Compute current angle
    v1, v2 = graph.edge_shifts[:, 0], graph.edge_shifts[:, 1]
    cos_theta = jnp.einsum("ij,ij->i", v1, v2) / (
        jnp.linalg.norm(v1, axis=-1) * jnp.linalg.norm(v2, axis=-1)
    )
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

    # Check for near-linear case (theta0 close to pi)
    is_linear = jnp.abs(theta0 - jnp.pi) < inp.parameters.linear_tol

    # Linear case: U = K*(1 + cos(theta))
    energy_linear = k * (1.0 + cos_theta)

    # General case: U = K * [C0 + C1*cos(theta) + C2*cos(2*theta)]
    # Using cos(2t) = 2cos^2(t) - 1
    c0, c1, c2 = _compute_cosine_coefficients(theta0)
    cos_2theta = 2.0 * cos_theta**2 - 1.0
    energy_general = k * (c0 + c1 * cos_theta + c2 * cos_2theta)

    edge_energy = jnp.where(is_linear, energy_linear, energy_general)

    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


def make_cosine_angle_potential[
    State,
    Ptch: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, CosineAngleParameters],
    probe: Probe[State, Ptch, IsGraphProbe[IsBondedParticles, Literal[3]]] | None,
    gradient_lens: Lens[CosineAngleInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create UFF-style cosine angle potential for explicitly defined angles.

    Args:
        particles_view: Extracts particle data (positions, species) with system index
        edge_indices_view: Extracts angle connectivity (triplets)
        systems_view: Extracts indexed system data (cell)
        parameter_view: Extracts [CosineAngleParameters][kups.potential.classical.cosine_angle.CosineAngleParameters]
        probe: Graph probe for incremental particle and neighbor-list updates
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure
        out_cache_lens: Cache location lens

    Returns:
        Cosine angle [Potential][kups.core.potential.Potential]
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[3]](
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
        energy_fn=cosine_angle_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
    return potential


if TYPE_CHECKING:
    _: EnergyFunction[Any, CosineAngleInput] = cosine_angle_energy
