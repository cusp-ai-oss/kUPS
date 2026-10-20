"""A classical force field for UiO-66 + CO₂: GCMC's nonbonded terms plus UFF4MOF bonds.

Nonbonded terms are GCMC's: UFF Lennard-Jones by element and the CIF's DDEC
charges for the framework, TraPPE for CO₂, Ewald with GCMC's precision. As in
GCMC, they act between molecules only: framework-guest and guest-guest.

The framework is held together by UFF4MOF bonded terms (Rappé 1992; Addicoat
2014; Coupry 2016): UFF4MOF force constants and torsions, with every bond length
and angle taken from the CIF, as lammps-interface's ``keep_metal_geometry``
does for the metal. The crystal structure is then the framework's minimum.
Keeping UFF4MOF's 1-4 nonbonded terms and organic geometry instead strains the
DDEC-charged framework so much that NPT shrinks it by 10-25 %.

Each bonded term has its own parameters: its energy is a polynomial
``Σₘ cₘ (g - g₀)ᵐ`` in one geometric quantity ``g`` (a distance or a cosine).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass as std_dataclass
from dataclasses import replace
from typing import Any, Callable, NamedTuple

import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from ase.neighborlist import neighbor_list

from kups.application.md.data import MdState
from kups.application.potential.classical.ewald import make_ewald_from_state
from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
    make_lennard_jones_tail_correction_from_state,
)
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.utils.particles import particles_from_ase
from kups.core.cell import Cell
from kups.core.constants import KILO_CALORIE, MOL
from kups.core.data import Index, Table
from kups.core.lens import Lens, bind, const_lens
from kups.core.neighborlist import FixedEdgesNeighborList
from kups.core.patch import IdPatch, WithPatch
from kups.core.potential import EMPTY_LENS, Potential, ScaledPotential, sum_potentials
from kups.core.typing import ExclusionId, Label, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field
from kups.potential.classical import EwaldParameters
from kups.potential.classical.ewald.parameters import estimate_ewald_parameters
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
)
from kups.potential.common.energy import LocalSumComposer, PotentialFromEnergy
from kups.potential.common.geometry import (
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import (
    GRAPH_GEOMETRY,
    GraphConstructor,
    GraphInputConstructor,
)

from . import EXAMPLES
from .structure import MAX_CO2

CIF = EXAMPLES / "host" / "RUBTAK.cif"
# Å, LJ and Ewald real space. GCMC's 6 Å is just under half the cell's width:
# NPT's fluctuations would cross it and drag in periodic images.
CUTOFF = 5.5
EWALD_PRECISION = 1e-6  # as in GCMC
KCAL = KILO_CALORIE / MOL  # eV per kcal/mol
# UFF4MOF atom types: bond radius (Å), angle (°), effective charge, sp² torsion
# barrier, electronegativity.
UFF = {
    "H_": (0.354, 180.0, 0.712, 0.0, 4.528),
    "C_R": (0.729, 120.0, 1.912, 2.0, 5.343),
    "O_2": (0.634, 120.0, 2.300, 2.0, 8.741),
    "O_3_f": (0.634, 109.47, 2.300, 2.0, 8.741),
    "Zr8f4": (1.680, 109.47, 3.667, 0.2, 3.400),
}
# sp² carbon, split over 3 permutations; UFF stiffens it next to an O_2.
INVERSION = {"default": 6 * KCAL / 3, "O_2": 50 * KCAL / 3}
# Flexible TraPPE CO₂: the rigid model's bond length, with stretch and bend
# stiffness matching CO₂'s 2349 cm⁻¹ and 667 cm⁻¹ modes.
CO2_BOND, CO2_STRETCH, CO2_BEND = 1.16, 88.5, 4.8  # Å, eV/Å², eV
CO2_LABELS = ("C_co2", "O_co2", "O_co2")
CO2_CHARGES = (0.7, -0.35, -0.35)

HOST = ase.io.read(CIF, store_tags=True)
assert isinstance(HOST, ase.Atoms)
N_FRAMEWORK = len(HOST)
LABELS = (*HOST.info["_atom_site_label"], *CO2_LABELS * MAX_CO2)
CHARGES = np.array([*HOST.info["_atom_site_charge"], *CO2_CHARGES * MAX_CO2])
SLOTS = np.repeat(np.arange(MAX_CO2), 3)  # CO₂ slot of each guest atom
GRADIENT: Lens[Any, PositionsAndCell] = GRAPH_GEOMETRY.nest(POSITIONS_AND_CELL)


def _cosine(u: jax.Array, v: jax.Array) -> jax.Array:
    return (u * v).sum(-1) / jnp.sqrt((u * u).sum(-1) * (v * v).sum(-1))


def distance(x: jax.Array) -> jax.Array:
    return jnp.linalg.norm(x[:, 0], axis=-1)


def cos_angle(x: jax.Array) -> jax.Array:
    """Cosine of the angle at the first atom of ``(centre, a, b)``."""
    return _cosine(x[:, 0], x[:, 1])


def cos_torsion(x: jax.Array) -> jax.Array:
    """Cosine of the torsion along the chain ``(a, b, c, d)``."""
    b1, b2, b3 = x[:, 0], x[:, 1] - x[:, 0], x[:, 2] - x[:, 1]
    return _cosine(jnp.cross(b1, b2), jnp.cross(b2, b3))


def cos_inversion(x: jax.Array) -> jax.Array:
    """Cosine of the angle between ``centre-i`` and the plane ``(centre, j, k)``."""
    sin = _cosine(x[:, 0], jnp.cross(x[:, 1], x[:, 2]))
    return jnp.sqrt(1 - sin**2)


@dataclass
class Polynomial:
    """Per-system sum of each term's ``Σₘ cₘ (g - g₀)ᵐ``.

    Terms of padding atoms (outside every system) are evaluated on a regular
    stand-in geometry, so their discarded energies and gradients stay finite.
    """

    geometry: Callable[[jax.Array], jax.Array] = field(static=True)

    def __call__(self, inp: Any) -> WithPatch[Table[SystemId, jax.Array], IdPatch[Any]]:
        (g0, coefficients), graph = inp
        system = graph.edge_batch_mask
        stand_in = jnp.eye(3)[: graph.edge_shifts.shape[1]]
        x = jnp.where(system.valid_mask[:, None, None], graph.edge_shifts, stand_in)
        g = self.geometry(x) - g0
        energy = coefficients[:, -1]
        for c in coefficients.T[-2::-1]:
            energy = energy * g + c
        return WithPatch(system.sum_over(energy), IdPatch[Any]())


@std_dataclass(frozen=True)
class Terms:
    """Bonded terms of one kind: particle tuples and per-term parameters."""

    geometry: Callable[[jax.Array], jax.Array]
    indices: list[tuple[int, ...]]
    g0: list[float]
    coefficients: list[list[float]]

    @classmethod
    def of(cls, geometry: Callable[[jax.Array], jax.Array]) -> Terms:
        return cls(geometry, [], [], [])

    def add(self, indices: tuple[int, ...], g0: float, *coefficients: float) -> None:
        self.indices.append(indices)
        self.g0.append(g0)
        self.coefficients.append(list(coefficients))

    def potential[S](self, state: Lens[S, MdState]) -> Potential[S, Any, Any, Any]:
        keys = tuple(map(ParticleId, range(len(LABELS))))
        edges = Index(keys, jnp.asarray(self.indices))
        width = max(map(len, self.coefficients))
        coefficients = [c + [0.0] * (width - len(c)) for c in self.coefficients]
        parameters = (jnp.asarray(self.g0), jnp.asarray(coefficients))
        graph = GraphConstructor(
            particles=state.focus(lambda s: s.particles),
            systems=state.focus(lambda s: s.systems),
            neighborlist=lambda _: FixedEdgesNeighborList(edges),
            probe=None,
        )
        return PotentialFromEnergy(
            energy_fn=Polynomial(self.geometry),
            composer=LocalSumComposer(
                GraphInputConstructor(graph, const_lens(parameters))
            ),
            gradient_lens=GRADIENT,
            hessian_lens=EMPTY_LENS,
            hessian_idx_view=EMPTY_LENS,
            patch_idx_view=position_and_cell_idx_view,
            cache_lens=None,
        )


def _bond_length(t1: str, t2: str, order: float) -> float:
    """UFF r₀, with bond-order and electronegativity corrections."""
    (r1, *_, x1), (r2, *_, x2) = UFF[t1], UFF[t2]
    r_bo = -0.1332 * (r1 + r2) * math.log(order)
    r_en = r1 * r2 * (math.sqrt(x1) - math.sqrt(x2)) ** 2 / (x1 * r1 + x2 * r2)
    return r1 + r2 + r_bo - r_en


def bonded_terms() -> list[Terms]:
    """UFF4MOF bonds, angles, torsions and inversions of the CIF's framework,
    and a flexible CO₂ in every slot."""
    symbols = HOST.get_chemical_symbols()
    radii = {"H": 0.31, "C": 0.76, "O": 0.66, "Zr": 1.5}  # Zr as lammps-interface
    i, j, d = neighbor_list("ijD", HOST, [1.15 * radii[s] for s in symbols])
    vectors: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for a, b, v in zip(i.tolist(), j.tolist(), d):
        if (symbols[a], symbols[b]) != ("Zr", "Zr"):
            vectors[a][b] = v
    neighbours = {a: sorted(vs) for a, vs in sorted(vectors.items())}

    def uff_type(a: int) -> str:
        oxygen = "O_2" if "C" in {symbols[b] for b in neighbours[a]} else "O_3_f"
        return {"Zr": "Zr8f4", "C": "C_R", "H": "H_", "O": oxygen}[symbols[a]]

    types = [uff_type(a) for a in range(N_FRAMEWORK)]
    carboxyl = {a for a in neighbours if {types[b] for b in neighbours[a]} >= {"O_2"}}

    def order(a: int, b: int) -> float:
        pair = {types[a], types[b]}
        if "Zr8f4" in pair:
            return 0.5
        aromatic = pair == {"C_R"} and not {a, b} & carboxyl
        return 1.5 if aromatic or pair == {"C_R", "O_2"} else 1.0

    def r0(a: int, b: int) -> float:
        return _bond_length(types[a], types[b], order(a, b))

    bonds, angles = Terms.of(distance), Terms.of(cos_angle)
    torsions, inversions = Terms.of(cos_torsion), Terms.of(cos_inversion)
    for a, nbrs in neighbours.items():
        for b in nbrs:
            if a < b:  # ½k (r - r₀)²
                k = 664.12 * UFF[types[a]][2] * UFF[types[b]][2] / r0(a, b) ** 3
                length = float(np.linalg.norm(vectors[a][b]))
                bonds.add((a, b), length, 0.0, 0.0, k * KCAL / 2)
        cos_uff = math.cos(math.radians(UFF[types[a]][1]))
        for x, y in [(x, y) for x in nbrs for y in nbrs if x < y]:
            rx, ry = r0(a, x), r0(a, y)
            rxy = math.sqrt(rx**2 + ry**2 - 2 * rx * ry * cos_uff)
            k = 664.12 * UFF[types[x]][2] * UFF[types[y]][2] / rxy**5 * KCAL
            k *= 3 * rx * ry * (1 - cos_uff**2) - rxy**2 * cos_uff
            # UFF's curvature K at the CIF's angle: K (cos θ - cos θ₀)² / 2sin²θ₀
            u, w = vectors[a][x], vectors[a][y]
            cos0 = float(u @ w / np.linalg.norm(u) / np.linalg.norm(w))
            angles.add((a, x, y), cos0, 0.0, 0.0, k / (2 * (1 - cos0**2)))
        if types[a] == "C_R":  # K (1 - cos ω) for each atom out of the others' plane
            k = INVERSION["O_2" if a in carboxyl else "default"]
            x, y, z = nbrs
            for p in [(x, y, z), (y, z, x), (z, x, y)]:
                inversions.add((a, *p), 0.0, k, -k)
    for b, c in [(b, c) for b in neighbours for c in neighbours[b] if b < c]:
        if {types[b], types[c]} & {"Zr8f4", "H_"}:
            continue  # no torsions around metal or terminal bonds
        u = math.sqrt(UFF[types[b]][3] * UFF[types[c]][3])
        v = 5 * u * (1 + 4.18 * math.log(order(b, c))) * KCAL
        v /= (len(neighbours[b]) - 1) * (len(neighbours[c]) - 1)
        for a in neighbours[b]:
            for d in neighbours[c]:
                if c != a != d != b:  # sp²-sp²: V/2 (1 - cos 2φ)
                    torsions.add((a, b, c, d), 0.0, v, 0.0, -v)
    for slot in range(MAX_CO2):
        c, o1, o2 = range(N_FRAMEWORK + 3 * slot, N_FRAMEWORK + 3 * slot + 3)
        bonds.add((c, o1), CO2_BOND, 0.0, 0.0, CO2_STRETCH / 2)
        bonds.add((c, o2), CO2_BOND, 0.0, 0.0, CO2_STRETCH / 2)
        angles.add((c, o1, o2), 0.0, CO2_BEND, CO2_BEND)  # K (1 + cos θ)
    return [bonds, angles, torsions, inversions]


def lj_parameters() -> GlobalTailCorrectedLennardJonesParameters:
    """GCMC's Lennard-Jones table, for the labels present. As the pair terms,
    the tail correction skips framework-framework pairs."""
    table = yaml.safe_load(open(EXAMPLES / "lennard_jones" / "trappe.yaml"))
    labels = sorted(set(LABELS))
    lj = GlobalTailCorrectedLennardJonesParameters.from_dict(
        cutoff=CUTOFF,
        parameters={label: table["parameters"][label] for label in labels},
        mixing_rule=table["mixing_rule"],
        tail_correction=table["tail_correction"],
    )
    guest = jnp.array([label in CO2_LABELS for label in lj.labels])
    return replace(lj, tail_corrected=lj.tail_corrected & (guest[:, None] | guest))


class _System(NamedTuple):
    cell: Cell[Any]


def ewald_parameters() -> EwaldParameters:
    """Ewald parameters estimated from the framework's charges and cell, as GCMC's."""
    particles, cell, _ = particles_from_ase(CIF)
    estimate = estimate_ewald_parameters(
        particles.data.charges,
        cell,
        real_cutoff=CUTOFF,
        epsilon_total=EWALD_PRECISION,
    )
    systems = Table((SystemId(0),), _System(cell[None]))
    alpha, cutoff, k_max = (
        systems.set_data(jnp.array([x]))
        for x in (estimate.alpha, estimate.real_cutoff, estimate.k_max)
    )
    return EwaldParameters.from_cutoffs(systems, alpha, cutoff, k_max)


def make_potential[S](
    state: Lens[S, MdState], charged: bool = True
) -> tuple[Potential[S, Any, Any, Any], Table[SystemId, jax.Array]]:
    """The force field on MD's padded state, and its cutoff."""
    labels = Index.new(list(map(Label, LABELS)))
    # Padding atoms sit apart, so pair terms within a padding slot stay finite.
    apart = jnp.outer(jnp.arange(len(LABELS)), jnp.ones(3))

    def typed(charges: np.ndarray, groups: np.ndarray) -> Lens[S, MdState]:
        """``state`` with GCMC's labels, ``charges`` (none on padding), and no
        nonbonded terms within each of ``groups``."""
        exclusion = Index.integer(
            jnp.asarray(groups),
            n=int(groups.max()) + 1,
            label=ExclusionId,
            max_count=int(np.bincount(groups).max()),
        )

        def view(s: MdState) -> MdState:
            p = s.particles.data
            real = p.system.valid_mask
            return bind(
                s,
                lambda s: (
                    s.particles.data.labels,
                    s.particles.data.charges,
                    s.particles.data.exclusion,
                    s.particles.data.positions,
                ),
            ).set(
                (
                    labels,
                    jnp.where(real, jnp.asarray(charges), 0.0),
                    exclusion,
                    jnp.where(real[:, None], p.positions, apart),
                )
            )

        return state.focus(view)

    # Lennard-Jones treats the framework as one molecule. Ewald's exclusions
    # are minimum-image pairs, which would jump as framework pairs cross half
    # the cell, so it subtracts the framework's own Ewald sum instead.
    molecules = np.concatenate([np.zeros(N_FRAMEWORK, int), 1 + SLOTS])
    atoms = np.concatenate([np.arange(N_FRAMEWORK), N_FRAMEWORK + SLOTS])
    framework = np.arange(len(LABELS)) < N_FRAMEWORK
    # Constants computed on the CPU are identical on every start, so the
    # programs embedding them hit the persistent compile cache.
    with jax.default_device(jax.devices("cpu")[0]):
        lj = lj_parameters()
        potentials = [
            make_lennard_jones_from_state(
                typed(CHARGES, molecules), parameters=lj, gradient=POSITIONS_AND_CELL
            ),
            make_lennard_jones_tail_correction_from_state(
                typed(CHARGES, molecules), parameters=lj, gradient=POSITIONS_AND_CELL
            ),
            *(terms.potential(state) for terms in bonded_terms()),
        ]
        if charged:
            ewald = ewald_parameters()
            potentials += [
                ScaledPotential(
                    make_ewald_from_state(
                        typed(charges, atoms),
                        parameters=ewald,
                        gradient=POSITIONS_AND_CELL,
                        include_exclusion_mask=True,
                    ),
                    scale,
                )
                for charges, scale in [(CHARGES, 1), (CHARGES * framework, -1)]
            ]
    return sum_potentials(*potentials), lj.cutoff
