# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests that _from_state factories work correctly.

For each classical potential the tests verify:
1. _from_state produces finite energy.
2. Energy changes when positions are perturbed.
3. Incremental energy matches full recomputation across multiple perturbation
   types (move, mask, edge swap).
"""

from typing import Any, Literal, NamedTuple

import jax.numpy as jnp
import pytest
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.data import WithCache, WithIndices
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import Lens, identity_lens, lens
from kups.core.neighborlist import AllDenseNearestNeighborList, Edges
from kups.core.patch import IndexLensPatch, Patch
from kups.core.potential import EMPTY, EmptyType, PotentialOut
from kups.core.typing import (
    ExclusionId,
    InclusionId,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass
from kups.potential.classical.cosine_angle import (
    CosineAngleParameters,
    make_cosine_angle_from_state,
)
from kups.potential.classical.dihedral import (
    DihedralParameters,
    make_dihedral_from_state,
)
from kups.potential.classical.harmonic import (
    HarmonicAngleParameters,
    HarmonicBondParameters,
    make_harmonic_angle_from_state,
    make_harmonic_bond_from_state,
)
from kups.potential.classical.inversion import (
    InversionParameters,
    make_inversion_from_state,
)
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    make_lennard_jones_from_state,
)
from kups.potential.classical.morse import (
    MorseBondParameters,
    make_morse_bond_from_state,
)
from kups.potential.common.graph import UpdatedEdges

# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------

N_SPECIES = 2
N_PARTICLES = 4
_LABELS = ("A", "B")
_PARTICLE_INDEX = tuple(ParticleId(i) for i in range(N_PARTICLES))

EmptyCache = PotentialOut[EmptyType, EmptyType]
_empty_cache = PotentialOut(Table.arange(jnp.zeros(1), label=SystemId), EMPTY, EMPTY)


@dataclass
class ParticleData:
    positions: Array
    labels: Index[str]
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]
    atomic_numbers: Array
    charges: Array


@dataclass
class SystemData:
    unitcell: UnitCell
    cutoff: Array


# ---------------------------------------------------------------------------
# State: WithCache-wrapped parameters for _from_state / _from_state_with_updates
# ---------------------------------------------------------------------------


@dataclass
class State:
    particles: Table[ParticleId, ParticleData]
    systems: Table[SystemId, SystemData]
    neighborlist: AllDenseNearestNeighborList
    lj_parameters: WithCache[LennardJonesParameters, EmptyCache]
    bond_edges: Edges[Literal[2]]
    harmonic_bond_parameters: WithCache[HarmonicBondParameters, EmptyCache]
    angle_edges: Edges[Literal[3]]
    harmonic_angle_parameters: WithCache[HarmonicAngleParameters, EmptyCache]
    morse_bond_parameters: WithCache[MorseBondParameters, EmptyCache]
    cosine_angle_edges: Edges[Literal[3]]
    cosine_angle_parameters: WithCache[CosineAngleParameters, EmptyCache]
    dihedral_edges: Edges[Literal[4]]
    dihedral_parameters: WithCache[DihedralParameters, EmptyCache]
    inversion_edges: Edges[Literal[4]]
    inversion_parameters: WithCache[InversionParameters, EmptyCache]


# ---------------------------------------------------------------------------
# Probe dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RadiusProbe:
    particles: WithIndices[ParticleId, ParticleData]
    neighborlist_after: AllDenseNearestNeighborList
    neighborlist_before: AllDenseNearestNeighborList


@dataclass
class EdgeSetProbe2:
    particles: WithIndices[ParticleId, ParticleData]
    edges: UpdatedEdges[Literal[2]]
    capacity: FixedCapacity[int]


@dataclass
class EdgeSetProbe3:
    particles: WithIndices[ParticleId, ParticleData]
    edges: UpdatedEdges[Literal[3]]
    capacity: FixedCapacity[int]


@dataclass
class EdgeSetProbe4:
    particles: WithIndices[ParticleId, ParticleData]
    edges: UpdatedEdges[Literal[4]]
    capacity: FixedCapacity[int]


# ---------------------------------------------------------------------------
# Particle / state builders
# ---------------------------------------------------------------------------

_INITIAL_POSITIONS = jnp.array(
    [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.5, 0.0], [1.5, 1.5, 1.5]]
)


def _make_particles(
    positions: Array,
    system_ids: Array,
) -> Table[ParticleId, ParticleData]:
    labels = Index.new(["A", "B", "A", "B"])
    system = Index.integer(system_ids, n=1, label=SystemId)
    inclusion = Index.integer(system_ids, n=1, label=InclusionId)
    exclusion = Index.integer(jnp.arange(N_PARTICLES), label=ExclusionId)
    data = ParticleData(
        positions=positions,
        labels=labels,
        system=system,
        inclusion=inclusion,
        exclusion=exclusion,
        atomic_numbers=jnp.array([6, 8, 6, 8], dtype=int),
        charges=jnp.array([0.1, -0.1, 0.1, -0.1]),
    )
    return Table.arange(data, label=ParticleId)


def _make_edges(pidx: tuple[ParticleId, ...], idx: Array, shifts: Array) -> Edges:
    return Edges(Index(pidx, idx), shifts)


class _CommonParts(NamedTuple):
    particles: Table[ParticleId, ParticleData]
    systems: Table[SystemId, SystemData]
    bond_edges: Edges[Literal[2]]
    angle_edges: Edges[Literal[3]]
    dihedral_edges: Edges[Literal[4]]
    inversion_edges: Edges[Literal[4]]
    lj: LennardJonesParameters
    hb: HarmonicBondParameters
    ha: HarmonicAngleParameters
    mb: MorseBondParameters
    ca: CosineAngleParameters
    dh: DihedralParameters
    inv: InversionParameters


def _build_common(positions: Array | None = None) -> _CommonParts:
    if positions is None:
        positions = _INITIAL_POSITIONS
    system_ids = jnp.zeros(N_PARTICLES, dtype=int)
    particles = _make_particles(positions, system_ids)
    unitcell = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10.0)
    systems = Table.arange(SystemData(unitcell, jnp.array([5.0])), label=SystemId)
    pidx = particles.keys
    ns = N_SPECIES
    return _CommonParts(
        particles=particles,
        systems=systems,
        bond_edges=_make_edges(
            pidx, jnp.array([[0, 1], [1, 2], [2, 3]]), jnp.zeros((3, 1, 3))
        ),
        angle_edges=_make_edges(
            pidx, jnp.array([[0, 1, 2], [1, 2, 3]]), jnp.zeros((2, 2, 3))
        ),
        dihedral_edges=_make_edges(
            pidx, jnp.array([[0, 1, 2, 3]]), jnp.zeros((1, 3, 3))
        ),
        inversion_edges=_make_edges(
            pidx, jnp.array([[1, 0, 2, 3]]), jnp.zeros((1, 3, 3))
        ),
        lj=LennardJonesParameters(
            labels=_LABELS,
            sigma=jnp.ones((ns, ns)),
            epsilon=jnp.ones((ns, ns)) * 0.5,
            cutoff=Table((SystemId(0),), jnp.array([5.0])),
        ),
        hb=HarmonicBondParameters(
            labels=_LABELS, x0=jnp.ones((ns, ns)) * 1.5, k=jnp.ones((ns, ns)) * 100.0
        ),
        ha=HarmonicAngleParameters(
            labels=_LABELS,
            theta0=jnp.ones((ns, ns, ns)) * 109.5,
            k=jnp.ones((ns, ns, ns)) * 50.0,
        ),
        mb=MorseBondParameters(
            labels=_LABELS,
            r0=jnp.ones((ns, ns)) * 1.5,
            D=jnp.ones((ns, ns)) * 2.0,
            alpha=jnp.ones((ns, ns)),
        ),
        ca=CosineAngleParameters(
            labels=_LABELS,
            theta0=jnp.ones((ns, ns, ns)) * jnp.radians(109.5),
            k=jnp.ones((ns, ns, ns)) * 50.0,
        ),
        dh=DihedralParameters(
            labels=_LABELS,
            V=jnp.ones((ns, ns, ns, ns)) * 2.0,
            n=jnp.ones((ns, ns, ns, ns), dtype=int) * 3,
            phi0=jnp.ones((ns, ns, ns, ns)) * jnp.pi,
        ),
        inv=InversionParameters(
            labels=_LABELS,
            omega0=jnp.zeros((ns, ns, ns, ns)),
            k=jnp.ones((ns, ns, ns, ns)) * 6.0,
        ),
    )


def _make_state(positions: Array | None = None) -> State:
    c = _build_common(positions)
    nl = AllDenseNearestNeighborList(
        avg_edges=FixedCapacity(N_PARTICLES**2),
        avg_image_candidates=FixedCapacity(N_PARTICLES**2),
    )
    ec = _empty_cache
    return State(
        particles=c.particles,
        systems=c.systems,
        neighborlist=nl,
        lj_parameters=WithCache(c.lj, ec),
        bond_edges=c.bond_edges,
        harmonic_bond_parameters=WithCache(c.hb, ec),
        angle_edges=c.angle_edges,
        harmonic_angle_parameters=WithCache(c.ha, ec),
        morse_bond_parameters=WithCache(c.mb, ec),
        cosine_angle_edges=c.angle_edges,
        cosine_angle_parameters=WithCache(c.ca, ec),
        dihedral_edges=c.dihedral_edges,
        dihedral_parameters=WithCache(c.dh, ec),
        inversion_edges=c.inversion_edges,
        inversion_parameters=WithCache(c.inv, ec),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def state() -> State:
    return _make_state()


@pytest.fixture(scope="module")
def state_lens() -> Lens[State, State]:
    return identity_lens(State)


# ---------------------------------------------------------------------------
# PerturbationSpec / PotentialConfig
# ---------------------------------------------------------------------------


class PerturbationSpec(NamedTuple):
    name: str
    indices: Array
    new_positions: Array
    new_system_ids: Array | None = None
    edge_updates: dict[str, tuple[Array, Any]] | None = None


class PotentialConfig(NamedTuple):
    name: str
    factory: Any
    degree: int | None
    edge_field: str | None


_POTENTIALS: list[PotentialConfig] = [
    PotentialConfig("lennard_jones", make_lennard_jones_from_state, None, None),
    PotentialConfig("harmonic_bond", make_harmonic_bond_from_state, 2, "bond_edges"),
    PotentialConfig("harmonic_angle", make_harmonic_angle_from_state, 3, "angle_edges"),
    PotentialConfig("morse_bond", make_morse_bond_from_state, 2, "bond_edges"),
    PotentialConfig(
        "cosine_angle", make_cosine_angle_from_state, 3, "cosine_angle_edges"
    ),
    PotentialConfig("dihedral", make_dihedral_from_state, 4, "dihedral_edges"),
    PotentialConfig("inversion", make_inversion_from_state, 4, "inversion_edges"),
]


# ---------------------------------------------------------------------------
# Perturbation factories
# ---------------------------------------------------------------------------


def _move_single() -> PerturbationSpec:
    new_pos = _INITIAL_POSITIONS.at[0].add(jnp.array([0.3, 0.1, 0.2]))
    return PerturbationSpec("move_single", jnp.array([0]), new_pos)


def _move_multiple() -> PerturbationSpec:
    new_pos = _INITIAL_POSITIONS.at[0].add(jnp.array([0.3, 0.1, 0.2]))
    new_pos = new_pos.at[2].add(jnp.array([-0.1, 0.2, 0.15]))
    return PerturbationSpec("move_multiple", jnp.array([0, 2]), new_pos)


def _mask_particle() -> PerturbationSpec:
    return PerturbationSpec(
        "mask_particle",
        jnp.array([3]),
        _INITIAL_POSITIONS,
        new_system_ids=jnp.array([0, 0, 0, 1]),
    )


def _edge_swap() -> PerturbationSpec:
    pidx = _PARTICLE_INDEX
    return PerturbationSpec(
        "edge_swap",
        jnp.array([], dtype=int),
        _INITIAL_POSITIONS,
        edge_updates={
            "bond_edges": (
                jnp.array([2]),
                _make_edges(pidx, jnp.array([[0, 3]]), jnp.zeros((1, 1, 3))),
            ),
            "angle_edges": (
                jnp.array([1]),
                _make_edges(pidx, jnp.array([[0, 2, 3]]), jnp.zeros((1, 2, 3))),
            ),
            "cosine_angle_edges": (
                jnp.array([1]),
                _make_edges(pidx, jnp.array([[0, 2, 3]]), jnp.zeros((1, 2, 3))),
            ),
            "dihedral_edges": (
                jnp.array([0]),
                _make_edges(pidx, jnp.array([[3, 1, 2, 0]]), jnp.zeros((1, 3, 3))),
            ),
            "inversion_edges": (
                jnp.array([0]),
                _make_edges(pidx, jnp.array([[0, 1, 2, 3]]), jnp.zeros((1, 3, 3))),
            ),
        },
    )


_PERTURBATIONS = [_move_single(), _move_multiple(), _mask_particle(), _edge_swap()]


# ---------------------------------------------------------------------------
# Edge lenses
# ---------------------------------------------------------------------------

_EDGE_LENSES: dict[str, Lens[State, Any]] = {
    "bond_edges": lens(lambda s: s.bond_edges, cls=State),
    "angle_edges": lens(lambda s: s.angle_edges, cls=State),
    "cosine_angle_edges": lens(lambda s: s.cosine_angle_edges, cls=State),
    "dihedral_edges": lens(lambda s: s.dihedral_edges, cls=State),
    "inversion_edges": lens(lambda s: s.inversion_edges, cls=State),
}

_PARTICLES_LENS = lens(lambda s: s.particles, cls=State)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_point_changes(
    state: State,
    new_positions: Array,
    indices: Array,
    new_system_ids: Array | None = None,
) -> WithIndices[ParticleId, ParticleData]:
    import jax

    pdata = state.particles.data
    sys_ids = (
        new_system_ids[indices]
        if new_system_ids is not None
        else jnp.zeros(indices.size, dtype=int)
    )
    # Slice particle data preserving Index label metadata
    sliced = jax.tree.map(lambda x: x[indices], pdata)
    changed_data = ParticleData(
        positions=new_positions[indices],
        labels=sliced.labels,
        system=Index(pdata.system.keys, sys_ids, _cls=pdata.system.cls),
        inclusion=Index(pdata.inclusion.keys, sys_ids, _cls=pdata.inclusion.cls),
        exclusion=Index(pdata.exclusion.keys, indices, _cls=pdata.exclusion.cls),
        atomic_numbers=sliced.atomic_numbers,
        charges=sliced.charges,
    )
    pidx = Index(
        tuple(ParticleId(i) for i in range(N_PARTICLES)), indices, _cls=ParticleId
    )
    return WithIndices(pidx, changed_data)


def _make_patch(spec: PerturbationSpec) -> IndexLensPatch[State, Any]:
    system_ids = (
        spec.new_system_ids
        if spec.new_system_ids is not None
        else jnp.zeros(N_PARTICLES, dtype=int)
    )
    particles = _make_particles(spec.new_positions, system_ids)
    mask_idx = Index((SystemId(0),), system_ids)
    return IndexLensPatch(data=particles, mask_idx=mask_idx, lens=_PARTICLES_LENS)


def _build_probe(pot: PotentialConfig, spec: PerturbationSpec) -> Any:
    def _point_changes(
        state: State, patch: Patch[State]
    ) -> WithIndices[ParticleId, ParticleData]:
        new_state: State = patch(
            state, state.systems.set_data(jnp.ones(len(state.systems), dtype=bool))
        )
        new_positions = new_state.particles.data.positions
        new_system_ids = (
            new_state.particles.data.system.indices
            if spec.new_system_ids is not None
            else None
        )
        return _make_point_changes(state, new_positions, spec.indices, new_system_ids)

    def _edge_updates(degree: int) -> UpdatedEdges[Any]:
        pidx = _PARTICLE_INDEX
        if (
            spec.edge_updates is not None
            and pot.edge_field is not None
            and pot.edge_field in spec.edge_updates
        ):
            slot_idx, new_edges = spec.edge_updates[pot.edge_field]
            return UpdatedEdges(slot_idx, new_edges)
        return UpdatedEdges(
            jnp.array([], dtype=int),
            Edges(
                Index(pidx, jnp.zeros((0, degree), dtype=int)),
                jnp.zeros((0, degree - 1, 3)),
            ),
        )

    if pot.degree is None:

        def radius_probe(state: State, patch: Patch[State]) -> RadiusProbe:
            return RadiusProbe(
                particles=_point_changes(state, patch),
                neighborlist_after=state.neighborlist,
                neighborlist_before=state.neighborlist,
            )

        return radius_probe

    if pot.degree == 2:

        def edge_probe2(state: State, patch: Patch[State]) -> EdgeSetProbe2:
            return EdgeSetProbe2(
                particles=_point_changes(state, patch),
                edges=_edge_updates(2),
                capacity=FixedCapacity(N_PARTICLES**2),
            )

        return edge_probe2

    if pot.degree == 3:

        def edge_probe3(state: State, patch: Patch[State]) -> EdgeSetProbe3:
            return EdgeSetProbe3(
                particles=_point_changes(state, patch),
                edges=_edge_updates(3),
                capacity=FixedCapacity(N_PARTICLES**2),
            )

        return edge_probe3

    if pot.degree == 4:

        def edge_probe4(state: State, patch: Patch[State]) -> EdgeSetProbe4:
            return EdgeSetProbe4(
                particles=_point_changes(state, patch),
                edges=_edge_updates(4),
                capacity=FixedCapacity(N_PARTICLES**2),
            )

        return edge_probe4

    raise ValueError(f"Unsupported degree: {pot.degree}")


def _make_reference_state(
    state: State,
    spec: PerturbationSpec,
    patch: IndexLensPatch[State, Any],
) -> State:
    """Apply patch + edge updates to produce a reference State for full eval."""
    ref: State = patch(
        state, state.systems.set_data(jnp.ones(len(state.systems), dtype=bool))
    )
    if spec.edge_updates:
        for field_name, (slot_idx, new_edges) in spec.edge_updates.items():
            el = _EDGE_LENSES[field_name]
            old = el.get(ref)
            ref = el.set(ref, old.at(slot_idx).set(new_edges))
    return ref


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFromState:
    """Tests for _from_state factories (full evaluation, no updates)."""

    @pytest.mark.parametrize("pot", _POTENTIALS, ids=[p.name for p in _POTENTIALS])
    def test_finite_energy_and_changes_with_position(
        self,
        state: State,
        state_lens: Lens[State, State],
        pot: PotentialConfig,
    ):
        name = pot.name
        potential = pot.factory(state_lens)

        # 1. Finite energy check
        out = potential(state)
        e1 = out.data.total_energies.data
        assert e1.shape == (1,)
        assert jnp.isfinite(e1).all(), f"{name}: energy is not finite: {e1}"

        # 2. Energy changes with perturbed positions
        perturbed_positions = state.particles.data.positions + jnp.array(
            [[0.3, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.1, -0.3, 0.0]]
        )
        d = state.particles.data
        new_data = ParticleData(
            positions=perturbed_positions,
            labels=d.labels,
            system=d.system,
            inclusion=d.inclusion,
            exclusion=d.exclusion,
            atomic_numbers=d.atomic_numbers,
            charges=d.charges,
        )
        new_particles = Table(state.particles.keys, new_data)
        sl = identity_lens(State)
        state2 = sl.focus(lambda s: s.particles).set(state, new_particles)
        e2 = potential(state2).data.total_energies.data
        assert not jnp.allclose(e1, e2), (
            f"{name}: energy did not change after perturbation"
        )


class TestFromStateWithUpdates:
    """Tests for _from_state_with_updates: incremental energy matches full recomputation."""

    @pytest.mark.parametrize(
        "pot",
        _POTENTIALS,
        ids=[p.name for p in _POTENTIALS],
    )
    def test_incremental_update(
        self,
        state: State,
        state_lens: Lens[State, State],
        pot: PotentialConfig,
    ):
        # Build the reference potential (no probe) for full recomputation.
        basic_pot = pot.factory(state_lens)

        perturbations = [
            spec
            for spec in _PERTURBATIONS
            if not (spec.edge_updates is not None and pot.degree is None)
        ]

        for spec in perturbations:
            probe = _build_probe(pot, spec)
            updates_pot = pot.factory(state_lens, probe)

            # 1. Full eval to populate cache
            out_full = updates_pot(state, patch=None)
            state_cached: State = out_full.patch(
                state,
                state.systems.set_data(jnp.ones(len(state.systems), dtype=bool)),
            )

            # 2. Incremental evaluation with patch
            pos_patch = _make_patch(spec)
            out_incr = updates_pot(state_cached, patch=pos_patch)
            e_incr = out_incr.data.total_energies.data

            # 3. Full recomputation on reference state
            ref_state = _make_reference_state(state_cached, spec, pos_patch)
            e_full = basic_pot(ref_state).data.total_energies.data

            assert jnp.allclose(e_incr, e_full, atol=1e-6), (
                f"{pot.name}-{spec.name}: incremental {e_incr} != full {e_full}"
            )
