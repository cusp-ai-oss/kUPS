# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LocalMLIAP potential from kups.potential.mliap.local."""

from __future__ import annotations

import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import bind, lens
from kups.core.neighborlist import AllDenseNearestNeighborList
from kups.core.patch import Accept, Patch
from kups.core.potential import PotentialOut
from kups.core.result import as_result_function
from kups.core.typing import ExclusionId, InclusionId, ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass
from kups.potential.mliap.local import (
    LocalMLIAPCache,
    LocalMLIAPComposer,
    LocalMLIAPData,
    make_local_mliap_potential,
)

from ..clear_cache import clear_cache  # noqa: F401


# Test data structures
@dataclass
class AtomData:
    positions: Array
    atomic_numbers: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class SystemData:
    unitcell: UnitCell
    cutoff: Array


# Simple model functions for testing - smaller dimensions for speed
def make_simple_model(embed_dim: int = 4):
    """Create simple model functions for testing."""
    n_species = 3
    embedding = jnp.ones((n_species, embed_dim)) * 0.1
    readout_w = jnp.ones((embed_dim, 1)) * 0.1

    def init_function(atomic_numbers: Array) -> Array:
        return embedding[atomic_numbers]

    def edge_function(node1: Array, node2: Array, difference_vectors: Array) -> Array:
        dist = jnp.linalg.norm(difference_vectors, axis=-1, keepdims=True)
        return node2 / (dist + 1.0)

    def readout_function(node_emb: Array, msg_sum: Array) -> Array:
        return ((node_emb + msg_sum) @ readout_w).squeeze(-1)

    return init_function, edge_function, readout_function


@dataclass
class State:
    atoms: Table[ParticleId, AtomData]
    systems: Table[SystemId, SystemData]
    model_config: LocalMLIAPData
    out: PotentialOut[tuple[()], tuple[()]]
    full_neighborlist: AllDenseNearestNeighborList
    update_nnlist: AllDenseNearestNeighborList


@dataclass
class AtomPatch(Patch[State]):
    indices: Array
    new_positions: Array

    def __call__(self, state: State, accept: Accept) -> State:
        prev_positions = state.atoms.data.positions[self.indices]
        mask = accept[Index.new(list(accept.keys))]
        new_pos = jnp.where(mask, self.new_positions, prev_positions)
        return (
            bind(state)
            .focus(lambda s: s.atoms.data.positions)
            .at(self.indices)
            .set(new_pos)
        )


# Use module scope to avoid re-creating fixtures
@pytest.fixture(scope="module")
def simple_system():
    """Create a simple test system - smaller for speed."""
    n_atoms = 8
    embed_dim = 4

    # Spread atoms far apart to minimize edges
    positions = jnp.array([[i * 3.0, 0.0, 0.0] for i in range(n_atoms)], dtype=float)
    atomic_numbers = jnp.zeros((n_atoms,), dtype=int)
    unitcell = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 30.0)

    init_fn, edge_fn, readout_fn = make_simple_model(embed_dim)

    system_idx = Index.new([SystemId(0)] * n_atoms)
    inclusion_idx = Index.new([InclusionId(0)] * n_atoms)
    exclusion_idx = Index.new([ExclusionId(0)] * n_atoms)
    atoms = Table.arange(
        AtomData(
            positions=positions,
            atomic_numbers=atomic_numbers,
            system=system_idx,
            inclusion=inclusion_idx,
            exclusion=exclusion_idx,
        ),
        label=ParticleId,
    )
    systems = Table.arange(
        SystemData(unitcell=unitcell, cutoff=jnp.array([2.5])),
        label=SystemId,
    )

    state = State(
        atoms=atoms,
        systems=systems,
        model_config=LocalMLIAPData(
            cutoff=Table((SystemId(0),), jnp.array([2.5])),
            init_function=init_fn,
            edge_function=edge_fn,
            readout_function=readout_fn,
            cache=LocalMLIAPCache(
                node_init=jnp.zeros((n_atoms, embed_dim)),
                msg_sum=jnp.zeros((n_atoms, embed_dim)),
            ),
        ),
        out=PotentialOut(Table.arange(jnp.zeros((1,)), label=SystemId), (), ()),
        full_neighborlist=AllDenseNearestNeighborList(
            avg_edges=FixedCapacity(n_atoms),
            avg_image_candidates=FixedCapacity(n_atoms),
        ),
        update_nnlist=AllDenseNearestNeighborList(
            avg_edges=FixedCapacity(n_atoms),
            avg_image_candidates=FixedCapacity(n_atoms),
        ),
    )

    def make_patch(state: State) -> AtomPatch:
        return AtomPatch(
            indices=jnp.arange(2),
            new_positions=jnp.array([[0.1, 0.1, 0.1], [3.1, 0.0, 0.0]]),
        )

    return state, make_patch


@dataclass
class ProbeParticles:
    """Satisfies WithIndices[ParticleId, AtomData]."""

    indices: Index[ParticleId]
    data: AtomData


@dataclass
class SimpleRadiusProbe:
    particles: ProbeParticles
    neighborlist_after: AllDenseNearestNeighborList
    neighborlist_before: AllDenseNearestNeighborList


def _make_probe(state: State, patch: AtomPatch) -> SimpleRadiusProbe:
    change_idx = Index(state.atoms.keys, patch.indices)
    n_change = len(patch.indices)
    system_idx = Index.new([SystemId(0)] * n_change)
    inclusion_idx = Index.new([InclusionId(0)] * n_change)
    exclusion_idx = Index.new([ExclusionId(0)] * n_change)
    particles = ProbeParticles(
        indices=change_idx,
        data=AtomData(
            positions=patch.new_positions,
            atomic_numbers=state.atoms.data.atomic_numbers[patch.indices],
            system=system_idx,
            inclusion=inclusion_idx,
            exclusion=exclusion_idx,
        ),
    )
    return SimpleRadiusProbe(
        particles=particles,
        neighborlist_after=state.update_nnlist,
        neighborlist_before=state.update_nnlist,
    )


@pytest.fixture(scope="module")
def composer():
    """Create a LocalMLIAPComposer for testing."""
    return LocalMLIAPComposer[State, AtomData, SystemData, AtomPatch](
        particles=lambda s: s.atoms,
        systems=lambda s: s.systems,
        cutoffs=lambda s: s.systems.map_data(lambda d: d.cutoff),
        neighborlist=lambda s: s.full_neighborlist,
        model=lens(lambda s: s.model_config, cls=State),
        probe=_make_probe,
    )


@pytest.fixture(scope="module")
def potential():
    """Create a potential using the factory function."""
    return make_local_mliap_potential(
        particles_view=lambda s: s.atoms,
        systems_view=lambda s: s.systems,
        cutoffs_view=lambda s: s.systems.map_data(lambda d: d.cutoff),
        neighborlist_view=lambda s: s.full_neighborlist,
        model_lens=lens(lambda s: s.model_config, cls=State),
        probe=_make_probe,
        gradient_lens=lens(lambda x: ()),
        hessian_lens=lens(lambda x: ()),
        hessian_idx_view=lambda x: (),
        out_cache_lens=lens(lambda x: x.out, cls=State),
        patch_idx_view=lambda x: PotentialOut(
            Table((SystemId(0),), Index.new((SystemId(0),)), _cls=SystemId), (), ()
        ),
    )


class TestLocalMLIAPEnergy:
    """Tests for local_mliap_energy functions."""

    def test_full_energy_computation(self, simple_system, potential):
        """Test full energy computation produces valid, non-zero output."""
        state, _ = simple_system
        potential_fn = as_result_function(potential)

        state = potential_fn(state).fix_or_raise(state)
        result = potential_fn(state)
        result.raise_assertion()

        energy = result.value.data.total_energies.data
        assert energy.shape == (1,)
        assert jnp.isfinite(energy).all()
        # With non-trivial model functions and atoms at non-zero positions,
        # energy should be non-zero
        assert not jnp.allclose(energy, 0.0)

    def test_incremental_vs_full(self, simple_system, potential):
        """Test that incremental update matches full recomputation."""
        state, make_patch = simple_system
        potential_fn = as_result_function(potential)

        # Initialize
        state = potential_fn(state).fix_or_raise(state)
        out = potential_fn(state)
        out.raise_assertion()
        state = out.value.patch(
            state, state.systems.set_data(jnp.ones(len(state.systems), dtype=jnp.bool_))
        )

        # Incremental with patch
        patch = make_patch(state)
        state = potential_fn(state, patch).fix_or_raise(state)
        out_incremental = potential_fn(state, patch)
        out_incremental.raise_assertion()

        # Full recomputation
        new_state = patch(
            state, state.systems.set_data(jnp.ones(len(state.systems), dtype=jnp.bool_))
        )
        out_full = potential_fn(new_state)
        out_full.raise_assertion()

        npt.assert_allclose(
            out_incremental.value.data.total_energies.data,
            out_full.value.data.total_energies.data,
            rtol=1e-5,
        )


class TestLocalMLIAPCache:
    """Tests for LocalMLIAPCache."""

    def test_cache_update(self, simple_system, potential):
        """Test that cache is updated after energy computation."""
        state, _ = simple_system
        potential_fn = as_result_function(potential)

        assert jnp.allclose(state.model_config.cache.node_init, 0.0)

        state = potential_fn(state).fix_or_raise(state)
        result = potential_fn(state)
        result.raise_assertion()
        state = result.value.patch(
            state, state.systems.set_data(jnp.ones(len(state.systems), dtype=jnp.bool_))
        )

        assert not jnp.allclose(state.model_config.cache.node_init, 0.0)


class TestLocalMLIAPComposer:
    """Tests for LocalMLIAPComposer."""

    def test_full_composition(self, simple_system, composer):
        """Test composition without patch returns single summand."""
        state, _ = simple_system
        result = composer(state, None)

        assert len(result) == 1
        assert result[0].weight == 1.0
        assert result[0].inp.point_cloud_changes is None

    def test_patch_composition(self, simple_system, composer):
        """Test composition with patch includes changes."""
        state, make_patch = simple_system
        result = composer(state, make_patch(state))

        assert len(result) == 1
        assert result[0].inp.point_cloud_changes is not None


class TestModelFunctions:
    """Tests for model function protocols."""

    def test_function_shapes(self):
        """Test model functions produce correct output shapes."""
        init_fn, edge_fn, readout_fn = make_simple_model(embed_dim=8)

        # Init function
        species = jnp.array([0, 1, 2])
        assert init_fn(species).shape == (3, 8)

        # Edge function
        node1 = jnp.ones((5, 8))
        node2 = jnp.ones((5, 8))
        diff_vecs = jnp.ones((5, 3))
        assert edge_fn(node1, node2, diff_vecs).shape == (5, 8)

        # Readout function
        node_emb = jnp.ones((10, 8))
        msg_sum = jnp.ones((10, 8))
        assert readout_fn(node_emb, msg_sum).shape == (10,)
