# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.capacity import CapacityError, LensCapacity
from kups.core.data import WithIndices
from kups.core.data.buffered import Buffered, system_view
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import bind, lens
from kups.core.result import as_result_function
from kups.core.typing import GroupId, MotifId, ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass, key_chain
from kups.core.utils.position import center_of_mass, to_relative_positions
from kups.mcmc.moves import (
    ExchangeChanges,
    ExchangeMove,
    delete_random_motif,
    insert_random_motif,
    random_rotate_groups,
    random_select_groups,
    translate_groups,
)

from ..clear_cache import clear_cache  # noqa: F401


def _make_index(token, ids, n, max_count=None):
    return Index.integer(ids, n=n, label=token, max_count=max_count)


@dataclass
class Particles:
    positions: jax.Array


@dataclass
class ParticlesWithMass:
    positions: jax.Array
    weights: jax.Array


@dataclass
class GroupSystemData:
    """Particle data with system and group indices."""

    positions: jax.Array
    system: Index[SystemId]
    group: Index[GroupId]


@dataclass
class GroupSystemDataWithMass:
    """Particle data with system and group indices and mass."""

    positions: jax.Array
    weights: jax.Array
    system: Index[SystemId]
    group: Index[GroupId]


@dataclass
class SystemData:
    """Data with only system index."""

    system: Index[SystemId]


@dataclass
class PositionSystemData:
    """Particle data with positions and system index."""

    positions: jax.Array
    system: Index[SystemId]


@dataclass
class UnitCellData:
    """System data with unit cell."""

    unitcell: UnitCell


@dataclass
class GroupData:
    """Group data with labels and system index."""

    labels: Index[str]
    system: Index[SystemId]


class TestRandomSelectGroups:
    @classmethod
    def setup_class(cls):
        cls.n_groups_per_sys = 10
        cls.max_group_size = 3
        cls.n_sys = 5
        cls.group_system_ids = Table.arange(
            SystemData(
                system=_make_index(
                    SystemId,
                    jnp.arange(cls.n_sys, dtype=int).repeat(cls.n_groups_per_sys),
                    cls.n_sys,
                    max_count=cls.n_groups_per_sys,
                ),
            ),
            label=GroupId,
        )
        cls.particle_data = Table.arange(
            GroupSystemData(
                positions=jnp.zeros(
                    (cls.n_sys * cls.n_groups_per_sys * cls.max_group_size, 3)
                ),
                system=_make_index(
                    SystemId,
                    jnp.arange(cls.n_sys, dtype=int).repeat(
                        cls.max_group_size * cls.n_groups_per_sys
                    ),
                    cls.n_sys,
                    max_count=cls.n_groups_per_sys * cls.max_group_size,
                ),
                group=_make_index(
                    GroupId,
                    jnp.arange(cls.n_groups_per_sys * cls.n_sys, dtype=int).repeat(
                        cls.max_group_size
                    ),
                    cls.n_groups_per_sys * cls.n_sys,
                    max_count=cls.max_group_size,
                ),
            ),
            label=ParticleId,
        )
        cls.capacity = LensCapacity(15, lens(lambda x: x))

    def test_basic(self):
        key = jax.random.key(0)
        chain = key_chain(key)
        index_result = as_result_function(random_select_groups)(
            next(chain),
            self.group_system_ids,
            self.particle_data,
            capacity=self.capacity,
        )
        index_result.raise_assertion()
        indices = index_result.value.indices
        selected_systems = self.particle_data.data.system.indices[indices]
        target = (
            jnp.arange(self.n_sys)[:, None]
            .repeat(self.max_group_size, axis=-1)
            .reshape(-1)
        )
        npt.assert_allclose(selected_systems, target)

    def test_permuted(self):
        key = jax.random.key(1)
        chain = key_chain(key)
        group_permutation = jax.random.permutation(
            next(chain), jnp.arange(self.n_groups_per_sys * self.n_sys)
        )
        group_system_ids = (
            bind(self.group_system_ids)
            .focus(lambda x: x.data.system.indices)
            .set(self.group_system_ids.data.system.indices[group_permutation])
        )
        permutation = jax.random.permutation(
            next(chain),
            jnp.arange(self.n_groups_per_sys * self.max_group_size * self.n_sys),
        )
        inv_group_permutation = jnp.argsort(group_permutation)
        particle_data = (
            bind(self.particle_data)
            .focus(lambda x: x.data.group.indices)
            .set(
                inv_group_permutation[self.particle_data.data.group.indices][
                    permutation
                ]
            )
        )
        particle_data = (
            bind(particle_data)
            .focus(lambda x: x.data.system.indices)
            .set(particle_data.data.system.indices[permutation])
        )
        index_result = as_result_function(random_select_groups)(
            next(chain), group_system_ids, particle_data, capacity=self.capacity
        )
        index_result.raise_assertion()
        indices = index_result.value.indices
        selected_systems = particle_data.data.system.indices[indices]
        particles_per_system = jax.ops.segment_sum(
            jnp.ones_like(selected_systems), selected_systems, self.n_sys
        )
        npt.assert_allclose(
            particles_per_system,
            jnp.full((self.n_sys,), fill_value=self.max_group_size),
        )

    def test_insufficient_capacity(self):
        key = jax.random.key(0)
        chain = key_chain(key)
        index_result = as_result_function(random_select_groups)(
            next(chain),
            self.group_system_ids,
            self.particle_data,
            capacity=LensCapacity(1, lens(lambda x: x)),
        )
        with pytest.raises(CapacityError):
            index_result.raise_assertion()


class TestRandomRotateGroups:
    @staticmethod
    def _make_particles(positions, system_ids, n_sys):
        """Build Table particles from raw arrays."""
        return Table.arange(
            GroupSystemData(
                positions,
                system=_make_index(SystemId, system_ids, n_sys),
                group=Index(tuple(GroupId(i) for i in range(n_sys)), system_ids),
            ),
            label=ParticleId,
        )

    @staticmethod
    def _make_systems(unitcell):
        """Build Table systems from a UnitCell."""
        return Table.arange(UnitCellData(unitcell=unitcell), label=SystemId)

    @classmethod
    def setup_class(cls):
        cls.n_sys = 3
        cls.n_particles_per_sys = 6

        lattice_vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cls.unitcell = TriclinicUnitCell.from_matrix(
            jnp.broadcast_to(lattice_vecs, (cls.n_sys, 3, 3))
        )
        cls.jit_rotate = staticmethod(jax.jit(random_rotate_groups))

        cls.positions = jnp.array(
            [
                [0.5, 0.5, 0.5],
                [0.51, 0.5, 0.5],
                [0.52, 0.5, 0.5],
                [0.53, 0.5, 0.5],
                [0.54, 0.5, 0.5],
                [0.55, 0.5, 0.5],
                [0.5, 0.6, 0.5],
                [0.51, 0.6, 0.5],
                [0.52, 0.6, 0.5],
                [0.53, 0.6, 0.5],
                [0.54, 0.6, 0.5],
                [0.55, 0.6, 0.5],
                [0.5, 0.5, 0.6],
                [0.51, 0.5, 0.6],
                [0.52, 0.5, 0.6],
                [0.53, 0.5, 0.6],
                [0.54, 0.5, 0.6],
                [0.55, 0.5, 0.6],
            ]
        )
        cls.system_ids = jnp.repeat(jnp.arange(cls.n_sys), cls.n_particles_per_sys)
        cls.step_width = jnp.array([0.1, 0.2, 0.3])
        cls.group_index = Index(
            tuple(GroupId(i) for i in range(cls.n_sys)),
            cls.system_ids,
        )
        cls.particles = cls._make_particles(cls.positions, cls.system_ids, cls.n_sys)
        cls.systems = cls._make_systems(cls.unitcell)

    def _make_mass_particles(self, positions):
        """Build GroupSystemDataWithMass Table from positions."""
        return Table.arange(
            GroupSystemDataWithMass(
                positions,
                jnp.ones(positions.shape[0]),
                system=_make_index(SystemId, self.system_ids, self.n_sys),
                group=self.group_index,
            ),
            label=ParticleId,
        )

    def test_rotation_properties(self):
        """Merged: basic, zero-step-width identity, COM preservation, distance
        preservation, different step widths, determinism, different keys,
        large step width, and non-cubic unit cell."""
        # Basic rotation: correct shape, finite, actually changed
        key = jax.random.key(0)
        rotated = self.jit_rotate(key, self.particles, self.systems, self.step_width)
        assert rotated.shape == self.positions.shape
        assert not jnp.allclose(rotated, self.positions)
        assert jnp.all(jnp.isfinite(rotated))

        # Zero step width preserves relative positions (identity rotation)
        zero_sw = jnp.zeros(self.n_sys)
        particles_m = self._make_mass_particles(self.positions)
        orig_com = center_of_mass(particles_m, self.unitcell)
        orig_rel = to_relative_positions(particles_m, self.unitcell, orig_com)
        rot_zero = self.jit_rotate(
            jax.random.key(1), self.particles, self.systems, zero_sw
        )
        rot_m = self._make_mass_particles(rot_zero)
        rot_com = center_of_mass(rot_m, self.unitcell)
        rot_rel = to_relative_positions(rot_m, self.unitcell, rot_com)
        npt.assert_allclose(orig_rel, rot_rel, atol=1e-10)

        # COM preservation with non-zero step width
        com_before = center_of_mass(self.particles, self.unitcell)
        rot2 = self.jit_rotate(
            jax.random.key(2), self.particles, self.systems, self.step_width
        )
        com_after = center_of_mass(
            self._make_particles(rot2, self.system_ids, self.n_sys), self.unitcell
        )
        npt.assert_allclose(com_after, com_before)

        # Distance from center preserved
        rot3 = self.jit_rotate(
            jax.random.key(3), self.particles, self.systems, self.step_width
        )
        rot3_m = self._make_mass_particles(rot3)
        rot3_com = center_of_mass(rot3_m, self.unitcell)
        rot3_rel = to_relative_positions(rot3_m, self.unitcell, rot3_com)
        npt.assert_allclose(
            jnp.linalg.norm(orig_rel, axis=1),
            jnp.linalg.norm(rot3_rel, axis=1),
            atol=1e-10,
        )

        # Different step widths produce different results
        rot_small = self.jit_rotate(
            jax.random.key(4),
            self.particles,
            self.systems,
            jnp.array([0.01, 0.01, 0.01]),
        )
        rot_large = self.jit_rotate(
            jax.random.key(4),
            self.particles,
            self.systems,
            jnp.array([1.0, 1.0, 1.0]),
        )
        assert not jnp.allclose(rot_small, rot_large)

        # Deterministic with same key
        k5 = jax.random.key(5)
        r1 = self.jit_rotate(k5, self.particles, self.systems, self.step_width)
        r2 = self.jit_rotate(k5, self.particles, self.systems, self.step_width)
        npt.assert_allclose(r1, r2, atol=1e-15)

        # Different keys produce different results
        r_a = self.jit_rotate(
            jax.random.key(6), self.particles, self.systems, self.step_width
        )
        r_b = self.jit_rotate(
            jax.random.key(7), self.particles, self.systems, self.step_width
        )
        assert not jnp.allclose(r_a, r_b)

        # Large step width: finite output
        rot_lg = self.jit_rotate(
            jax.random.key(9),
            self.particles,
            self.systems,
            jnp.array([2.0, 3.0, 4.0]),
        )
        assert rot_lg.shape == self.positions.shape
        assert jnp.all(jnp.isfinite(rot_lg))

        # Non-cubic (hexagonal) unit cell
        lattice_vecs = jnp.array(
            [[1.0, 0.0, 0.0], [0.5, jnp.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]]
        )
        hex_uc = TriclinicUnitCell.from_matrix(
            jnp.broadcast_to(lattice_vecs, (self.n_sys, 3, 3))
        )
        hex_sys = self._make_systems(hex_uc)
        rot_hex = self.jit_rotate(
            jax.random.key(10), self.particles, hex_sys, self.step_width
        )
        assert rot_hex.shape == self.positions.shape
        assert jnp.all(jnp.isfinite(rot_hex))
        assert not jnp.allclose(rot_hex, self.positions)

    def test_single_system(self):
        key = jax.random.key(8)
        n_particles = 4
        single_unitcell = TriclinicUnitCell.from_matrix(
            jnp.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
        )
        single_positions = jnp.array(
            [[0.5, 0.5, 0.5], [0.51, 0.5, 0.5], [0.5, 0.51, 0.5], [0.5, 0.5, 0.51]]
        )
        single_system_ids = jnp.zeros(n_particles, dtype=int)
        single_step_width = jnp.array([0.5])
        single_particles = self._make_particles(single_positions, single_system_ids, 1)
        single_systems = self._make_systems(single_unitcell)
        rotated = self.jit_rotate(
            key, single_particles, single_systems, single_step_width
        )
        assert rotated.shape == single_positions.shape
        assert jnp.all(jnp.isfinite(rotated))
        assert not jnp.allclose(rotated, single_positions)


class TestTranslateGroups:
    @staticmethod
    def _call(jit_fn, translations, positions, unitcell, system_ids):
        """Wrap raw arrays into Table types and call translate_groups."""
        n_sys = translations.shape[0]
        trans = Table.arange(translations, label=SystemId)
        particles = Table.arange(
            PositionSystemData(
                positions=positions,
                system=_make_index(SystemId, system_ids, n_sys),
            ),
            label=ParticleId,
        )
        systems = Table.arange(UnitCellData(unitcell=unitcell), label=SystemId)
        return jit_fn(trans, particles, systems)

    @classmethod
    def setup_class(cls):
        cls.n_sys = 3
        cls.n_particles_per_sys = 4
        lattice_vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cls.unitcell = TriclinicUnitCell.from_matrix(
            jnp.broadcast_to(lattice_vecs, (cls.n_sys, 3, 3))
        )
        cls.jit_translate = staticmethod(jax.jit(translate_groups))
        cls.positions = jnp.array(
            [
                [0.3, 0.3, 0.3],
                [0.4, 0.3, 0.3],
                [0.3, 0.4, 0.3],
                [0.4, 0.4, 0.3],
                [0.3, 0.3, 0.6],
                [0.4, 0.3, 0.6],
                [0.3, 0.4, 0.6],
                [0.4, 0.4, 0.6],
                [0.6, 0.3, 0.3],
                [0.7, 0.3, 0.3],
                [0.6, 0.4, 0.3],
                [0.7, 0.4, 0.3],
            ]
        )
        cls.system_ids = jnp.repeat(jnp.arange(cls.n_sys), cls.n_particles_per_sys)
        cls.translations = jnp.array(
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ]
        )

    def _assert_per_system(self, translations, positions=None, unitcell=None):
        """Translate and verify per-system expected positions (cubic wrap)."""
        pos = positions if positions is not None else self.positions
        uc = unitcell if unitcell is not None else self.unitcell
        translated = self._call(
            self.jit_translate, translations, pos, uc, self.system_ids
        )
        for sys_id in range(self.n_sys):
            mask = self.system_ids == sys_id
            expected = (pos[mask] + translations[sys_id] + 0.5) % 1.0 - 0.5
            npt.assert_allclose(translated[mask], expected, atol=1e-10)
        return translated

    def test_translation_formula(self):
        """Merged: basic, zero, wrapping, different-per-system, large, negative,
        exact verification, relative-position preservation, and non-cubic."""
        # Basic translation
        translated = self._assert_per_system(self.translations)
        assert translated.shape == self.positions.shape
        assert jnp.all(jnp.isfinite(translated))

        # Zero translation
        zero = jnp.zeros((self.n_sys, 3))
        t_zero = self._call(
            self.jit_translate, zero, self.positions, self.unitcell, self.system_ids
        )
        npt.assert_allclose(t_zero, (self.positions + 0.5) % 1.0 - 0.5, atol=1e-15)

        # Wrapping with large translations (also checks range)
        large = jnp.array([[1.5, 0.0, 0.0], [0.0, 2.3, 0.0], [0.0, 0.0, -0.7]])
        t_wrap = self._assert_per_system(large)
        assert jnp.all(t_wrap >= -0.5) and jnp.all(t_wrap < 0.5)

        # Different translations per system
        self._assert_per_system(
            jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        )

        # Very large translations
        t_vlarge = self._assert_per_system(
            jnp.array([[5.7, -3.2, 2.8], [-4.1, 6.3, -1.9], [2.4, -5.8, 4.6]])
        )
        assert jnp.all(t_vlarge >= -0.5) and jnp.all(t_vlarge < 0.5)

        # Negative translations
        self._assert_per_system(
            jnp.array([[-0.2, 0.0, 0.0], [0.0, -0.3, 0.0], [0.0, 0.0, -0.1]])
        )

        # Exact position verification (precise per-particle check)
        precise = jnp.array([[0.2, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.15]])
        t_precise = self._call(
            self.jit_translate, precise, self.positions, self.unitcell, self.system_ids
        )
        for i, (pos, sid) in enumerate(zip(self.positions, self.system_ids)):
            expected = (pos + precise[sid] + 0.5) % 1.0 - 0.5
            npt.assert_allclose(t_precise[i], expected, atol=1e-15)

        # Relative positions preserved under translation
        rel_trans = jnp.array([[0.3, 0.2, 0.1], [-0.1, 0.4, -0.2], [0.25, -0.15, 0.35]])
        t_rel = self._call(
            self.jit_translate,
            rel_trans,
            self.positions,
            self.unitcell,
            self.system_ids,
        )
        for sys_id in range(self.n_sys):
            mask = self.system_ids == sys_id
            orig = self.positions[mask]
            trans = t_rel[mask]
            for i in range(len(orig)):
                for j in range(i + 1, len(orig)):
                    _wrap = lambda d: jnp.where(  # noqa: E731
                        d > 0.5, d - 1.0, jnp.where(d < -0.5, d + 1.0, d)
                    )
                    npt.assert_allclose(
                        _wrap(orig[j] - orig[i]), _wrap(trans[j] - trans[i]), atol=1e-10
                    )

        # Non-cubic (hexagonal) unit cell
        lattice_vecs = jnp.array(
            [[1.0, 0.0, 0.0], [0.5, jnp.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]]
        )
        hex_uc = TriclinicUnitCell.from_matrix(
            jnp.broadcast_to(lattice_vecs, (self.n_sys, 3, 3))
        )
        small = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
        t_hex = self._call(
            self.jit_translate, small, self.positions, hex_uc, self.system_ids
        )
        for sys_id in range(self.n_sys):
            mask = self.system_ids == sys_id
            expected = hex_uc[sys_id].wrap(self.positions[mask] + small[sys_id])
            npt.assert_allclose(t_hex[mask], expected, atol=1e-10)

    def test_consistency_with_manual_implementation(self):
        manual_result = jnp.zeros_like(self.positions)
        for i in range(len(self.positions)):
            sys_id = self.system_ids[i]
            new_pos = self.positions[i] + self.translations[sys_id]
            wrapped_pos = (new_pos + 0.5) % 1.0 - 0.5
            manual_result = manual_result.at[i].set(wrapped_pos)
        function_result = self._call(
            self.jit_translate,
            self.translations,
            self.positions,
            self.unitcell,
            self.system_ids,
        )
        npt.assert_allclose(function_result, manual_result, atol=1e-15)

    def test_boundary_crossing_translations(self):
        boundary_positions = jnp.array(
            [
                [0.45, 0.25, 0.25],
                [0.49, 0.25, 0.25],
                [-0.45, 0.25, 0.25],
                [-0.49, 0.25, 0.25],
                [0.25, 0.45, 0.25],
                [0.25, 0.49, 0.25],
                [0.25, -0.45, 0.25],
                [0.25, -0.49, 0.25],
                [0.25, 0.25, 0.45],
                [0.25, 0.25, 0.49],
                [0.25, 0.25, -0.45],
                [0.25, 0.25, -0.49],
            ]
        )
        boundary_system_ids = jnp.repeat(jnp.arange(self.n_sys), 4)
        boundary_translations = jnp.array(
            [
                [0.2, 0.0, 0.0],
                [0.0, 0.15, 0.0],
                [0.0, 0.0, 0.1],
            ]
        )
        translated = self._call(
            self.jit_translate,
            boundary_translations,
            boundary_positions,
            self.unitcell,
            boundary_system_ids,
        )
        for i, (pos, sys_id) in enumerate(zip(boundary_positions, boundary_system_ids)):
            expected_pos = pos + boundary_translations[sys_id]
            expected_wrapped = (expected_pos + 0.5) % 1.0 - 0.5
            npt.assert_allclose(
                translated[i],
                expected_wrapped,
                atol=1e-15,
                err_msg=f"Boundary crossing translation failed for particle {i}",
            )
            assert jnp.all(translated[i] >= -0.5) and jnp.all(translated[i] < 0.5), (
                f"Particle {i} position {translated[i]} is outside valid range [-0.5, 0.5)"
            )

    def test_single_system(self):
        """Non-unit 2x2x2 cubic cell where Cartesian-to-fractional conversion matters."""
        lattice_vecs = 2.0 * jnp.eye(3)
        uc = TriclinicUnitCell.from_matrix(lattice_vecs[None])
        positions = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])
        system_ids = jnp.array([0, 0])
        translations = jnp.array([[0.3, 0.0, 0.0]])
        result = self._call(
            jax.jit(translate_groups), translations, positions, uc, system_ids
        )
        expected = uc[0].wrap(positions + translations[0])
        npt.assert_allclose(result, expected, atol=1e-10)


# Cache for ExchangeMove JIT-compiled function
_exchange_move_cache = {}


def _make_particles(positions, system_ids, n_sys, group_ids, n_groups, max_group_size):
    """Helper to create Buffered particle data for exchange move tests."""
    data = GroupSystemData(
        positions=positions,
        system=_make_index(SystemId, system_ids, n_sys),
        group=_make_index(GroupId, group_ids, n_groups, max_count=max_group_size),
    )
    n = len(positions)
    index = tuple(ParticleId(i) for i in range(n))
    return Buffered(index, data, system_view)


@dataclass
class _GroupData:
    """Group data with motif and system indices for exchange move tests."""

    motif: Index[MotifId]
    system: Index[SystemId]


def _make_groups(species, system_ids, n_sys, max_count=None):
    """Helper to create Buffered group data for exchange move tests."""
    n_groups = len(species)
    n_motifs = int(np.max(np.asarray(species))) + 1

    data = _GroupData(
        motif=_make_index(MotifId, species, n_motifs),
        system=_make_index(SystemId, system_ids, n_sys, max_count=max_count),
    )
    index = tuple(GroupId(i) for i in range(n_groups))
    return Buffered(index, data, system_view)


@dataclass
class _Motifs:
    """Local MotifData implementation for tests."""

    positions: Array
    motif: Index[MotifId]


def _make_motifs(positions, motif_ids, n_motifs, max_motif_size):
    """Helper to create Table[MotifParticleId, MotifData] for exchange move tests."""
    from kups.core.typing import MotifParticleId

    motif_index = _make_index(MotifId, motif_ids, n_motifs, max_count=max_motif_size)
    return Table.arange(
        _Motifs(positions=positions, motif=motif_index), label=MotifParticleId
    )


class TestExchangeMove:
    @classmethod
    def setup_class(cls):
        cls.systems = TriclinicUnitCell.from_matrix(jnp.eye(3))[None]

        if "move" not in _exchange_move_cache:
            move_fn = jax.jit(
                as_result_function(
                    ExchangeMove(
                        positions=lambda state: state["particles"],
                        groups=lambda state: state["groups"],
                        motifs=lambda state: state["motifs"],
                        unitcell=lambda state: Table.arange(
                            state["systems"], label=SystemId
                        ),
                        capacity=lambda state: state["capacity"],
                    )
                )
            )
            _exchange_move_cache["move"] = move_fn

        cls.move = staticmethod(_exchange_move_cache["move"])

    def test_insufficient_sizes_and_capacity(self):
        """Merged: insufficient group size, particle size, and capacity."""
        motifs = _make_motifs(jnp.zeros((1, 3)), jnp.zeros((1,), dtype=int), 1, 1)
        cap1 = LensCapacity(1, lens(lambda x: x["capacity"], cls=dict))
        cases = [
            # (key_seed, positions_shape, sys_ids_p, n_sys_p, group_ids, n_groups,
            #  max_gs, species, sys_ids_g, n_sys_g, max_count_g, capacity, exc, match)
            {
                "label": "insufficient_group_size",
                "key_seed": 2,
                "particles": _make_particles(
                    jax.random.uniform(
                        jax.random.key(2), (11, 3), minval=-0.5, maxval=0.5
                    ),
                    jnp.array([0] * 10 + [1]),
                    1,
                    jnp.arange(11),
                    11,
                    1,
                ),
                "groups": _make_groups(
                    jnp.zeros((10,), dtype=int),
                    jnp.zeros((10,), dtype=int),
                    1,
                ),
                "capacity": cap1,
                "exc": AssertionError,
                "match": "Array size insufficient",
                "method": "fix_or_raise",
            },
            {
                "label": "insufficient_particle_size",
                "key_seed": 3,
                "particles": _make_particles(
                    jax.random.uniform(
                        jax.random.key(3), (10, 3), minval=-0.5, maxval=0.5
                    ),
                    jnp.zeros((10,), dtype=int),
                    1,
                    jnp.arange(10),
                    10,
                    1,
                ),
                "groups": _make_groups(
                    jnp.zeros((11,), dtype=int),
                    jnp.array([0] * 10 + [1]),
                    1,
                    max_count=12,
                ),
                "capacity": cap1,
                "exc": AssertionError,
                "match": "Array size insufficient",
                "method": "fix_or_raise",
            },
            {
                "label": "insufficient_capacity",
                "key_seed": 3,
                "particles": _make_particles(
                    jax.random.uniform(
                        jax.random.key(3), (11, 3), minval=-0.5, maxval=0.5
                    ),
                    jnp.array([0] * 10 + [1]),
                    1,
                    jnp.arange(11),
                    11,
                    1,
                ),
                "groups": _make_groups(
                    jnp.zeros((11,), dtype=int),
                    jnp.array([0] * 10 + [1]),
                    1,
                    max_count=12,
                ),
                "capacity": LensCapacity(0, lens(lambda x: x["capacity"], cls=dict)),
                "exc": CapacityError,
                "match": None,
                "method": "raise_assertion",
            },
        ]
        for case in cases:
            chain = key_chain(jax.random.key(case["key_seed"]))
            _ = next(chain)  # consume one key for uniform
            state = {
                "particles": case["particles"],
                "groups": case["groups"],
                "motifs": motifs,
                "systems": self.systems,
                "capacity": case["capacity"],
            }
            if case["match"]:
                with pytest.raises(case["exc"], match=case["match"]):
                    self.move(next(chain), state).fix_or_raise(state)
            else:
                with pytest.raises(case["exc"]):
                    self.move(next(chain), state).raise_assertion()

    def test_atomic_single_system(self):
        chain = key_chain(jax.random.key(3))
        positions = jax.random.uniform(next(chain), (11, 3), minval=-0.5, maxval=0.5)
        particles = _make_particles(
            positions, jnp.array([0] * 10 + [1]), 1, jnp.arange(11), 11, 1
        )
        groups = _make_groups(
            jnp.zeros((11,), dtype=int),
            jnp.array([0] * 10 + [1]),
            1,
            max_count=12,
        )
        motifs = _make_motifs(jnp.zeros((1, 3)), jnp.zeros((1,), dtype=int), 1, 1)
        state = {
            "particles": particles,
            "groups": groups,
            "motifs": motifs,
            "systems": self.systems,
            "capacity": LensCapacity(1, lens(lambda x: x["capacity"], cls=dict)),
        }
        proposal_result = self.move(next(chain), state)
        proposal_result.raise_assertion()
        npt.assert_allclose(proposal_result.value[1].data, jnp.zeros((1,)))

        proposals = []
        n_inserts = 0
        n_dels = 0
        n = 3
        for _ in range(n):
            proposal = self.move(next(chain), state).value[0]
            proposals.append(proposal)
            assert np.all(proposal.particles.indices.indices < positions.shape[0])
            if proposal.particles.indices.indices >= 10:
                n_inserts += 1
                npt.assert_allclose(proposal.groups.indices.indices, 10)
                npt.assert_allclose(proposal.particles.data.data.group.indices, 10)
                npt.assert_allclose(proposal.particles.data.data.system.indices, 0)
                npt.assert_allclose(proposal.groups.data.data.system.indices, 0)
                assert np.all(
                    (proposal.particles.data.data.new_positions < 0.5)
                    & (proposal.particles.data.data.new_positions > -0.5)
                )
                npt.assert_allclose(proposal.groups.data.data.motif.indices, 0)
            else:
                n_dels += 1
                assert np.all(proposal.groups.indices.indices < 10)
                npt.assert_allclose(proposal.particles.data.data.system.indices, 1)
                npt.assert_allclose(proposal.groups.data.data.system.indices, 1)
        assert 0.1 < n_inserts / n < 0.9, f"Unexpected insert ratio: {n_inserts / n}"

    def test_atomic_two_systems(self):
        chain = key_chain(jax.random.key(4))
        positions = jax.random.uniform(next(chain), (12, 3), minval=-0.5, maxval=0.5)
        particles = _make_particles(
            positions,
            jnp.array([0] * 5 + [1] * 5 + [2] * 2),
            2,
            jnp.arange(12),
            12,
            1,
        )
        groups = _make_groups(
            jnp.zeros((12,), dtype=int),
            jnp.array([0] * 5 + [1] * 5 + [2] * 2),
            2,
            max_count=6,
        )
        motifs = _make_motifs(jnp.zeros((1, 3)), jnp.zeros((1,), dtype=int), 1, 1)
        state = {
            "particles": particles,
            "groups": groups,
            "motifs": motifs,
            "systems": self.systems[jnp.array([0, 0])],
            "capacity": LensCapacity(1, lens(lambda x: x["capacity"], cls=dict)),
        }
        with pytest.raises(CapacityError, match="Insufficient capacity"):
            self.move(next(chain), state).raise_assertion()

        state["capacity"] = LensCapacity(2, lens(lambda x: x["capacity"], cls=dict))
        proposal_result = self.move(next(chain), state)
        proposal_result.raise_assertion()
        npt.assert_allclose(proposal_result.value[1].data, jnp.zeros((2,)))

        proposals = []
        n_inserts = 0
        n_dels = 0
        n_iters = 3
        for _ in range(n_iters):
            proposal = self.move(next(chain), state).value[0]
            proposals.append(proposal)
            assert np.all(proposal.particles.indices.indices < positions.shape[0])
            p_insert_mask = proposal.particles.data.data.system.indices < 2
            g_insert_mask = proposal.groups.data.data.system.indices < 2
            n_inserts += np.sum(p_insert_mask)
            n_dels += np.sum(~p_insert_mask)
            assert np.all(
                (
                    (proposal.particles.data.data.new_positions < 0.5)
                    & (proposal.particles.data.data.new_positions > -0.5)
                )
                | ~p_insert_mask[:, None]
            )
            npt.assert_allclose(g_insert_mask.sum(), p_insert_mask.sum())
            assert np.all(
                (proposal.groups.data.data.motif.indices == 0) | ~g_insert_mask
            )
            assert np.all((proposal.particles.indices.indices < 10) | p_insert_mask)
            assert np.all(
                (proposal.particles.data.data.system.indices == 2) | p_insert_mask
            )
            assert np.all(
                (proposal.groups.data.data.system.indices == 2) | g_insert_mask
            )
        assert 0.1 < n_inserts / (n_iters * 2) < 0.9, (
            f"Unexpected insert ratio: {n_inserts / (n_iters * 2)}"
        )

    def test_multiple_motifs(self):
        chain = key_chain(jax.random.key(5))
        positions = jax.random.uniform(next(chain), (20, 3), minval=-0.5, maxval=0.5)
        particles = _make_particles(
            positions,
            jnp.array([0] * 6 + [1] * 4 + [2] * 10),
            2,
            jnp.arange(20),
            20,
            max_group_size=3,
        )
        groups = _make_groups(
            jnp.array([0, 1, 0, 1] + [0] * 16),
            jnp.array([0, 0, 1, 1] + [2] * 16),
            2,
            max_count=20,
        )
        motifs = _make_motifs(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.05, 0.087, 0.0],
                ]
            ),
            jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            3,
            max_motif_size=3,
        )
        state = {
            "particles": particles,
            "groups": groups,
            "motifs": motifs,
            "systems": self.systems[jnp.array([0, 0])],
            "capacity": LensCapacity(6, lens(lambda x: x["capacity"], cls=dict)),
        }
        proposal_result = self.move(next(chain), state)
        proposal_result.raise_assertion()
        npt.assert_allclose(proposal_result.value[1].data, jnp.zeros((2,)))

        motif_counts = jnp.zeros(3)
        for _ in range(5):
            proposal = self.move(next(chain), state).value[0]
            g_insert_mask = proposal.groups.data.data.system.indices < 2
            if jnp.any(g_insert_mask):
                selected_motifs = proposal.groups.data.data.motif.indices[g_insert_mask]
                for motif_id in selected_motifs:
                    if motif_id < 3:
                        motif_counts = motif_counts.at[motif_id].add(1)
        assert jnp.sum(motif_counts) > 0, f"No motifs were selected: {motif_counts}"

    def test_larger_motif(self):
        chain = key_chain(jax.random.key(6))
        positions = jax.random.uniform(next(chain), (20, 3), minval=-0.5, maxval=0.5)
        particles = _make_particles(
            positions,
            jnp.array([0] * 15 + [1] * 5),
            1,
            jnp.arange(20),
            20,
            max_group_size=5,
        )
        groups = _make_groups(
            jnp.zeros((20,), dtype=int),
            jnp.array([0] * 15 + [1] * 5),
            1,
            max_count=20,
        )
        motif_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, -0.1, 0.0],
            ]
        )
        motifs = _make_motifs(motif_positions, jnp.zeros(5, dtype=int), 1, 5)
        state = {
            "particles": particles,
            "groups": groups,
            "motifs": motifs,
            "systems": self.systems,
            "capacity": LensCapacity(5, lens(lambda x: x["capacity"], cls=dict)),
        }
        proposal_result = self.move(next(chain), state)
        proposal_result.raise_assertion()
        npt.assert_allclose(proposal_result.value[1].data, jnp.zeros((1,)))

        for _ in range(5):
            proposal = self.move(next(chain), state).value[0]
            p_insert_mask = proposal.particles.data.data.system.indices < 1
            n_insertions = jnp.sum(p_insert_mask)
            if n_insertions > 0:
                assert n_insertions == 5, (
                    f"Expected 5 particles in insertion, got {n_insertions}"
                )
                group_ids = proposal.particles.data.data.group.indices[p_insert_mask]
                assert jnp.all(group_ids == group_ids[0]), (
                    "Inserted particles should belong to same group"
                )

    def test_capacity_larger_than_motif_times_systems(self):
        chain = key_chain(jax.random.key(7))
        positions = jax.random.uniform(next(chain), (30, 3), minval=-0.5, maxval=0.5)
        particles = _make_particles(
            positions,
            jnp.array([0] * 8 + [1] * 6 + [2] * 16),
            2,
            jnp.arange(30),
            30,
            max_group_size=4,
        )
        groups = _make_groups(
            jnp.zeros((30,), dtype=int),
            jnp.array([0] * 8 + [1] * 6 + [2] * 16),
            2,
            max_count=30,
        )
        motif_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.05, 0.05, 0.1],
            ]
        )
        motifs = _make_motifs(motif_positions, jnp.zeros(4, dtype=int), 1, 4)
        state = {
            "particles": particles,
            "groups": groups,
            "motifs": motifs,
            "systems": self.systems[jnp.array([0, 0])],
            "capacity": LensCapacity(12, lens(lambda x: x["capacity"], cls=dict)),
        }
        proposal_result = self.move(next(chain), state)
        proposal_result.raise_assertion()
        npt.assert_allclose(proposal_result.value[1].data, jnp.zeros((2,)))

        n_inserts = 0
        n_dels = 0
        for _ in range(5):
            proposal = self.move(next(chain), state).value[0]
            valid_particle_mask = (
                proposal.particles.indices.indices < positions.shape[0]
            )
            assert np.all(
                valid_particle_mask
                | (proposal.particles.indices.indices == positions.shape[0])
            ), "Invalid particle IDs found"
            p_insert_mask = proposal.particles.data.data.system.indices < 2
            n_inserts += np.sum(p_insert_mask)
            n_dels += np.sum(~p_insert_mask)
            if jnp.any(p_insert_mask):
                insert_system_ids = proposal.particles.data.data.system.indices[
                    p_insert_mask
                ]
                assert jnp.all(insert_system_ids < 2), (
                    f"Invalid system IDs: {insert_system_ids}"
                )
                for sys_id in range(2):
                    sys_insertions = jnp.sum(insert_system_ids == sys_id)
                    if sys_insertions > 0:
                        assert sys_insertions <= 4, (
                            f"Too many insertions for system {sys_id}, got {sys_insertions}"
                        )
        assert 0.1 < n_inserts / (n_inserts + n_dels) < 0.9, (
            f"Unexpected insert/delete ratio: {n_inserts}/{n_dels}"
        )

    def test_mixed_motif_sizes(self):
        chain = key_chain(jax.random.key(8))
        positions = jax.random.uniform(next(chain), (35, 3), minval=-0.5, maxval=0.5)
        particles = _make_particles(
            positions,
            jnp.array([0] * 10 + [1] * 8 + [2] * 17),
            2,
            jnp.arange(35),
            35,
            max_group_size=5,
        )
        groups = _make_groups(
            jnp.array([0, 1, 2, 0, 1, 2] + [0] * 29),
            jnp.array([0] * 10 + [1] * 8 + [2] * 17),
            2,
            max_count=35,
        )
        motif_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, -0.1, 0.0],
            ]
        )
        motifs = _make_motifs(
            motif_positions,
            jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            3,
            5,
        )
        state = {
            "particles": particles,
            "groups": groups,
            "motifs": motifs,
            "systems": self.systems[jnp.array([0, 0])],
            "capacity": LensCapacity(10, lens(lambda x: x["capacity"], cls=dict)),
        }
        proposal_result = self.move(next(chain), state)
        proposal_result.raise_assertion()
        npt.assert_allclose(proposal_result.value[1].data, jnp.zeros((2,)))

        motif_usage = jnp.zeros(3)
        for _ in range(5):
            proposal = self.move(next(chain), state).value[0]
            p_insert_mask = proposal.particles.data.data.system.indices < 2
            g_insert_mask = proposal.groups.data.data.system.indices < 2
            if jnp.any(g_insert_mask):
                selected_motifs = proposal.groups.data.data.motif.indices[g_insert_mask]
                for motif_id in selected_motifs:
                    if motif_id < 3:
                        motif_usage = motif_usage.at[motif_id].add(1)
                n_inserted_particles = jnp.sum(p_insert_mask)
                if n_inserted_particles > 0:
                    assert n_inserted_particles <= 10, (
                        f"Too many particles inserted: {n_inserted_particles}"
                    )
        assert jnp.sum(motif_usage) > 0, f"No motifs were used: {motif_usage}"

    def test_high_capacity_multiple_motifs(self):
        chain = key_chain(jax.random.key(9))
        positions = jax.random.uniform(next(chain), (50, 3), minval=-0.5, maxval=0.5)
        particles = _make_particles(
            positions,
            jnp.array([0] * 10 + [1] * 8 + [2] * 32),
            2,
            jnp.arange(50),
            50,
            max_group_size=3,
        )
        groups = _make_groups(
            jnp.zeros((50,), dtype=int),
            jnp.array([0] * 10 + [1] * 8 + [2] * 32),
            2,
            max_count=50,
        )
        motif_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.05, 0.087, 0.0],
            ]
        )
        motifs = _make_motifs(motif_positions, jnp.array([0, 0, 0, 1, 1, 1]), 2, 3)
        state = {
            "particles": particles,
            "groups": groups,
            "motifs": motifs,
            "systems": self.systems[jnp.array([0, 0])],
            "capacity": LensCapacity(15, lens(lambda x: x["capacity"], cls=dict)),
        }
        proposal_result = self.move(next(chain), state)
        proposal_result.raise_assertion()
        npt.assert_allclose(proposal_result.value[1].data, jnp.zeros((2,)))

        max_particles_per_proposal = 0
        total_no_ops = 0
        n_iters = 3
        for _ in range(n_iters):
            proposal = self.move(next(chain), state).value[0]
            particle_ids = proposal.particles.indices.indices
            valid_particle_mask = particle_ids < positions.shape[0]
            no_op_mask = particle_ids >= positions.shape[0]
            n_valid_ops = jnp.sum(valid_particle_mask)
            n_no_ops = jnp.sum(no_op_mask)
            total_no_ops += int(n_no_ops)
            total_ops = n_valid_ops + n_no_ops
            assert total_ops == state["capacity"].size, (
                f"Expected {state['capacity'].size} total operations, got {total_ops}"
            )
            p_insert_mask = (
                proposal.particles.data.data.system.indices < 2
            ) & valid_particle_mask
            n_inserted = jnp.sum(p_insert_mask)
            max_particles_per_proposal = max(
                max_particles_per_proposal, int(n_inserted)
            )
            if jnp.any(no_op_mask):
                no_op_particle_ids = particle_ids[no_op_mask]
                assert jnp.all(no_op_particle_ids == positions.shape[0]), (
                    f"No-op particle IDs should be {positions.shape[0]}, got {no_op_particle_ids}"
                )
            if n_inserted > 0:
                insert_system_ids = proposal.particles.data.data.system.indices[
                    p_insert_mask
                ]
                assert jnp.all(insert_system_ids < 2), (
                    f"Invalid system IDs: {insert_system_ids}"
                )
                insert_group_ids = proposal.particles.data.data.group.indices[
                    p_insert_mask
                ]
                assert jnp.all(insert_group_ids < groups.size), (
                    f"Invalid group IDs: {insert_group_ids}"
                )
        assert max_particles_per_proposal >= 3, (
            f"Expected at least motif-sized proposals with high capacity, max was {max_particles_per_proposal}"
        )
        avg_no_ops_per_proposal = total_no_ops / n_iters
        assert avg_no_ops_per_proposal >= 8, (
            f"Expected significant no-ops with high capacity, avg was {avg_no_ops_per_proposal}"
        )


@dataclass
class _GroupDataWithMotif:
    """Group data satisfying HasMotifAndSystemIndex for insert/delete tests."""

    motif: Index[MotifId]
    system: Index[SystemId]


def _make_groups_with_motif(motif_ids, system_ids, n_sys, n_motifs, max_count=None):
    """Create Buffered groups satisfying HasMotifAndSystemIndex."""

    n = len(motif_ids)
    data = _GroupDataWithMotif(
        motif=_make_index(MotifId, motif_ids, n_motifs),
        system=_make_index(SystemId, system_ids, n_sys, max_count=max_count),
    )
    index = tuple(GroupId(i) for i in range(n))
    return Buffered(index, data, system_view)


def _exchange_state():
    """Shared state for insert/delete tests: 10 occupied + 1 free slot."""
    from kups.core.typing import MotifParticleId

    positions = jnp.zeros((11, 3))
    particles = _make_particles(
        positions, jnp.array([0] * 10 + [1]), 1, jnp.arange(11), 11, 1
    )
    groups = _make_groups_with_motif(
        jnp.zeros((11,), dtype=int),
        jnp.array([0] * 10 + [1]),
        n_sys=1,
        n_motifs=1,
        max_count=12,
    )
    motif_data = _Motifs(jnp.zeros((1, 3)), _make_index(MotifId, [0], 1, max_count=1))
    motifs = Table.arange(motif_data, label=MotifParticleId)
    uc = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10)
    return particles, groups, motifs, Table.arange(uc, label=SystemId)


class TestInsertRandomMotif:
    """Tests for insert_random_motif exchange change structure."""

    def test_insert_random_motif(self):
        particles, groups, motifs, uc = _exchange_state()
        cap = LensCapacity(1, lens(lambda x: x, cls=int))
        result = insert_random_motif(
            jax.random.key(0), motifs, particles, groups, uc, cap
        )
        # Type checks
        assert isinstance(result, ExchangeChanges)
        assert isinstance(result.particles, WithIndices)
        assert isinstance(result.particles.data, Buffered)
        assert isinstance(result.groups, WithIndices)
        assert isinstance(result.groups.data, Buffered)
        # Occupation all true for insertion
        assert result.particles.data.occupation.all()
        assert result.groups.data.occupation.all()
        # Non-empty indices
        assert result.particles.indices.indices.shape[0] > 0
        assert result.groups.indices.indices.shape[0] > 0


class TestDeleteRandomMotif:
    """Tests for delete_random_motif exchange change structure."""

    def test_delete_random_motif(self):
        particles, groups, motifs, _ = _exchange_state()
        cap = LensCapacity(1, lens(lambda x: x, cls=int))
        result = delete_random_motif(jax.random.key(0), motifs, particles, groups, cap)
        # Type checks
        assert isinstance(result, ExchangeChanges)
        assert isinstance(result.particles, WithIndices)
        assert isinstance(result.particles.data, Buffered)
        assert isinstance(result.groups, WithIndices)
        assert isinstance(result.groups.data, Buffered)
        # Occupation all false for deletion
        assert not result.particles.data.occupation.any()
        assert not result.groups.data.occupation.any()
