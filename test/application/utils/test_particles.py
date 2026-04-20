# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.utils.particles."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from kups.application.utils.particles import Particles
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import InclusionId, Label, ParticleId, SystemId


class TestParticlesCreation:
    """Test Particles dataclass construction and Index fields."""

    def test_basic_construction(self):
        n = 4
        particles = Particles(
            positions=jnp.zeros((n, 3)),
            masses=jnp.ones(n),
            atomic_numbers=jnp.array([29, 29, 29, 29]),
            charges=jnp.zeros(n),
            labels=Index.new([Label("Cu")] * 4),
            system=Index.integer(jnp.zeros(n, dtype=int), label=SystemId),
        )
        assert particles.positions.shape == (n, 3)
        assert particles.masses.shape == (n,)
        assert particles.atomic_numbers.shape == (n,)
        assert particles.charges.shape == (n,)

    def test_system_index(self):
        n = 4
        system = Index.integer(jnp.zeros(n, dtype=int), label=SystemId)
        particles = Particles(
            positions=jnp.zeros((n, 3)),
            masses=jnp.ones(n),
            atomic_numbers=jnp.ones(n, dtype=int),
            charges=jnp.zeros(n),
            labels=Index.new([Label("A"), Label("A"), Label("B"), Label("B")]),
            system=system,
        )
        assert particles.system.keys == (SystemId(0),)
        npt.assert_array_equal(particles.system.indices, jnp.zeros(n, dtype=int))

    def test_multi_system(self):
        ids = jnp.array([0, 0, 1, 1])
        system = Index.integer(ids, label=SystemId)
        particles = Particles(
            positions=jnp.zeros((4, 3)),
            masses=jnp.ones(4),
            atomic_numbers=jnp.ones(4, dtype=int),
            charges=jnp.zeros(4),
            labels=Index.new([Label("A"), Label("A"), Label("B"), Label("B")]),
            system=system,
        )
        assert particles.system.keys == (SystemId(0), SystemId(1))
        npt.assert_array_equal(particles.system.indices, ids)


class TestInclusionProperty:
    """Test that inclusion returns system re-labeled as InclusionId."""

    def test_inclusion_labels(self):
        n = 3
        system = Index.integer(jnp.zeros(n, dtype=int), label=SystemId)
        particles = Particles(
            positions=jnp.zeros((n, 3)),
            masses=jnp.ones(n),
            atomic_numbers=jnp.ones(n, dtype=int),
            charges=jnp.zeros(n),
            labels=Index.new([Label("X")] * 3),
            system=system,
        )
        inclusion = particles.inclusion
        assert isinstance(inclusion, Index)
        assert all(isinstance(lbl, InclusionId) for lbl in inclusion.keys)
        assert inclusion.keys == (InclusionId(0),)
        npt.assert_array_equal(inclusion.indices, system.indices)

    def test_multi_system_inclusion(self):
        ids = jnp.array([0, 1, 0, 1])
        system = Index.integer(ids, label=SystemId)
        particles = Particles(
            positions=jnp.zeros((4, 3)),
            masses=jnp.ones(4),
            atomic_numbers=jnp.ones(4, dtype=int),
            charges=jnp.zeros(4),
            labels=Index.new([Label("A"), Label("B"), Label("A"), Label("B")]),
            system=system,
        )
        inclusion = particles.inclusion
        assert inclusion.keys == (InclusionId(0), InclusionId(1))
        npt.assert_array_equal(inclusion.indices, ids)


class TestParticlesFromAse:
    """Test particles_from_ase with a simple ASE structure."""

    def test_bulk_cu(self):
        from ase.build import bulk

        from kups.application.utils.particles import particles_from_ase

        atoms = bulk("Cu")
        particles, unitcell, uc_transform = particles_from_ase(atoms)

        # Check return type
        assert isinstance(particles, Table)

        # Check index labels are ParticleId
        assert all(isinstance(lbl, ParticleId) for lbl in particles.keys)
        assert len(particles) == len(atoms)

        # Check positions shape
        assert particles.data.positions.shape == (len(atoms), 3)

        # Check masses
        npt.assert_allclose(
            particles.data.masses, jnp.asarray(atoms.get_masses()), rtol=1e-5
        )

        # Check atomic numbers
        npt.assert_array_equal(
            particles.data.atomic_numbers,
            jnp.asarray(atoms.get_atomic_numbers()),
        )

        # Check charges default to zeros
        npt.assert_array_equal(particles.data.charges, jnp.zeros(len(atoms)))

        # Check labels
        assert particles.data.labels.keys == ("Cu",)

        # Check system index — single system
        assert particles.data.system.keys == (SystemId(0),)
        npt.assert_array_equal(
            particles.data.system.indices, jnp.zeros(len(atoms), dtype=int)
        )

    def test_multi_element(self):
        from ase import Atoms

        from kups.application.utils.particles import particles_from_ase

        atoms = Atoms(
            "NaCl",
            positions=[[0, 0, 0], [2.8, 0, 0]],
            cell=[5.6, 5.6, 5.6],
            pbc=True,
        )
        particles, unitcell, uc_transform = particles_from_ase(atoms)

        assert len(particles) == 2
        assert set(particles.data.labels.keys) == {"Cl", "Na"}
        npt.assert_array_equal(
            particles.data.atomic_numbers, jnp.asarray(atoms.get_atomic_numbers())
        )

    def test_with_charges_in_info(self):
        from ase import Atoms

        from kups.application.utils.particles import particles_from_ase

        atoms = Atoms(
            "NaCl",
            positions=[[0, 0, 0], [2.8, 0, 0]],
            cell=[5.6, 5.6, 5.6],
            pbc=True,
        )
        charges = np.array([1.0, -1.0])
        atoms.info["_atom_type_partial_charge"] = charges
        particles, _, _ = particles_from_ase(atoms)
        npt.assert_allclose(particles.data.charges, jnp.asarray(charges))

    def test_from_file_path(self, tmp_path):
        from ase import Atoms
        from ase.io import write

        from kups.application.utils.particles import particles_from_ase

        atoms = Atoms("Ar", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        path = tmp_path / "test.cif"
        write(str(path), atoms)
        particles, unitcell, uc_transform = particles_from_ase(str(path))
        assert len(particles) == 1
        npt.assert_array_equal(particles.data.atomic_numbers, jnp.array([18]))

    def test_uc_transform_produces_lower_triangular(self):
        from ase.build import bulk

        from kups.application.utils.particles import particles_from_ase

        atoms = bulk("Cu")
        _, unitcell, _ = particles_from_ase(atoms)
        lv = unitcell.lattice_vectors
        # Lower-triangular: upper triangle (excluding diagonal) should be zero
        npt.assert_allclose(lv[..., 0, 1], 0.0, atol=1e-10)
        npt.assert_allclose(lv[..., 0, 2], 0.0, atol=1e-10)
        npt.assert_allclose(lv[..., 1, 2], 0.0, atol=1e-10)
