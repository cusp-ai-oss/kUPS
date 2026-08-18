# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.utils.particles."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.application.utils.particles import Particles, particles_from_arrays
from kups.core.cell import Cell, PeriodicCell, TriclinicFrame, VacuumCell
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
        particles, cell, uc_transform = particles_from_ase(atoms)

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
        particles, cell, uc_transform = particles_from_ase(atoms)

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
        particles, cell, uc_transform = particles_from_ase(str(path))
        assert len(particles) == 1
        npt.assert_array_equal(particles.data.atomic_numbers, jnp.array([18]))

    def test_uc_transform_produces_lower_triangular(self):
        from ase.build import bulk

        from kups.application.utils.particles import particles_from_ase

        atoms = bulk("Cu")
        _, cell, _ = particles_from_ase(atoms)
        lv = cell.vectors
        # Lower-triangular: upper triangle (excluding diagonal) should be zero
        npt.assert_allclose(lv[..., 0, 1], 0.0, atol=1e-10)
        npt.assert_allclose(lv[..., 0, 2], 0.0, atol=1e-10)
        npt.assert_allclose(lv[..., 1, 2], 0.0, atol=1e-10)


class TestParticlesFromAsePbcDispatch:
    """`particles_from_ase` constructs the right Cell flavor for each pbc shape."""

    @staticmethod
    def _atoms(pbc):
        from ase import Atoms

        return Atoms("Ar", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=pbc)

    def test_fully_periodic_constructs_periodic_cell(self):
        from kups.application.utils.particles import particles_from_ase
        from kups.core.cell import PeriodicCell

        _, cell, _ = particles_from_ase(self._atoms((True, True, True)))
        assert isinstance(cell, PeriodicCell)
        assert cell.periodic == (True, True, True)

    def test_no_pbc_constructs_vacuum_cell(self):
        from kups.application.utils.particles import particles_from_ase
        from kups.core.cell import VacuumCell

        _, cell, _ = particles_from_ase(self._atoms((False, False, False)))
        assert isinstance(cell, VacuumCell)
        assert cell.periodic == (False, False, False)

    @pytest.mark.parametrize(
        "pbc",
        [(True, True, False), (True, False, True), (False, True, True)],
        ids=["xy", "xz", "yz"],
    )
    def test_2d_slab_carries_runtime_mask(self, pbc):
        from kups.application.utils.particles import particles_from_ase
        from kups.core.cell import Cell, PeriodicCell, VacuumCell

        _, cell, _ = particles_from_ase(self._atoms(pbc))
        assert isinstance(cell, Cell)
        assert not isinstance(cell, (PeriodicCell, VacuumCell))
        assert cell.periodic == pbc

    @pytest.mark.parametrize(
        "pbc",
        [(True, False, False), (False, True, False), (False, False, True)],
        ids=["x", "y", "z"],
    )
    def test_1d_wire_carries_runtime_mask(self, pbc):
        from kups.application.utils.particles import particles_from_ase
        from kups.core.cell import Cell, PeriodicCell, VacuumCell

        _, cell, _ = particles_from_ase(self._atoms(pbc))
        assert isinstance(cell, Cell)
        assert not isinstance(cell, (PeriodicCell, VacuumCell))
        assert cell.periodic == pbc


class TestParticlesFromArrays:
    """Test the public source-neutral particle constructor."""

    @staticmethod
    def _valid_inputs():
        return {
            "positions": np.array([[0.2, 0.4, 0.6], [1.1, 1.3, 1.5]]),
            "cell_vectors": np.diag([4.0, 5.0, 6.0]),
            "periodicity": (True, True, True),
            "masses": np.array([22.99, 35.45]),
            "atomic_numbers": np.array([11, 17]),
            "labels": ["Na", "Cl"],
        }

    @staticmethod
    def _assert_particle_parity(actual, expected):
        assert actual.keys == expected.keys
        assert all(isinstance(key, ParticleId) for key in actual.keys)
        npt.assert_allclose(actual.data.positions, expected.data.positions)
        npt.assert_allclose(actual.data.masses, expected.data.masses)
        npt.assert_array_equal(actual.data.atomic_numbers, expected.data.atomic_numbers)
        npt.assert_allclose(actual.data.charges, expected.data.charges)
        assert actual.data.labels.keys == expected.data.labels.keys
        npt.assert_array_equal(actual.data.labels.indices, expected.data.labels.indices)
        assert actual.data.system.keys == expected.data.system.keys
        npt.assert_array_equal(actual.data.system.indices, expected.data.system.indices)

    def test_numpy_orthogonal_periodic_matches_ase(self):
        from ase import Atoms

        from kups.application.utils.particles import particles_from_ase

        inputs = self._valid_inputs()
        charges = np.array([0.75, -0.75])
        atoms = Atoms(
            "NaCl",
            positions=inputs["positions"],
            cell=inputs["cell_vectors"],
            pbc=True,
        )
        atoms.set_masses(inputs["masses"])
        atoms.info["_atom_type_partial_charge"] = charges
        atoms.info["_atom_site_label"] = inputs["labels"]

        expected_particles, expected_cell, expected_transform = particles_from_ase(
            atoms
        )
        inputs["periodicity"] = np.array([True, True, True], dtype=bool)
        actual_particles, actual_cell, actual_transform = particles_from_arrays(
            **inputs,
            charges=charges,
        )

        self._assert_particle_parity(actual_particles, expected_particles)
        assert isinstance(actual_cell, PeriodicCell)
        assert type(actual_cell) is type(expected_cell)
        assert type(actual_cell.frame) is type(expected_cell.frame)
        assert isinstance(actual_cell.frame, TriclinicFrame)
        assert actual_cell.periodic == expected_cell.periodic
        npt.assert_allclose(actual_cell.vectors, expected_cell.vectors)
        npt.assert_allclose(actual_cell.frame.tril, expected_cell.frame.tril)
        probe = jnp.array([[0.25, 0.5, 0.75], [-0.4, 1.2, 0.3]])
        npt.assert_allclose(
            actual_transform(probe), expected_transform(probe), atol=1e-12
        )

    def test_jax_non_lower_triangular_joint_transformation(self):
        from ase import Atoms

        from kups.application.utils.particles import particles_from_ase

        positions = jnp.array([[0.4, 0.8, 1.2], [1.5, 0.3, 2.1]])
        cell_vectors = jnp.array([[0.0, 2.0, 0.0], [1.5, 0.2, 0.0], [0.3, 0.4, 3.0]])
        masses = jnp.array([12.0, 16.0])
        atomic_numbers = jnp.array([6, 8])
        labels = ["C", "O"]
        atoms = Atoms(
            "CO",
            positions=np.asarray(positions),
            cell=np.asarray(cell_vectors),
            pbc=(True, False, True),
        )
        atoms.set_masses(np.asarray(masses))

        expected_particles, expected_cell, expected_transform = particles_from_ase(
            atoms
        )
        actual_particles, actual_cell, actual_transform = particles_from_arrays(
            positions=positions,
            cell_vectors=cell_vectors,
            periodicity=jnp.array([True, False, True]),
            masses=masses,
            atomic_numbers=atomic_numbers,
            labels=labels,
        )

        self._assert_particle_parity(actual_particles, expected_particles)
        assert type(actual_cell) is Cell
        assert actual_cell.periodic == (True, False, True)
        assert all(type(value) is bool for value in actual_cell.periodic)
        npt.assert_allclose(actual_cell.vectors, expected_cell.vectors, atol=1e-12)
        npt.assert_allclose(
            actual_particles.data.positions,
            actual_transform(positions),
            atol=1e-12,
        )
        npt.assert_allclose(
            jnp.triu(actual_cell.vectors, k=1), jnp.zeros((3, 3)), atol=1e-12
        )
        probe = jnp.array([[0.6, -0.2, 0.9], [1.1, 0.7, -0.5]])
        npt.assert_allclose(
            actual_transform(probe), expected_transform(probe), atol=1e-12
        )

    def test_omitted_charges_default_to_floating_zeros(self):
        particles, _, _ = particles_from_arrays(**self._valid_inputs())

        npt.assert_array_equal(particles.data.charges, jnp.zeros(2))
        assert jnp.issubdtype(particles.data.charges.dtype, jnp.floating)

    def test_integer_masses_and_charges_are_promoted(self):
        inputs = self._valid_inputs()
        inputs["positions"] = np.array([[0, 1, 2], [1, 2, 3]], dtype=int)
        inputs["cell_vectors"] = np.diag(np.array([4, 5, 6], dtype=int))
        inputs["masses"] = np.array([23, 35], dtype=int)
        charges = np.array([1, -1], dtype=int)

        particles, cell, _ = particles_from_arrays(**inputs, charges=charges)

        assert jnp.issubdtype(particles.data.masses.dtype, jnp.floating)
        assert jnp.issubdtype(particles.data.charges.dtype, jnp.floating)
        assert jnp.issubdtype(particles.data.positions.dtype, jnp.floating)
        assert jnp.issubdtype(cell.vectors.dtype, jnp.floating)
        npt.assert_array_equal(particles.data.masses, jnp.array([23.0, 35.0]))
        npt.assert_array_equal(particles.data.charges, jnp.array([1.0, -1.0]))

    def test_numpy_boolean_mask_constructs_periodic_cell(self):
        inputs = self._valid_inputs()
        inputs["periodicity"] = np.array([True, True, True], dtype=bool)

        _, cell, _ = particles_from_arrays(**inputs)

        assert isinstance(cell, PeriodicCell)
        assert cell.periodic == (True, True, True)

    def test_numpy_boolean_tuple_constructs_vacuum_cell(self):
        inputs = self._valid_inputs()
        inputs["periodicity"] = (np.bool_(False),) * 3

        _, cell, _ = particles_from_arrays(**inputs)

        assert isinstance(cell, VacuumCell)
        assert cell.periodic == (False, False, False)

    def test_mixed_mask_constructs_generic_cell_with_python_booleans(self):
        inputs = self._valid_inputs()
        inputs["periodicity"] = jnp.array([True, False, True])

        _, cell, _ = particles_from_arrays(**inputs)

        assert type(cell) is Cell
        assert type(cell.periodic) is tuple
        assert cell.periodic == (True, False, True)
        assert all(type(value) is bool for value in cell.periodic)

    @pytest.mark.parametrize(
        ("argument", "value"),
        [
            ("positions", np.zeros((2, 2))),
            ("cell_vectors", np.eye(2)),
        ],
        ids=["positions", "cell-vectors"],
    )
    def test_invalid_geometry_shape(self, argument, value):
        inputs = self._valid_inputs()
        inputs[argument] = value

        with pytest.raises(ValueError, match=argument):
            particles_from_arrays(**inputs)

    @pytest.mark.parametrize(
        ("argument", "value"),
        [
            ("masses", np.ones(1)),
            ("atomic_numbers", np.array([11, 17, 19])),
            ("charges", np.zeros(1)),
        ],
    )
    def test_mismatched_particle_field_length(self, argument, value):
        inputs = self._valid_inputs()
        inputs[argument] = value

        with pytest.raises(ValueError, match=argument):
            particles_from_arrays(**inputs)

    def test_incorrect_label_count(self):
        inputs = self._valid_inputs()
        inputs["labels"] = ["Na"]

        with pytest.raises(ValueError, match="labels"):
            particles_from_arrays(**inputs)

    @pytest.mark.parametrize("labels", ["Na", b"Na"], ids=["str", "bytes"])
    def test_string_like_scalar_labels_are_rejected(self, labels):
        inputs = self._valid_inputs()
        inputs["labels"] = labels

        with pytest.raises(TypeError, match="labels"):
            particles_from_arrays(**inputs)

    def test_non_string_label(self):
        inputs = self._valid_inputs()
        inputs["labels"] = ["Na", 17]

        with pytest.raises(TypeError, match="labels"):
            particles_from_arrays(**inputs)

    def test_floating_atomic_numbers_are_rejected(self):
        inputs = self._valid_inputs()
        inputs["atomic_numbers"] = np.array([11.0, 17.0])

        with pytest.raises(TypeError, match="atomic_numbers"):
            particles_from_arrays(**inputs)

    @pytest.mark.parametrize(
        ("argument", "value"),
        [
            ("positions", np.zeros((2, 3), dtype=bool)),
            ("cell_vectors", np.eye(3, dtype=complex)),
            ("masses", np.array(["23", "35"])),
            ("charges", np.array([1.0, object()], dtype=object)),
            ("atomic_numbers", np.array([True, False])),
        ],
        ids=["boolean", "complex", "string", "object", "atomic-boolean"],
    )
    def test_incompatible_physical_dtype(self, argument, value):
        inputs = self._valid_inputs()
        inputs[argument] = value

        with pytest.raises(TypeError, match=argument):
            particles_from_arrays(**inputs)

    @pytest.mark.parametrize(
        "periodicity",
        [(1, 1, 0), ("yes", "yes", "no")],
        ids=["integer", "string"],
    )
    def test_non_boolean_periodicity_is_rejected(self, periodicity):
        inputs = self._valid_inputs()
        inputs["periodicity"] = periodicity

        with pytest.raises(TypeError, match="periodicity"):
            particles_from_arrays(**inputs)

    def test_periodicity_requires_exactly_three_values(self):
        inputs = self._valid_inputs()
        inputs["periodicity"] = [True, False]

        with pytest.raises(ValueError, match="periodicity"):
            particles_from_arrays(**inputs)
