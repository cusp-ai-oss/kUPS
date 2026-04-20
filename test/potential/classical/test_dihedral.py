# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UFF dihedral/torsion potential implementation."""

import jax
import jax.numpy as jnp
import pytest

from kups.core.data.index import Index
from kups.core.neighborlist import Edges
from kups.potential.classical.dihedral import DihedralParameters, dihedral_energy
from kups.potential.common.graph import GraphPotentialInput, HyperGraph

from .conftest import make_particles, make_systems

_LABELS = ("A", "B")


class TestDihedralParametersFromUFF:
    """Test DihedralParameters.from_uff for all UFF cases."""

    @classmethod
    def setup_class(cls):
        cls.labels = ("sp3", "sp2", "sp3g6", "nonmain")
        cls.bond_angle = jnp.array(
            [
                jnp.radians(109.5),
                jnp.radians(120.0),
                jnp.radians(109.5),
                jnp.radians(109.5),
            ]
        )
        cls.torsion_sp3 = jnp.array([2.0, 0.0, 6.8, 1.0])
        cls.torsion_sp2 = jnp.array([0.0, 2.0, 0.0, 0.0])
        cls.group = jnp.array([14, 14, 6, 0])

    def _from_uff(self, bond_order=None):
        return DihedralParameters.from_uff(
            self.labels,
            self.bond_angle,
            self.torsion_sp3,
            self.torsion_sp2,
            self.group,
            bond_order,
        )

    def test_sp3_sp3_general(self):
        params = self._from_uff()
        assert jnp.isclose(params.V[0, 0, 0, 0], 2.0)
        assert params.n[0, 0, 0, 0] == 3
        assert jnp.isclose(params.phi0[0, 0, 0, 0], jnp.pi)

    def test_sp3_sp3_group6(self):
        params = self._from_uff()
        assert jnp.isclose(params.V[0, 2, 2, 0], 6.8)
        assert params.n[0, 2, 2, 0] == 2
        assert jnp.isclose(params.phi0[0, 2, 2, 0], jnp.pi / 2)

    def test_sp2_sp2(self):
        params = self._from_uff()
        assert jnp.isclose(params.V[0, 1, 1, 0], 10.0)
        assert params.n[0, 1, 1, 0] == 2
        assert jnp.isclose(params.phi0[0, 1, 1, 0], jnp.pi)

    def test_sp2_sp2_with_bond_order(self):
        bond_order = jnp.ones((4, 4)) * 2.0
        params = self._from_uff(bond_order)
        expected_V = 10.0 * (1.0 + 4.18 * jnp.log(2.0))
        assert jnp.isclose(params.V[0, 1, 1, 0], expected_V, rtol=1e-5)

    def test_sp3_sp2_general(self):
        params = self._from_uff()
        assert jnp.isclose(params.V[0, 0, 1, 0], 1.0)
        assert params.n[0, 0, 1, 0] == 6
        assert jnp.isclose(params.phi0[0, 0, 1, 0], 0.0)

    def test_sp3_sp2_propene(self):
        params = self._from_uff()
        assert jnp.isclose(params.V[1, 1, 0, 0], 2.0)
        assert params.n[1, 1, 0, 0] == 3
        assert jnp.isclose(params.phi0[1, 1, 0, 0], jnp.pi)
        assert jnp.isclose(params.V[0, 0, 1, 1], 2.0)
        assert params.n[0, 0, 1, 1] == 3
        assert jnp.isclose(params.phi0[0, 0, 1, 1], jnp.pi)

    def test_sp3_sp2_group6(self):
        params = self._from_uff()
        assert params.n[0, 2, 1, 0] == 2
        assert jnp.isclose(params.phi0[0, 2, 1, 0], jnp.pi / 2)

    def test_non_main_group(self):
        params = self._from_uff()
        assert jnp.isclose(params.V[0, 3, 0, 0], 0.0)
        assert jnp.isclose(params.V[0, 0, 3, 0], 0.0)
        assert jnp.isclose(params.V[0, 3, 3, 0], 0.0)


def _positions_for_dihedral(phi) -> jax.Array:
    """Generate positions for 4 atoms with dihedral angle phi."""
    i = jnp.array([0.0, 0.0, 0.0])
    j = jnp.array([1.0, 0.0, 0.0])
    k = jnp.array([1.5, 1.0, 0.0])

    fourth_trans = jnp.array([2.5, 1.0, 0.0])
    jk = k - j
    jk_norm = jk / jnp.linalg.norm(jk)
    kl_trans = fourth_trans - k

    angle = phi - jnp.pi
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    kl_rot = (
        kl_trans * cos_a
        + jnp.cross(jk_norm, kl_trans) * sin_a
        + jk_norm * jnp.dot(jk_norm, kl_trans) * (1 - cos_a)
    )
    fourth = k + kl_rot

    return jnp.stack([i, j, k, fourth])


_jit_energy = jax.jit(dihedral_energy)


class TestDihedralEnergy:
    """Test the dihedral_energy function."""

    @classmethod
    def setup_class(cls):
        cls.species = ["A", "A", "A", "A"]
        cls.system_ids = [0, 0, 0, 0]
        cls.unitcells_lv = jnp.eye(3)[None] * 20.0
        shape = (2, 2, 2, 2)
        cls.V = jnp.ones(shape) * 2.0
        cls.n = jnp.ones(shape) * 3.0
        cls.phi0 = jnp.ones(shape) * jnp.pi
        cls.params = DihedralParameters(labels=_LABELS, V=cls.V, n=cls.n, phi0=cls.phi0)

    def _make_graph(self, positions):
        particles = make_particles(positions, self.species, self.system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2, 3]])),
            shifts=jnp.zeros((1, 3, 3)),
        )
        return HyperGraph(particles, systems, edges)

    def _energy(self, positions, params):
        graph = self._make_graph(positions)
        return _jit_energy(
            GraphPotentialInput(graph=graph, parameters=params)
        ).data.data[0]

    def _gradient(self, positions, params):
        return jax.jit(jax.grad(lambda p: self._energy(p, params)))(positions)

    def test_energy_values(self):
        """Merged: equilibrium, nonequilibrium, and formula tests."""
        # Equilibrium: energy=0 at phi0
        for phi0_deg, n in [(180, 3), (180, 2), (0, 6), (60, 3)]:
            phi0 = jnp.radians(phi0_deg)
            shape = (2, 2, 2, 2)
            params = DihedralParameters(
                labels=_LABELS,
                V=self.V,
                n=jnp.ones(shape) * n,
                phi0=jnp.ones(shape) * phi0,
            )
            assert jnp.isclose(
                self._energy(_positions_for_dihedral(phi0), params), 0.0, atol=1e-5
            )

        # Nonequilibrium: energy > 0
        for phi0_deg, test_deg, n in [(180, 45, 3), (180, 90, 3), (0, 45, 6)]:
            phi0 = jnp.radians(phi0_deg)
            shape = (2, 2, 2, 2)
            params = DihedralParameters(
                labels=_LABELS,
                V=self.V,
                n=jnp.ones(shape) * n,
                phi0=jnp.ones(shape) * phi0,
            )
            assert (
                self._energy(_positions_for_dihedral(jnp.radians(test_deg)), params) > 0
            )

        # Formula: trans=0, cis=2, 90deg=1
        assert jnp.isclose(
            self._energy(_positions_for_dihedral(jnp.pi), self.params), 0.0, atol=1e-5
        )
        assert jnp.isclose(
            self._energy(_positions_for_dihedral(0.0), self.params), 2.0, atol=1e-5
        )
        assert jnp.isclose(
            self._energy(_positions_for_dihedral(jnp.pi / 2), self.params),
            1.0,
            atol=1e-5,
        )

        # Zero barrier
        params = DihedralParameters(
            labels=_LABELS, V=jnp.zeros_like(self.V), n=self.n, phi0=self.phi0
        )
        assert jnp.isclose(
            self._energy(_positions_for_dihedral(0.5), params), 0.0, atol=1e-10
        )

    def test_gradient_values(self):
        """Merged: equilibrium gradient=0, nonequilibrium gradient nonzero."""
        for phi0_deg, n in [(180, 3), (180, 2), (0, 6), (60, 3)]:
            phi0 = jnp.radians(phi0_deg)
            shape = (2, 2, 2, 2)
            params = DihedralParameters(
                labels=_LABELS,
                V=self.V,
                n=jnp.ones(shape) * n,
                phi0=jnp.ones(shape) * phi0,
            )
            assert jnp.allclose(
                self._gradient(_positions_for_dihedral(phi0), params), 0.0, atol=1e-5
            )

        for phi0_deg, test_deg, n in [(180, 45, 3), (180, 90, 3), (0, 45, 6)]:
            phi0 = jnp.radians(phi0_deg)
            shape = (2, 2, 2, 2)
            params = DihedralParameters(
                labels=_LABELS,
                V=self.V,
                n=jnp.ones(shape) * n,
                phi0=jnp.ones(shape) * phi0,
            )
            grad = self._gradient(
                _positions_for_dihedral(jnp.radians(test_deg)), params
            )
            assert jnp.linalg.norm(grad) > 1e-6

    def test_assertion_wrong_order(self):
        positions = _positions_for_dihedral(0.0)[:3]
        species = ["A", "A", "A"]
        system_ids = [0, 0, 0]
        particles = make_particles(positions, species, system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2]])),
            shifts=jnp.zeros((1, 2, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        with pytest.raises(AssertionError, match="quadruplet interactions"):
            dihedral_energy(GraphPotentialInput(graph=graph, parameters=self.params))

    def test_different_species(self):
        species = ["A", "B", "A", "B"]
        system_ids = [0, 0, 0, 0]
        V = jnp.zeros((2, 2, 2, 2)).at[0, 1, 0, 1].set(4.0)
        params = DihedralParameters(labels=_LABELS, V=V, n=self.n, phi0=self.phi0)
        positions = _positions_for_dihedral(jnp.pi)
        particles = make_particles(positions, species, system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2, 3]])),
            shifts=jnp.zeros((1, 3, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        result = dihedral_energy(GraphPotentialInput(graph=graph, parameters=params))
        assert jnp.isclose(result.data.data[0], 0.0, atol=1e-5)

    def test_multiple_dihedrals(self):
        positions = jnp.concatenate(
            [_positions_for_dihedral(jnp.pi), jnp.array([[3.0, 2.0, 0.0]])]
        )
        species = ["A", "A", "A", "A", "A"]
        system_ids = [0, 0, 0, 0, 0]
        particles = make_particles(positions, species, system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2, 3], [1, 2, 3, 4]])),
            shifts=jnp.zeros((2, 3, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        result = dihedral_energy(
            GraphPotentialInput(graph=graph, parameters=self.params)
        )
        assert result.data.data.shape == (1,)
        assert jnp.isfinite(result.data.data[0])

    def test_multiple_graphs(self):
        pos_trans = _positions_for_dihedral(jnp.pi)
        pos_cis = _positions_for_dihedral(0.0)
        positions = jnp.concatenate([pos_trans, pos_cis])
        species = ["A"] * 8
        system_ids = [0, 0, 0, 0, 1, 1, 1, 1]
        particles = make_particles(positions, species, system_ids)
        systems = make_systems(jnp.eye(3)[None].repeat(2, axis=0) * 20.0)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]])),
            shifts=jnp.zeros((2, 3, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        result = dihedral_energy(
            GraphPotentialInput(graph=graph, parameters=self.params)
        )
        assert result.data.data.shape == (2,)
        assert jnp.isclose(result.data.data[0], 0.0, atol=1e-5)
        assert jnp.isclose(result.data.data[1], 2.0, atol=1e-5)
