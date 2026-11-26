# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the two-body D3(BJ) mathematical kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax import Array

from kups.core.cell import Cell, OrthogonalFrame, VacuumCell
from kups.core.constants import BOHR
from kups.core.data import Index, Table
from kups.core.neighborlist import Edges
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.common.graph import GraphPotentialInput, HyperGraph
from kups.potential.dispersion.d3 import (
    D3BJParameters,
    D3Parameters,
    D3ReferenceData,
    _reference_weights,
    d3_bj_edge_energy,
    d3_c6_coefficients,
    d3_c8_coefficients,
    d3_coordination_numbers,
    d3_energy,
)


@dataclass
class _Particles:
    positions: Array
    atomic_numbers: Array
    system: Index[SystemId]


@dataclass
class _Systems:
    cell: Cell


def _table(*values: float) -> Table[SystemId, Array]:
    return Table(
        tuple(SystemId(i) for i in range(len(values))),
        jnp.asarray(values),
    )


def _argon_reference() -> D3ReferenceData:
    """Small direct-Z table containing only the published argon row."""
    n_elements = 19
    return D3ReferenceData(
        covalent_radii=jnp.zeros(n_elements).at[18].set(1.28),
        r4r2=jnp.zeros(n_elements).at[18].set(1.8239535940463758),
        reference_cn=jnp.zeros((n_elements, 1)),
        reference_c6=(
            jnp.zeros((n_elements, n_elements, 1, 1))
            .at[18, 18, 0, 0]
            .set(38.62784329646292)
        ),
        reference_mask=jnp.zeros((n_elements, 1), dtype=bool).at[18, 0].set(True),
    )


def _parameters(
    *cutoffs: float, cn_cutoffs: tuple[float, ...] | None = None
) -> D3Parameters:
    cn_cutoffs = cutoffs if cn_cutoffs is None else cn_cutoffs
    return D3Parameters(
        damping=D3BJParameters(
            s6=jnp.asarray(1.0),
            s8=jnp.asarray(0.7875),
            a1=jnp.asarray(0.4289),
            a2=jnp.asarray(4.4407 * BOHR),
        ),
        reference=_argon_reference(),
        cutoff=_table(*cutoffs),
        cn_cutoff=_table(*cn_cutoffs),
    )


def _graph(
    positions: Array,
    atomic_numbers: Array,
    system_indices: Array,
    edge_indices: Array,
) -> HyperGraph:
    num_systems = int(jnp.max(system_indices)) + 1
    system_keys = tuple(SystemId(i) for i in range(num_systems))
    particles = Table.arange(
        _Particles(
            positions=jnp.asarray(positions),
            atomic_numbers=jnp.asarray(atomic_numbers),
            system=Index(system_keys, jnp.asarray(system_indices)),
        ),
        label=ParticleId,
    )
    systems = Table.arange(
        _Systems(
            VacuumCell(OrthogonalFrame(jnp.full((num_systems, 3), 100.0, dtype=float)))
        ),
        label=SystemId,
    )
    edge_indices = jnp.asarray(edge_indices)
    edges = Edges(
        indices=Index(particles.keys, edge_indices),
        shifts=jnp.zeros((len(edge_indices), 1, 3)),
    )
    return HyperGraph(particles, systems, edges)


def _argon_dimer_graph(*, padded: bool = False) -> HyperGraph:
    edges = [[0, 1], [1, 0]]
    if padded:
        edges.extend([[2, 2]] * 32)
    return _graph(
        positions=jnp.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]),
        atomic_numbers=jnp.array([18, 18]),
        system_indices=jnp.array([0, 0]),
        edge_indices=jnp.asarray(edges),
    )


class TestD3Math:
    def test_coordination_numbers_reduce_directed_edges(self) -> None:
        coordination = d3_coordination_numbers(
            distances=jnp.ones(3),
            covalent_radii_pairs=jnp.full((3, 2), 0.5),
            central_atom_indices=jnp.array([0, 1, 2]),
            num_particles=2,
            edge_mask=jnp.array([True, True, False]),
        )
        npt.assert_allclose(coordination, [0.5, 0.5], rtol=1e-12)

    def test_c6_contracts_endpoint_weights_with_the_pair_matrix(self) -> None:
        weights = jnp.array([[[0.25, 0.75], [0.5, 0.5]]])
        reference_c6 = jnp.array([[[2.0, 4.0], [6.0, 8.0]]])

        actual = d3_c6_coefficients(weights, reference_c6)

        expected = (
            np.asarray(weights[0, 0])
            @ np.asarray(reference_c6[0])
            @ np.asarray(weights[0, 1])
        )
        npt.assert_allclose(actual, [expected], rtol=1e-12)

    def test_reference_weights_are_normalized_per_node(self) -> None:
        actual = _reference_weights(
            coordination_numbers=jnp.array([0.25]),
            reference_cn=jnp.array([[0.0, 1.0]]),
            reference_mask=jnp.ones((1, 2), dtype=bool),
        )
        exponent = -4.0 * (0.25 - np.array([0.0, 1.0])) ** 2
        expected = np.exp(exponent - exponent.max())
        expected /= expected.sum()
        npt.assert_allclose(actual, [expected], rtol=1e-12)

    def test_reference_weights_are_stable_far_from_references(self) -> None:
        actual = _reference_weights(
            coordination_numbers=jnp.array([100.0]),
            reference_cn=jnp.array([[0.0, 1.0]]),
            reference_mask=jnp.ones((1, 2), dtype=bool),
        )
        npt.assert_allclose(actual, [[0.0, 1.0]], atol=1e-12)
        assert bool(jnp.isfinite(actual).all())

    def test_reference_weights_vanish_on_a_fully_masked_row(self) -> None:
        actual = _reference_weights(
            coordination_numbers=jnp.array([0.0]),
            reference_cn=jnp.zeros((1, 2)),
            reference_mask=jnp.zeros((1, 2), dtype=bool),
        )
        npt.assert_array_equal(actual, [[0.0, 0.0]])

    def test_c8_coefficients(self) -> None:
        c6 = jnp.array([2.0])
        r4r2_pairs = jnp.array([[1.0, 4.0]])

        actual = d3_c8_coefficients(c6, r4r2_pairs)

        # C8 = 3 * Q_i * Q_j * C6
        npt.assert_allclose(actual, [3.0 * 1.0 * 4.0 * 2.0], rtol=1e-12)

    def test_bj_edge_energy(self) -> None:
        # C6 and C8 are supplied directly, so this exercises the damped pair
        # expression alone and not the C6 -> C8 promotion.
        c6 = jnp.array([2.0])
        c8 = jnp.array([24.0])
        r4r2_pairs = jnp.array([[1.0, 4.0]])
        damping = D3BJParameters(
            s6=jnp.asarray(1.0),
            s8=jnp.asarray(0.5),
            a1=jnp.asarray(0.4),
            a2=jnp.asarray(2.0),
        )

        actual = d3_bj_edge_energy(jnp.array([3.0]), c6, c8, r4r2_pairs, damping)

        r0 = 0.4 * np.sqrt(3.0 * 1.0 * 4.0) + 2.0
        expected = -(2.0 / (3.0**6 + r0**6) + 0.5 * 24.0 / (3.0**8 + r0**8))
        npt.assert_allclose(actual, [expected], rtol=1e-12)


class TestD3Energy:
    def test_energy_uses_node_coordination_for_edge_c6(self) -> None:
        reference_c6 = jnp.array([[2.0, 4.0], [4.0, 8.0]])
        reference = D3ReferenceData(
            covalent_radii=jnp.array([0.0, 0.5]),
            r4r2=jnp.array([0.0, 1.5]),
            reference_cn=jnp.array([[0.0, 0.0], [0.0, 1.0]]),
            reference_c6=(jnp.zeros((2, 2, 2, 2)).at[1, 1].set(reference_c6)),
            reference_mask=jnp.array([[False, False], [True, True]]),
        )
        damping = D3BJParameters(
            s6=jnp.asarray(1.0),
            s8=jnp.asarray(0.5),
            a1=jnp.asarray(0.4),
            a2=jnp.asarray(2.0),
        )
        parameters = D3Parameters(damping, reference, _table(10.0), _table(10.0))
        positions = jnp.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [3.5, 0.0, 0.0]])
        graph = _graph(
            positions=positions,
            atomic_numbers=jnp.ones(3, dtype=int),
            system_indices=jnp.zeros(3, dtype=int),
            edge_indices=jnp.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]),
        )

        actual = float(d3_energy(GraphPotentialInput(parameters, graph)).data.data[0])

        positions_np = np.asarray(positions)
        coordination = np.zeros(3)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                distance = np.linalg.norm(positions_np[j] - positions_np[i])
                coordination[i] += 1.0 / (1.0 + np.exp(-16.0 * (1.0 / distance - 1.0)))
        weights = np.exp(-4.0 * (coordination[:, None] - np.array([0.0, 1.0])) ** 2)
        weights /= weights.sum(axis=-1, keepdims=True)
        expected = 0.0
        ratio = 3.0 * 1.5**2
        damping_length = 0.4 * np.sqrt(ratio) + 2.0
        for i in range(3):
            for j in range(i + 1, 3):
                distance = np.linalg.norm(positions_np[j] - positions_np[i])
                c6 = weights[i] @ np.asarray(reference_c6) @ weights[j]
                c8 = ratio * c6
                expected -= c6 / (distance**6 + damping_length**6)
                expected -= 0.5 * c8 / (distance**8 + damping_length**8)
        npt.assert_allclose(actual, expected, rtol=1e-12)

    def test_argon_dimer_matches_simple_dftd3(self) -> None:
        graph = _argon_dimer_graph()
        inp = GraphPotentialInput(_parameters(12.0), graph)
        actual = d3_energy(inp).data.data
        npt.assert_allclose(actual, [-0.010745634079163811], rtol=1e-12)

        def energy(positions: Array) -> Array:
            particles = Table(
                graph.particles.keys,
                _Particles(
                    positions,
                    graph.particles.data.atomic_numbers,
                    graph.particles.data.system,
                ),
            )
            moved = HyperGraph(particles, graph.systems, graph.edges)
            return d3_energy(GraphPotentialInput(inp.parameters, moved)).data.data.sum()

        gradient = jax.grad(energy)(graph.particles.data.positions)
        expected_gradient = jnp.array(
            [[-0.01031742754289883, 0.0, 0.0], [0.01031742754289883, 0.0, 0.0]]
        )
        npt.assert_allclose(gradient, expected_gradient, atol=1e-12)

    def test_jit_and_padded_edges_remain_finite(self) -> None:
        graph = _argon_dimer_graph(padded=True)
        inp = GraphPotentialInput(_parameters(12.0), graph)
        actual = jax.jit(d3_energy)(inp).data.data
        npt.assert_allclose(actual, [-0.010745634079163811], rtol=1e-12)

        def energy(positions: Array) -> Array:
            particles = Table(
                graph.particles.keys,
                _Particles(
                    positions,
                    graph.particles.data.atomic_numbers,
                    graph.particles.data.system,
                ),
            )
            moved = HyperGraph(particles, graph.systems, graph.edges)
            return d3_energy(GraphPotentialInput(inp.parameters, moved)).data.data.sum()

        gradient = jax.grad(energy)(graph.particles.data.positions)
        assert bool(jnp.isfinite(gradient).all())

    def test_energy_cutoff_is_strict_and_resolved_per_system(self) -> None:
        """System 1 sits at exactly ``r == cutoff`` and so contributes nothing.

        D3 compares ``distances < cutoff`` strictly, matching every other kUPS
        pair potential. System 0 keeps the same pair inside a wider cutoff.
        """
        graph = _graph(
            positions=jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.8, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.8, 0.0, 0.0],
                ]
            ),
            atomic_numbers=jnp.array([18, 18, 18, 18]),
            system_indices=jnp.array([0, 0, 1, 1]),
            edge_indices=jnp.array([[0, 1], [1, 0], [2, 3], [3, 2]]),
        )
        actual = d3_energy(GraphPotentialInput(_parameters(12.0, 3.8), graph)).data.data
        npt.assert_allclose(actual, [-0.010745634079163811, 0.0], rtol=1e-12)

    def test_cn_cutoff_bounds_the_environment_not_the_energy(self) -> None:
        """An edge outside ``cn_cutoff`` but inside ``cutoff`` still has energy.

        ``cn_cutoff`` bounds the coordination-number sum that selects C6;
        ``cutoff`` bounds the dispersion sum. An edge between the two must be
        paid for in the energy while leaving the environment untouched.
        """
        reference_c6 = jnp.array([[2.0, 4.0], [4.0, 8.0]])
        reference = D3ReferenceData(
            covalent_radii=jnp.array([0.0, 0.5]),
            r4r2=jnp.array([0.0, 1.5]),
            reference_cn=jnp.array([[0.0, 0.0], [0.0, 1.0]]),
            reference_c6=(jnp.zeros((2, 2, 2, 2)).at[1, 1].set(reference_c6)),
            reference_mask=jnp.array([[False, False], [True, True]]),
        )
        damping = D3BJParameters(
            s6=jnp.asarray(1.0),
            s8=jnp.asarray(0.5),
            a1=jnp.asarray(0.4),
            a2=jnp.asarray(2.0),
        )
        distance = 1.0
        graph = _graph(
            positions=jnp.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
            atomic_numbers=jnp.ones(2, dtype=int),
            system_indices=jnp.zeros(2, dtype=int),
            edge_indices=jnp.array([[0, 1], [1, 0]]),
        )

        ratio = 3.0 * 1.5**2
        damping_length = 0.4 * np.sqrt(ratio) + 2.0

        def expected(coordination: float) -> float:
            weights = np.exp(-4.0 * (coordination - np.array([0.0, 1.0])) ** 2)
            weights /= weights.sum()
            c6 = float(weights @ np.asarray(reference_c6) @ weights)
            c8 = ratio * c6
            return -(
                c6 / (distance**6 + damping_length**6)
                + 0.5 * c8 / (distance**8 + damping_length**8)
            )

        # cn_cutoff below the separation: the pair is energetic but invisible to
        # the coordination number, so both atoms keep CN == 0.
        outside_cn = D3Parameters(damping, reference, _table(10.0), _table(0.5))
        without_environment = float(
            d3_energy(GraphPotentialInput(outside_cn, graph)).data.data[0]
        )
        npt.assert_allclose(without_environment, expected(0.0), rtol=1e-12)

        # widening cn_cutoff past the pair gives each atom
        # CN = sigmoid(16 * (1.0 / 1.0 - 1)) = 0.5, which selects a different C6
        inside_cn = D3Parameters(damping, reference, _table(10.0), _table(10.0))
        with_environment = float(
            d3_energy(GraphPotentialInput(inside_cn, graph)).data.data[0]
        )
        npt.assert_allclose(with_environment, expected(0.5), rtol=1e-12)

        assert with_environment < without_environment

    def test_isolated_atom_has_zero_energy(self) -> None:
        graph = _graph(
            positions=jnp.zeros((1, 3)),
            atomic_numbers=jnp.array([18]),
            system_indices=jnp.array([0]),
            edge_indices=jnp.empty((0, 2), dtype=int),
        )
        actual = d3_energy(GraphPotentialInput(_parameters(12.0), graph)).data.data
        npt.assert_array_equal(actual, [0.0])
