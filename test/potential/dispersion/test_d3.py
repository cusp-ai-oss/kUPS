# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Structural and numerical properties of the D3(BJ) potential."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_AND_CELL
from kups.potential.common.graph import GRAPH_GEOMETRY, GraphPotentialInput
from kups.potential.dispersion import available_functionals
from kups.potential.dispersion.d3 import (
    D3_DEFAULT_CUTOFF,
    K_WEIGHT,
    D3Parameters,
    d3_c6_coefficients,
    d3_coordination_numbers,
    d3_edge_energy,
    d3_energy,
    validate_atomic_numbers,
)
from test.potential.dispersion._graphs import build_graph
from test.potential.dispersion._systems import SYSTEMS, System

_CUTOFF = 10.0
_PARAMETERS = D3Parameters.from_functional("pbe", cutoff=_CUTOFF)
_POSITION_LENS = GRAPH_GEOMETRY.nest(POSITIONS_AND_CELL)
_FRECHET_LENS = GRAPH_GEOMETRY.nest(FRECHET_FILTER)


def _energy(systems: list[System], parameters: D3Parameters | None = None, **kwargs):
    parameters = parameters or _PARAMETERS
    graph = build_graph(systems, float(parameters.cutoff.data[0]), **kwargs)
    return d3_energy(GraphPotentialInput(parameters, graph)).data.data


def _energy_of_positions(system: System, positions, parameters=None) -> float:
    replaced = System(system.numbers, np.asarray(positions), system.cell, system.pbc)
    return float(_energy([replaced], parameters)[0])


class TestParameters:
    def test_from_functional_and_units(self) -> None:
        """Merged: lookup, name folding, a2 unit conversion, unknown functional."""
        params = D3Parameters.from_functional("PBE")
        npt.assert_allclose(float(params.s6), 1.0)
        npt.assert_allclose(float(params.s8), 0.7875)
        npt.assert_allclose(float(params.a1), 0.4289)
        # a2 is published in Bohr and stored in Angstrom
        npt.assert_allclose(float(params.a2), 4.4407 * 0.529177210544, rtol=1e-9)
        assert float(D3Parameters.from_functional("pbe-0").s8) == 1.2177
        with pytest.raises(KeyError, match="No tabulated"):
            D3Parameters.from_functional("not-a-functional")
        assert "r2scan" in available_functionals()

    def test_cn_cutoff_is_capped_at_the_pair_cutoff(self) -> None:
        """A single neighbor list serves both, so cn_cutoff may never exceed cutoff."""
        with pytest.warns(UserWarning):
            params = D3Parameters.from_damping(
                s8=1.0, a1=0.4, a2=4.4, cutoff=8.0, cn_cutoff=25.0
            )
        assert float(params.cn_cutoff.data[0]) == 8.0
        params = D3Parameters.from_damping(
            s8=1.0, a1=0.4, a2=4.4, cutoff=30.0, cn_cutoff=21.0
        )
        assert float(params.cn_cutoff.data[0]) == 21.0

    def test_per_system_cutoffs(self) -> None:
        params = D3Parameters.from_damping(
            s8=1.0, a1=0.4, a2=4.4, cutoff=jnp.array([8.0, 12.0])
        )
        npt.assert_allclose(np.asarray(params.cutoff.data), [8.0, 12.0])
        npt.assert_allclose(np.asarray(params.cn_cutoff.data), [8.0, 12.0])

    def test_cutoffs_are_broadcast_symmetrically(self) -> None:
        """A scalar on either side must broadcast against a per-system array."""
        params = D3Parameters.from_damping(
            s8=1.0, a1=0.4, a2=4.4, cutoff=9.0, cn_cutoff=jnp.array([8.0, 7.0])
        )
        npt.assert_allclose(np.asarray(params.cutoff.data), [9.0, 9.0])
        npt.assert_allclose(np.asarray(params.cn_cutoff.data), [8.0, 7.0])

    @pytest.mark.parametrize(
        "cutoff", [-9.0, 0.0, float("nan"), float("inf"), jnp.array([])]
    )
    def test_invalid_cutoffs_are_rejected(self, cutoff) -> None:
        """A negative cutoff would otherwise be squared and silently accepted."""
        with pytest.raises(ValueError):
            D3Parameters.from_damping(s8=1.0, a1=0.4, a2=4.4, cutoff=cutoff)

    def test_explicit_cn_cutoff_above_cutoff_warns(self) -> None:
        """Silently changing a value the caller asked for would be surprising."""
        with pytest.warns(UserWarning, match="cn_cutoff exceeds cutoff"):
            params = D3Parameters.from_damping(
                s8=1.0, a1=0.4, a2=4.4, cutoff=8.0, cn_cutoff=25.0
            )
        assert float(params.cn_cutoff.data[0]) == 8.0
        # the default being clamped is routine, so it must stay silent
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            D3Parameters.from_damping(s8=1.0, a1=0.4, a2=4.4, cutoff=8.0)

    def test_validate_atomic_numbers(self) -> None:
        validate_atomic_numbers(jnp.array([1, 6, 103]))
        with pytest.raises(ValueError, match="1..103"):
            validate_atomic_numbers(jnp.array([1, 104]))
        with pytest.raises(ValueError, match="1..103"):
            validate_atomic_numbers(jnp.array([0, 6]))


class TestInvariances:
    def test_isolated_atom_has_no_dispersion(self) -> None:
        """A single atom has no pairs, so both the energy and its CN vanish."""
        graph = build_graph([SYSTEMS["ar_atom"]], _CUTOFF)
        inp = GraphPotentialInput(_PARAMETERS, graph)
        assert float(d3_energy(inp).data.data[0]) == 0.0
        npt.assert_array_equal(np.asarray(d3_coordination_numbers(inp)), [0.0])

    @pytest.mark.parametrize("name", ["water_dimer", "si_diamond"])
    def test_translation_invariance(self, name: str) -> None:
        system = SYSTEMS[name]
        shift = np.array([0.37, -1.24, 2.03])
        base = _energy_of_positions(system, system.positions)
        moved = _energy_of_positions(system, system.positions + shift)
        npt.assert_allclose(moved, base, rtol=1e-12)

    def test_rotation_invariance(self) -> None:
        system = SYSTEMS["benzene"]
        angle = 0.7
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        base = _energy_of_positions(system, system.positions)
        rotated = _energy_of_positions(system, system.positions @ rotation.T)
        npt.assert_allclose(rotated, base, rtol=1e-10)

    def test_permutation_invariance(self) -> None:
        """Reordering atoms must not change the energy or the multiset of CNs."""
        system = SYSTEMS["water_dimer"]
        order = np.array([3, 0, 5, 2, 1, 4])
        permuted = System(
            system.numbers[order], system.positions[order], system.cell, system.pbc
        )
        base = float(_energy([system])[0])
        npt.assert_allclose(float(_energy([permuted])[0]), base, rtol=1e-12)

        cn_base = np.asarray(
            d3_coordination_numbers(
                GraphPotentialInput(_PARAMETERS, build_graph([system], _CUTOFF))
            )
        )
        cn_perm = np.asarray(
            d3_coordination_numbers(
                GraphPotentialInput(_PARAMETERS, build_graph([permuted], _CUTOFF))
            )
        )
        npt.assert_allclose(np.sort(cn_base), np.sort(cn_perm), rtol=1e-12)

    def test_supercell_consistency(self) -> None:
        """E(2x2x2 supercell) == 8 x E(cell): the strongest periodic-image check."""
        from ase.build import bulk

        cutoff = 8.0
        parameters = D3Parameters.from_functional("pbe", cutoff=cutoff)
        cell = bulk("Si", "diamond", a=5.43, cubic=True)
        supercell = cell * (2, 2, 2)

        def system_of(atoms) -> System:
            return System(
                np.array(atoms.get_atomic_numbers()),
                atoms.positions.copy(),
                atoms.cell.array.copy(),
                (True, True, True),
            )

        single = float(_energy([system_of(cell)], parameters, edge_capacity=1 << 14)[0])
        eight = float(
            _energy([system_of(supercell)], parameters, edge_capacity=1 << 14)[0]
        )
        npt.assert_allclose(eight, 8.0 * single, rtol=1e-8)


class TestGradients:
    @pytest.mark.parametrize("name", ["water_dimer", "co2"])
    def test_finite_difference_position_gradient(self, name: str) -> None:
        """Central differences at h = 1e-5 Å; float64 keeps cancellation at ~1e-10."""
        system = SYSTEMS[name]
        graph = build_graph([system], _CUTOFF)
        inp = GraphPotentialInput(_PARAMETERS, graph)
        analytic = np.asarray(
            jax.grad(lambda d: d3_energy(_POSITION_LENS.set(inp, d)).data.data.sum())(
                _POSITION_LENS.get(inp)
            ).positions.data
        )

        positions = np.asarray(graph.particles.data.positions)
        step = 1e-5
        numeric = np.zeros_like(positions)
        for atom in range(positions.shape[0]):
            for axis in range(3):
                plus = positions.copy()
                plus[atom, axis] += step
                minus = positions.copy()
                minus[atom, axis] -= step
                numeric[atom, axis] = (
                    _energy_of_positions(system, plus)
                    - _energy_of_positions(system, minus)
                ) / (2 * step)
        npt.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)

    def test_finite_difference_cell_gradient(self) -> None:
        """Total dE/d(strain) from FRECHET_FILTER against an affine deformation.

        Under ``h -> (I + e) h`` with atoms riding at fixed fractional coordinates,
        ``dE/de`` is exactly the virial that the cell filter's autodiff produces.
        """
        system = SYSTEMS["si_diamond"]
        cutoff = 8.0
        parameters = D3Parameters.from_functional("pbe", cutoff=cutoff)

        def energy_under_strain(strain: np.ndarray) -> float:
            deformation = np.eye(3) + strain
            cell = system.cell @ deformation.T
            fractional = system.positions @ np.linalg.inv(system.cell)
            deformed = System(system.numbers, fractional @ cell, cell, system.pbc)
            return float(_energy([deformed], parameters, edge_capacity=1 << 14)[0])

        graph = build_graph([system], cutoff, edge_capacity=1 << 14)
        inp = GraphPotentialInput(parameters, graph)
        dofs = _FRECHET_LENS.get(inp)
        gradient = jax.grad(
            lambda d: d3_energy(_FRECHET_LENS.set(inp, d)).data.data.sum()
        )(dofs)
        vectors = graph.systems.data.cell.frame.vectors[0]
        cell_gradient = np.asarray(
            graph.systems.data.cell.frame.vectors_gradient(gradient.cell.data.frame)[0]
        )
        analytic = np.asarray(vectors).T @ cell_gradient

        step = 1e-6
        numeric = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                plus = np.zeros((3, 3))
                plus[i, j] = step
                numeric[i, j] = (
                    energy_under_strain(plus) - energy_under_strain(-plus)
                ) / (2 * step)
        # compare the symmetric part: only that is physically determined
        npt.assert_allclose(
            (analytic + analytic.T) / 2, (numeric + numeric.T) / 2, atol=1e-6
        )


class TestNumericalSafety:
    @pytest.mark.parametrize("atomic_number", [104, 0, -1])
    def test_energy_rejects_unsupported_atomic_numbers(
        self, atomic_number: int
    ) -> None:
        """The public energy kernel must reject invalid table indices eagerly."""
        system = System(
            np.array([atomic_number, 6]),
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            None,
            (False, False, False),
        )
        lo, hi = min(atomic_number, 6), max(atomic_number, 6)
        with pytest.raises(
            ValueError,
            match=rf"D3 covers atomic numbers 1\.\.103; got {lo}\.\.{hi}\.",
        ):
            _energy([system])

    def test_padded_edges_produce_finite_energy_and_gradients(self) -> None:
        """Padded rows collapse both endpoints onto one point, so r would be 0.

        ``segment_sum(mode="drop")`` protects the energy, but a zero cotangent
        times an infinite partial derivative would still yield NaN gradients, so
        the kernel must substitute a safe distance before dividing.
        """
        graph = build_graph([SYSTEMS["water"]], _CUTOFF, edge_capacity=1 << 14)
        n_valid = int(graph.edges.indices.valid_mask.all(axis=-1).sum())
        assert n_valid == 6
        assert len(graph.edges) > 1000, "test is meaningless without heavy padding"

        inp = GraphPotentialInput(_PARAMETERS, graph)
        per_edge = np.asarray(d3_edge_energy(inp))
        assert np.isfinite(per_edge).all()
        assert np.count_nonzero(per_edge) == n_valid

        assert np.isfinite(np.asarray(d3_coordination_numbers(inp))).all()
        assert np.isfinite(
            np.asarray(d3_c6_coefficients(inp, d3_coordination_numbers(inp)))
        ).all()

        gradient = jax.grad(
            lambda d: d3_energy(_POSITION_LENS.set(inp, d)).data.data.sum()
        )(_POSITION_LENS.get(inp))
        assert np.isfinite(np.asarray(gradient.positions.data)).all()

    def test_extreme_coordination_number_weights_stay_finite(self) -> None:
        """Weights must not underflow to a zero normalization for unusual CNs.

        A carbon squeezed among many neighbours reaches a coordination number far
        above every tabulated reference, where ``exp(-k3 dCN^2)`` underflows to
        exactly zero. Without the max-subtraction in ``_reference_weights`` the
        normalization would then be zero, every weight would collapse to zero,
        and ``C6`` would come out zero rather than wrong-but-finite.

        The premise is asserted rather than assumed. It is easy to write a
        "crowded" system that never actually reaches the underflow threshold --
        in float64 that needs ``dCN`` above ~13.3, which is a much higher
        coordination number than it looks -- and such a test passes while
        exercising nothing.
        """
        reference = _PARAMETERS.reference
        carbon_refs = np.asarray(reference.reference_cn[6])[
            np.asarray(reference.reference_mask[6])
        ]

        rng = np.random.default_rng(0)
        positions = rng.normal(scale=0.75, size=(24, 3))
        positions[0] = 0.0
        crowded = System(np.full(24, 6), positions, None, (False, False, False))
        graph = build_graph([crowded], _CUTOFF)
        inp = GraphPotentialInput(_PARAMETERS, graph)

        cn = np.asarray(d3_coordination_numbers(inp))
        # the nearest reference is the one that decides whether anything survives
        gap = cn.max() - carbon_refs.max()
        unguarded = np.exp(-K_WEIGHT * gap**2)
        assert unguarded == 0.0, (
            f"premise not met: CN {cn.max():.2f} leaves exp(-k3 dCN^2) = "
            f"{unguarded:.2e}, which does not underflow, so the guard this test "
            "exists for is never reached"
        )

        c6 = np.asarray(d3_c6_coefficients(inp, jnp.asarray(cn)))
        assert np.isfinite(c6).all()
        # zero here is the failure mode: it means the normalization collapsed
        assert (c6[: int(graph.edges.indices.valid_mask.all(-1).sum())] > 0).all()
        assert np.isfinite(float(d3_energy(inp).data.data[0]))


class TestCutoffBehaviour:
    def test_energy_converges_monotonically_with_cutoff(self) -> None:
        """Dispersion is attractive, so extending the cutoff can only lower E."""
        system = SYSTEMS["si_diamond"]
        energies = []
        for cutoff in (6.0, 8.0, 10.0, 12.0):
            parameters = D3Parameters.from_functional("pbe", cutoff=cutoff)
            energies.append(
                float(_energy([system], parameters, edge_capacity=1 << 15)[0])
            )
        assert all(b <= a for a, b in zip(energies, energies[1:])), energies
        # the tail is small by 12 Å: successive increments must be shrinking
        deltas = [abs(b - a) for a, b in zip(energies, energies[1:])]
        assert deltas[-1] < deltas[0]

    def test_default_cutoff_is_the_documented_value(self) -> None:
        assert D3_DEFAULT_CUTOFF == 15.0
        assert float(D3Parameters.from_functional("pbe").cutoff.data[0]) == 15.0

    def test_per_system_cutoffs_are_applied_per_system(self) -> None:
        """A per-system table must reach the system it belongs to, not its neighbour."""
        system = SYSTEMS["water_dimer"]
        graph = build_graph([system] * 3, 10.0)

        def energies(cutoff):
            parameters = D3Parameters.from_damping(
                s8=0.7875, a1=0.4289, a2=4.4407, cutoff=cutoff
            )
            return np.asarray(
                d3_energy(GraphPotentialInput(parameters, graph)).data.data
            )

        wide, narrow = energies(10.0), energies(2.5)
        assert wide[0] < narrow[0] < 0.0, "the two cutoffs must be distinguishable"
        mixed = energies(jnp.array([10.0, 2.5, 10.0]))
        npt.assert_allclose(mixed, [wide[0], narrow[0], wide[0]], rtol=1e-12)
        # a scalar still stands for every system
        npt.assert_allclose(energies(10.0), [wide[0]] * 3, rtol=1e-12)

    def test_cutoff_table_shorter_than_the_batch_is_rejected(self) -> None:
        """Gathering the raw array would clamp out of bounds and answer silently.

        JAX resolves an out-of-range index to the last entry, so a two-entry
        table evaluated over three systems would hand the third system the
        second's cutoff and return a plausible energy. Only a table sized 1 or
        ``n_systems`` is meaningful.
        """
        graph = build_graph([SYSTEMS["water_dimer"]] * 3, 10.0)
        parameters = D3Parameters.from_damping(
            s8=0.7875, a1=0.4289, a2=4.4407, cutoff=jnp.array([10.0, 2.5])
        )
        with pytest.raises(AssertionError, match="Cannot broadcast Table sizes"):
            d3_energy(GraphPotentialInput(parameters, graph))
