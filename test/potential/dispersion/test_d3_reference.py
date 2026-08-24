# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Validation of the D3(BJ) kernels against simple-dftd3 and the NumPy oracle."""

from __future__ import annotations

import jax
import numpy as np
import numpy.testing as npt
import pytest

from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.core.constants import BOHR
from kups.potential.common.graph import GRAPH_GEOMETRY, GraphPotentialInput
from kups.potential.dispersion.d3 import (
    D3_REFERENCE_CN_CUTOFF,
    D3_REFERENCE_CUTOFF,
    D3Parameters,
    d3_coordination_numbers,
    d3_energy,
)
from test.potential.dispersion._graphs import build_graph
from test.potential.dispersion._oracle import (
    coordination_numbers as oracle_cn,
)
from test.potential.dispersion._oracle import (
    dispersion_energy as oracle_energy,
)
from test.potential.dispersion._oracle import reference_tables
from test.potential.dispersion._reference_values import REFERENCE
from test.potential.dispersion._systems import (
    MOLECULAR,
    PBE_D3BJ_BOHR,
    PERIODIC,
    SYSTEMS,
    TRANSITION_METAL,
)

# Reference values were generated at simple-dftd3's own cutoffs; matching them is
# what makes the comparison exact rather than truncation-limited.
_PARAMETERS = D3Parameters.from_damping(
    **PBE_D3BJ_BOHR, cutoff=D3_REFERENCE_CUTOFF, cn_cutoff=D3_REFERENCE_CN_CUTOFF
)
_GRADIENT_LENS = GRAPH_GEOMETRY.nest(POSITIONS_AND_CELL)

# Molecules need only enough cutoff to span the molecule; periodic cells need the
# full reference cutoff, which is what makes those graphs large.
_MOLECULE_CUTOFF = 12.0
_PERIODIC_EDGE_CAPACITY = 1 << 17

# Tolerances sit a small factor above the measured worst case across all seventeen
# systems, so a real regression fails rather than hiding under slack. Agreement is
# limited by float64 accumulation over ~5e4 edges and by the order the two codes
# traverse them, not by the physics -- and not by the committed literals, which
# `repr` round-trips exactly. Measured worst cases, all on
# the triclinic si_sheared cell: energy 4.1e-10 relative, gradient 1.1e-9 eV/Å,
# stress 1.2e-10 eV/Å³.
_ENERGY_RTOL = 1e-9
_GRADIENT_ATOL = 5e-9  # eV/Å; several systems have gradients below 1e-5 eV/Å
_STRESS_ATOL = 5e-10  # eV/Å³


def _graph(name: str):
    system = SYSTEMS[name]
    if system.cell is None:
        return build_graph([system], _MOLECULE_CUTOFF)
    return build_graph(
        [system], float(D3_REFERENCE_CUTOFF), edge_capacity=_PERIODIC_EDGE_CAPACITY
    )


def _energy_and_gradients(name: str):
    inp = GraphPotentialInput(_PARAMETERS, _graph(name))

    def total(dofs):
        return d3_energy(_GRADIENT_LENS.set(inp, dofs)).data.data.sum()

    energy, gradient = jax.value_and_grad(total)(_GRADIENT_LENS.get(inp))
    return float(energy), gradient, inp


class TestAgainstSimpleDftd3:
    """Energies, gradients and virials against committed simple-dftd3 values."""

    @pytest.mark.parametrize("name", MOLECULAR + PERIODIC + TRANSITION_METAL)
    def test_energy(self, name: str) -> None:
        inp = GraphPotentialInput(_PARAMETERS, _graph(name))
        energy = float(d3_energy(inp).data.data[0])
        expected = REFERENCE[name]["energy"]
        if expected == 0.0:
            assert energy == 0.0
        else:
            npt.assert_allclose(energy, expected, rtol=_ENERGY_RTOL)

    @pytest.mark.parametrize("name", MOLECULAR + PERIODIC + TRANSITION_METAL)
    def test_position_gradient(self, name: str) -> None:
        """Autodiff through CN -> C6 -> damping reproduces the reference gradient.

        Periodic cells are included: they exercise the lattice-translation path,
        and the triclinic cell carries the largest residual of any system here.
        """
        _, gradient, _ = _energy_and_gradients(name)
        npt.assert_allclose(
            np.asarray(gradient.positions.data),
            np.asarray(REFERENCE[name]["gradient"]),
            atol=_GRADIENT_ATOL,
        )

    @pytest.mark.parametrize("name", PERIODIC + TRANSITION_METAL)
    def test_stress_matches_virial(self, name: str) -> None:
        """kUPS' stress is ``-virial / V``; the sign convention is opposite.

        Components touching a non-periodic axis are compared only for the
        periodic block: kUPS zeroes them by convention (``_periodic_mask`` in
        [kups.observables.stress][]) whereas simple-dftd3 has no per-axis
        periodicity and reports a value there. For the fully periodic cells the
        mask is all-True and the whole tensor is compared.
        """
        from kups.observables.stress import stress_via_virial_theorem
        from test.potential.dispersion._graphs import virial_tables

        _, gradient, inp = _energy_and_gradients(name)
        particles, systems = virial_tables(inp.graph, gradient)
        sigma = np.asarray(stress_via_virial_theorem(particles, systems).data[0])
        volume = float(inp.graph.systems.data.cell.volume[0])
        expected = -np.asarray(REFERENCE[name]["virial"]) / volume

        periodic = np.array(SYSTEMS[name].pbc)
        mask = periodic[:, None] & periodic[None, :]
        npt.assert_allclose(sigma[mask], expected[mask], atol=_STRESS_ATOL)
        # the open axis must be zeroed, not merely different
        npt.assert_array_equal(sigma[~mask], 0.0)


class TestAgainstOracle:
    """The vectorized kernels against the explicit-loop NumPy reference."""

    @classmethod
    def setup_class(cls) -> None:
        cls.tables = reference_tables()
        cls.damping = dict(PBE_D3BJ_BOHR)
        cls.damping["a2"] = cls.damping["a2"] * BOHR

    @pytest.mark.parametrize("name", ["water", "co2", "benzene"])
    def test_coordination_numbers(self, name: str) -> None:
        """Merged: CN values, and that they are chemically sensible."""
        system = SYSTEMS[name]
        graph = build_graph([system], _MOLECULE_CUTOFF)
        actual = np.asarray(
            d3_coordination_numbers(GraphPotentialInput(_PARAMETERS, graph))
        )
        expected = oracle_cn(
            system.numbers,
            np.asarray(graph.particles.data.positions),
            self.tables["covalent_radii"],
            cn_cutoff=float(_PARAMETERS.cn_cutoff.data[0]),
        )
        npt.assert_allclose(actual, expected, rtol=1e-12)
        if name == "water":
            # oxygen has two bonds, each hydrogen one
            npt.assert_allclose(actual, [1.98557106, 0.99364729, 0.99364729], rtol=1e-6)

    @pytest.mark.parametrize("name", ["ar2", "hf", "water_dimer", "benzene"])
    def test_energy(self, name: str) -> None:
        system = SYSTEMS[name]
        graph = build_graph([system], _MOLECULE_CUTOFF)
        actual = float(d3_energy(GraphPotentialInput(_PARAMETERS, graph)).data.data[0])
        expected = oracle_energy(
            system.numbers,
            np.asarray(graph.particles.data.positions),
            self.tables,
            cell=None,
            cutoff=_MOLECULE_CUTOFF,
            cn_cutoff=float(_PARAMETERS.cn_cutoff.data[0]),
            **self.damping,
        )
        npt.assert_allclose(actual, expected, rtol=1e-12)


class TestLiveSimpleDftd3:
    """Optional re-check against a live simple-dftd3, if the ``d3ref`` extra is installed.

    Skipped in CI, which installs no optional extras. The committed reference
    values in ``_reference_values.py`` are the mechanism that actually runs.
    """

    def test_energy_matches_live_reference(self) -> None:
        interface = pytest.importorskip("dftd3.interface")
        system = SYSTEMS["water_dimer"]
        model = interface.DispersionModel(
            numbers=system.numbers, positions=system.positions / BOHR
        )
        model.set_realspace_cutoff(
            float(D3_REFERENCE_CUTOFF / BOHR), 0.0, float(D3_REFERENCE_CN_CUTOFF / BOHR)
        )
        param = interface.RationalDampingParam(**PBE_D3BJ_BOHR, s9=0.0, alp=14.0)
        from kups.core.constants import HARTREE

        expected = model.get_dispersion(param, grad=False)["energy"] * HARTREE
        graph = build_graph([system], _MOLECULE_CUTOFF)
        actual = float(d3_energy(GraphPotentialInput(_PARAMETERS, graph)).data.data[0])
        npt.assert_allclose(actual, expected, rtol=_ENERGY_RTOL)
