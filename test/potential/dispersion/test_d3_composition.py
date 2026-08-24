# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""D3 through the state-binding adapter: composition with other potentials, batching.

This is the use case from issue #140 — an MLIP or classical force field plus a
dispersion correction — expressed with the abstractions a user would actually
reach for.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.dispersion.d3 import make_d3_from_state
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.core.capacity import FixedCapacity
from kups.core.data import Index, Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    AllDenseNearestNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.potential import sum_potentials
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.potential.dispersion.d3 import D3Parameters
from test.potential.dispersion._graphs import D3Points, D3Systems, build_graph
from test.potential.dispersion._systems import SYSTEMS, System

_CUTOFF = 9.0
_D3 = D3Parameters.from_functional("pbe", cutoff=_CUTOFF)


@dataclass
class _State:
    """The minimal state the D3 and LJ adapters require."""

    particles: Table[ParticleId, D3Points]
    systems: Table[SystemId, D3Systems]
    neighborlist_params: UniversalNeighborlistParameters


@dataclass
class _LJPoints(D3Points):
    """Particles that additionally carry the species labels LJ looks up."""

    labels: Index[str]


def _state_for(systems: list[System], *, labels: bool = False) -> _State:
    """Reuse the graph builder's particle/system construction, then wrap it."""
    graph = build_graph(systems, _CUTOFF)
    data = graph.particles.data
    if labels:
        from ase.data import chemical_symbols

        symbols = [chemical_symbols[int(z)] for z in np.asarray(data.atomic_numbers)]
        data = _LJPoints(
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            system=data.system,
            inclusion=data.inclusion,
            exclusion=data.exclusion,
            labels=Index.new(symbols),
        )
    return _State(
        particles=Table.arange(data, label=ParticleId),
        systems=graph.systems,
        neighborlist_params=UniversalNeighborlistParameters(
            avg_edges=1024, avg_candidates=1024, avg_image_candidates=1024, cells=64
        ),
    )


def _dense_factory(state, cutoffs):
    """A deterministic neighbor list, so composition is compared like for like."""
    capacity = FixedCapacity(4096)
    return AllDenseNearestNeighborList(
        avg_edges=capacity, avg_image_candidates=capacity, cutoffs=cutoffs
    )


_LENS = identity_lens(_State)


def _d3_potential(**kwargs):
    return make_d3_from_state(
        _LENS, parameters=_D3, neighborlist_factory=_dense_factory, **kwargs
    )


class TestAdapter:
    def test_energy_matches_the_kernel_and_gradients_are_finite(self) -> None:
        """Merged: adapter wiring, energy parity with the raw kernel, finite forces."""
        from kups.potential.common.graph import GraphPotentialInput
        from kups.potential.dispersion.d3 import d3_energy

        system = SYSTEMS["water_dimer"]
        state = _state_for([system])
        out = _d3_potential(gradient=POSITIONS_AND_CELL)(state).data

        direct = float(
            d3_energy(
                GraphPotentialInput(_D3, build_graph([system], _CUTOFF))
            ).data.data[0]
        )
        npt.assert_allclose(float(out.total_energies.data[0]), direct, rtol=1e-12)
        assert float(out.total_energies.data[0]) < 0.0
        assert np.isfinite(np.asarray(out.gradients.positions.data)).all()

    def test_without_gradient_lens_no_gradients_are_computed(self) -> None:
        from kups.core.potential import EmptyType

        out = _d3_potential()(_state_for([SYSTEMS["water"]])).data
        # EmptyType is a pytree dataclass, so it is rebuilt during tracing and
        # compares by type rather than identity
        assert isinstance(out.gradients, EmptyType)
        assert isinstance(out.hessians, EmptyType)


class TestComposition:
    """``E(a + d3) == E(a) + E(d3)`` — exactly what issue #140 asks for."""

    @classmethod
    def setup_class(cls) -> None:
        cls.state = _state_for([SYSTEMS["water_dimer"]], labels=True)
        cls.lj_parameters = LennardJonesParameters.from_dict(
            cutoff=_CUTOFF,
            parameters={"O": (3.17, 0.0067), "H": (1.0, 0.0)},
            mixing_rule="lorentz_berthelot",
        )

    def _potentials(self, gradient):
        lens = identity_lens(_State)
        lj = make_lennard_jones_from_state(
            lens,
            parameters=self.lj_parameters,
            gradient=gradient,
            neighborlist_factory=_dense_factory,
        )
        d3 = make_d3_from_state(
            lens,
            parameters=_D3,
            gradient=gradient,
            neighborlist_factory=_dense_factory,
        )
        return lj, d3

    def test_energies_are_additive(self) -> None:
        lj, d3 = self._potentials(None)
        total = sum_potentials(lj, d3)
        e_lj = float(lj(self.state).data.total_energies.data[0])
        e_d3 = float(d3(self.state).data.total_energies.data[0])
        e_total = float(total(self.state).data.total_energies.data[0])
        npt.assert_allclose(e_total, e_lj + e_d3, rtol=1e-12)
        # both terms must actually contribute, or the test proves nothing
        assert abs(e_lj) > 1e-6 and abs(e_d3) > 1e-6

    def test_gradients_are_additive(self) -> None:
        """The summed potential is only well-typed if both share a gradient filter."""
        lj, d3 = self._potentials(POSITIONS_AND_CELL)
        total = sum_potentials(lj, d3)
        g_lj = np.asarray(lj(self.state).data.gradients.positions.data)
        g_d3 = np.asarray(d3(self.state).data.gradients.positions.data)
        g_total = np.asarray(total(self.state).data.gradients.positions.data)
        npt.assert_allclose(g_total, g_lj + g_d3, rtol=1e-10, atol=1e-14)

    def test_d3_lowers_the_energy_of_a_dimer(self) -> None:
        """A sanity check that the correction is attractive and non-trivial."""
        lj, d3 = self._potentials(None)
        e_lj = float(lj(self.state).data.total_energies.data[0])
        e_total = float(sum_potentials(lj, d3)(self.state).data.total_energies.data[0])
        assert e_total < e_lj


class TestProductionNeighborList:
    """D3 through the neighbor list a real run uses, not the fixed dense one.

    Every other test here pins ``neighborlist_factory`` to a deterministic dense
    list so that composition and batching are compared like for like. That
    leaves the default -- ``AdaptiveNeighborList.from_state``, which dispatches
    between dense and cell-list implementations -- unexercised, so the reference
    agreement would be established against a neighbor list nobody runs.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "water_dimer",  # molecular: a vacuum box, where the density estimate is ~0
            "co2",
            "si_diamond",
            "si_sheared",  # triclinic
            "nacl",
            "si_slab",  # periodic in x/y, open in z
        ],
    )
    def test_matches_the_fixed_dense_list(self, name: str) -> None:
        """The default factory must reproduce the dense list exactly, not merely closely.

        Capacities come from ``estimate`` and are grown by the assertion retry
        loop, which is the mechanism a real run relies on -- and which this test
        must go through rather than around. ``estimate`` sizes from particle
        density, so a molecule in the builder's 1000 Å vacuum box starts at
        ``avg_edges=1``; evaluating that directly returns a truncated, silently
        wrong energy, and only the retry loop grows it.
        """
        from kups.potential.common.evaluation import (
            evaluate_potential_and_fix,
            potential_with_assertions,
        )
        from kups.potential.common.graph import GraphPotentialInput
        from kups.potential.dispersion.d3 import d3_energy

        system = SYSTEMS[name]
        graph = build_graph([system], _CUTOFF, edge_capacity=1 << 16)
        expected = float(d3_energy(GraphPotentialInput(_D3, graph)).data.data[0])
        assert expected < 0.0, "the reference energy must be non-trivial"

        state = _State(
            particles=graph.particles,
            systems=graph.systems,
            neighborlist_params=UniversalNeighborlistParameters.estimate(
                graph.particles.data.system.counts, graph.systems, _D3.cutoff
            ),
        )
        # no neighborlist_factory: this is the production default
        potential = make_d3_from_state(identity_lens(_State), parameters=_D3)
        _, out = evaluate_potential_and_fix(potential_with_assertions(potential), state)
        npt.assert_allclose(
            float(out.data.total_energies.data[0]), expected, rtol=1e-12
        )


class TestBatching:
    @pytest.mark.parametrize(
        "names",
        [
            ("water", "co2", "benzene"),  # different compositions and atom counts
            ("water_dimer", "ar2"),
        ],
    )
    def test_batched_matches_individual(self, names: tuple[str, ...]) -> None:
        """Systems in a batch must not interact, whatever their size or composition."""
        systems = [SYSTEMS[name] for name in names]
        batched = _d3_potential()(_state_for(systems)).data.total_energies.data
        assert batched.shape == (len(systems),)
        individual = [
            float(_d3_potential()(_state_for([system])).data.total_energies.data[0])
            for system in systems
        ]
        npt.assert_allclose(np.asarray(batched), individual, rtol=1e-12)

    def test_batched_periodic_cells_of_different_shape(self) -> None:
        """Per-system *cells* index correctly under batching.

        Says nothing about per-system cutoffs: the parameters here are scalar, so
        every system sees the same value. That case is
        ``test_per_system_cutoffs_reach_their_own_system``.
        """
        names = ("si_diamond", "nacl", "si_sheared")
        systems = [SYSTEMS[name] for name in names]
        batched = _d3_potential()(_state_for(systems)).data.total_energies.data
        individual = [
            float(_d3_potential()(_state_for([system])).data.total_energies.data[0])
            for system in systems
        ]
        npt.assert_allclose(np.asarray(batched), individual, rtol=1e-12)

    def test_per_system_cutoffs_reach_their_own_system(self) -> None:
        """A per-system cutoff table must reach the system it belongs to.

        The adapter hands ``parameters.cutoff`` to the neighbor list factory and
        the kernel masks with the same table, so a bug that collapses the table to
        its first entry -- or broadcasts one system's value across the batch --
        changes energies without raising. Every other batching test here uses a
        scalar cutoff, where such a bug is invisible because all entries agree.

        Two identical argon dimers at 8 Å with cutoffs 6 Å and 12 Å: the first is
        out of range and must contribute nothing, the second is in range.
        """
        dimer = System(
            np.array([18, 18]),
            np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
            None,
            (False, False, False),
        )
        parameters = D3Parameters.from_damping(
            s8=0.7875, a1=0.4289, a2=4.4407, cutoff=jnp.array([6.0, 12.0])
        )
        potential = make_d3_from_state(
            identity_lens(_State),
            parameters=parameters,
            neighborlist_factory=_dense_factory,
        )
        energies = np.asarray(
            potential(_state_for([dimer, dimer])).data.total_energies.data
        )
        assert energies.shape == (2,)
        assert energies[0] == 0.0, "8 Å is outside the 6 Å cutoff of system 0"
        # value cross-checked against simple-dftd3 for a single Ar2 at 8 Å
        npt.assert_allclose(energies[1], -1.6397172064686028e-4, rtol=1e-9)

    def test_batched_gradients_are_block_diagonal(self) -> None:
        """An atom's force may not depend on a system it does not belong to."""
        systems = [SYSTEMS["water"], SYSTEMS["co2"]]
        state = _state_for(systems)
        potential = _d3_potential(gradient=POSITIONS_AND_CELL)
        gradients = np.asarray(potential(state).data.gradients.positions.data)

        moved = _state_for(
            [
                SYSTEMS["water"],
                System(
                    SYSTEMS["co2"].numbers,
                    SYSTEMS["co2"].positions + np.array([0.11, 0.0, 0.0]),
                    None,
                    (False, False, False),
                ),
            ]
        )
        moved_gradients = np.asarray(potential(moved).data.gradients.positions.data)
        n_water = len(SYSTEMS["water"].numbers)
        npt.assert_allclose(
            moved_gradients[:n_water], gradients[:n_water], rtol=1e-10, atol=1e-14
        )


class TestParameterSource:
    """Parameters may be bound into the potential or carried on the state."""

    def test_state_carried_parameters_match_passed_parameters(self) -> None:
        """Both routes must produce the same potential, as for every other adapter."""

        @dataclass
        class _StateWithParameters:
            particles: Table[ParticleId, D3Points]
            systems: Table[SystemId, D3Systems]
            neighborlist_params: UniversalNeighborlistParameters
            d3_parameters: D3Parameters

        base = _state_for([SYSTEMS["water_dimer"]])
        carried = _StateWithParameters(
            particles=base.particles,
            systems=base.systems,
            neighborlist_params=base.neighborlist_params,
            d3_parameters=_D3,
        )

        from_state = make_d3_from_state(
            identity_lens(_StateWithParameters), neighborlist_factory=_dense_factory
        )
        passed = _d3_potential()

        npt.assert_allclose(
            float(from_state(carried).data.total_energies.data[0]),
            float(passed(base).data.total_energies.data[0]),
            rtol=1e-12,
        )

    def test_state_carried_cutoff_drives_the_neighbor_list(self) -> None:
        """The cutoff must be read through the parameter view, not a stale closure.

        Checked by observable effect rather than instrumentation: the cutoff is a
        traced value inside the jitted potential, so it cannot be inspected from a
        recording factory. An argon dimer at 8 Å is outside a 6 Å cutoff and inside
        a 12 Å one, so the state's value decides whether any edge exists at all.
        """

        @dataclass
        class _StateWithParameters:
            particles: Table[ParticleId, D3Points]
            systems: Table[SystemId, D3Systems]
            neighborlist_params: UniversalNeighborlistParameters
            d3_parameters: D3Parameters

        dimer = System(
            np.array([18, 18]),
            np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
            None,
            (False, False, False),
        )
        potential = make_d3_from_state(
            identity_lens(_StateWithParameters), neighborlist_factory=_dense_factory
        )

        def energy_at(cutoff: float) -> float:
            base = _state_for([dimer])
            state = _StateWithParameters(
                particles=base.particles,
                systems=base.systems,
                neighborlist_params=base.neighborlist_params,
                d3_parameters=D3Parameters.from_functional("pbe", cutoff=cutoff),
            )
            return float(potential(state).data.total_energies.data[0])

        assert energy_at(6.0) == 0.0
        assert energy_at(12.0) < 0.0

    def test_neighbor_list_is_built_from_cutoff_not_cn_cutoff(self) -> None:
        """The pair cutoff sizes the list; the CN cutoff only masks within it.

        ``cn_cutoff`` is capped at ``cutoff`` and the two are often equal, which
        makes them easy to confuse at the one call site that matters. Building the
        list from ``cn_cutoff`` would still pass every test with a scalar
        parameter set -- the two are then the same number -- and would silently
        truncate the dispersion sum whenever a caller narrows the CN range, which
        is the documented way to make CN cheaper.

        An argon dimer at 20 Å is inside a 30 Å pair cutoff and outside a 10 Å CN
        cutoff, so the choice decides whether the pair exists at all.
        """
        dimer = System(
            np.array([18, 18]),
            np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            None,
            (False, False, False),
        )
        parameters = D3Parameters.from_damping(
            s8=0.7875, a1=0.4289, a2=4.4407, cutoff=30.0, cn_cutoff=10.0
        )
        assert float(parameters.cn_cutoff.data[0]) == 10.0, "premise: the two differ"

        state = _state_for([dimer])
        potential = make_d3_from_state(
            identity_lens(_State),
            parameters=parameters,
            neighborlist_factory=_dense_factory,
        )
        energy = float(potential(state).data.total_energies.data[0])
        assert energy < 0.0, (
            "the 20 Å pair is inside the 30 Å pair cutoff and must contribute; "
            "zero means the list was sized from cn_cutoff"
        )


class TestStateSemantics:
    """Regressions for defects that only appear through the state-binding adapter."""

    def _state_with_exclusion(self, groups: np.ndarray) -> _State:
        """A water molecule whose atoms share the given exclusion ids."""
        from kups.core.typing import ExclusionId

        base = _state_for([SYSTEMS["water"]])
        data = base.particles.data
        n = len(base.particles)
        replaced = D3Points(
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            system=data.system,
            inclusion=data.inclusion,
            exclusion=Index(tuple(map(ExclusionId, range(n))), jnp.asarray(groups)),
        )
        return _State(
            particles=Table.arange(replaced, label=ParticleId),
            systems=base.systems,
            neighborlist_params=base.neighborlist_params,
        )

    def _state_with_inclusion(self, segments: np.ndarray) -> _State:
        """A water molecule whose atoms carry the given native inclusion ids."""
        from kups.core.typing import InclusionId

        base = _state_for([SYSTEMS["water"]])
        data = base.particles.data
        n = len(base.particles)
        replaced = D3Points(
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            system=data.system,
            inclusion=Index(tuple(map(InclusionId, range(n))), jnp.asarray(segments)),
            exclusion=data.exclusion,
        )
        return _State(
            particles=Table.arange(replaced, label=ParticleId),
            systems=base.systems,
            neighborlist_params=base.neighborlist_params,
        )

    def test_native_inclusion_segments_do_not_delete_dispersion(self) -> None:
        """The adapter must re-derive ``inclusion`` from the system, not reuse it.

        [InclusionMatchMask][kups.core.neighborlist.masks.InclusionMatchMask] drops
        every pair whose endpoints sit in different inclusion segments. A state is
        free to use that index for something finer than "which system" -- and if
        the adapter passed it through, D3 would return exactly zero for a molecule
        whose atoms happen to be in distinct segments. Zero is the dangerous
        answer here: it is finite, plausible for a weak correction, and wrong.

        This is the ``inclusion`` twin of
        ``test_molecular_exclusion_groups_do_not_delete_dispersion``; the two
        indices fail in different ways and need separate cover.
        """
        potential = _d3_potential()
        one_segment = float(
            potential(
                self._state_with_inclusion(np.zeros(3, dtype=int))
            ).data.total_energies.data[0]
        )
        assert one_segment < 0.0, "water must have non-zero intramolecular dispersion"
        per_atom_segments = float(
            potential(
                self._state_with_inclusion(np.arange(3))
            ).data.total_energies.data[0]
        )
        npt.assert_allclose(per_atom_segments, one_segment, rtol=1e-12)

    def test_molecular_exclusion_groups_do_not_delete_dispersion(self) -> None:
        """Dispersion acts within a molecule, so exclusion groups must not remove it.

        A state's ``exclusion`` index often means something other than "never
        interacts" — in an MCMC state it is the molecular group, used to strip
        intramolecular Coulomb. ``ExclusionMask`` drops the minimum-image pair of
        anything sharing an id, so passing that index through unchanged would
        silently zero every intramolecular D3 term. The adapter must re-index
        particles with one exclusion id each.
        """
        potential = _d3_potential()
        unique = float(
            potential(
                self._state_with_exclusion(np.arange(3))
            ).data.total_energies.data[0]
        )
        grouped = float(
            potential(
                self._state_with_exclusion(np.zeros(3, dtype=int))
            ).data.total_energies.data[0]
        )
        assert unique < 0.0, "water must have non-zero intramolecular dispersion"
        npt.assert_allclose(grouped, unique, rtol=1e-12)

    def _buffered_state(self, n_pad: int) -> _State:
        """A water dimer in a particle buffer with ``n_pad`` unoccupied slots.

        Built through ``Buffered`` rather than by hand so the test tracks the
        real sanitisation: a grand-canonical state (``mcmc_rigid``) keeps its
        particles in a buffer whose free slots are zeroed, which means their
        ``atomic_numbers`` read back as ``Z = 0``.
        """
        from kups.core.data.buffered import Buffered
        from kups.core.typing import ExclusionId, InclusionId

        base = _state_for([SYSTEMS["water_dimer"]])
        data = base.particles.data
        n_real = len(base.particles)
        # an out-of-bounds system id is what marks a slot unoccupied
        system = jnp.concatenate(
            [data.system.indices, jnp.full(n_pad, len(base.systems))]
        )
        raw = D3Points(
            positions=jnp.vstack([data.positions, jnp.zeros((n_pad, 3))]),
            atomic_numbers=jnp.concatenate(
                # junk in the free slots: Buffered must be what zeroes them
                [data.atomic_numbers, jnp.full(n_pad, 6)]
            ),
            system=Index((SystemId(0),), system),
            inclusion=Index((InclusionId(0),), system),
            exclusion=Index.integer(
                jnp.arange(n_real + n_pad), n=n_real + n_pad, label=ExclusionId
            ),
        )
        buffered = Buffered.arange(raw, view=lambda x: x.system, label=ParticleId)
        assert int(buffered.num_occupied) == n_real
        assert not np.asarray(buffered.data.atomic_numbers)[n_real:].any(), (
            "Buffered is expected to zero the free slots; the test relies on it"
        )
        return _State(
            particles=buffered,
            systems=base.systems,
            neighborlist_params=base.neighborlist_params,
        )

    def test_unoccupied_buffer_slots_are_not_mistaken_for_bad_elements(self) -> None:
        """A grand-canonical state must evaluate, not trip the element check.

        ``Buffered`` zeroes every non-index leaf of an unoccupied slot, so a
        GCMC particle table carries ``Z = 0`` whenever it is not saturated —
        which is essentially always. Those slots are already inert: their system
        id is out of bounds, so the neighbor list drops them and their reference
        weights are masked to zero. Rejecting them as unsupported elements would
        block D3 from the adsorption workflow it exists to serve.
        """
        potential = _d3_potential()
        packed = float(
            potential(_state_for([SYSTEMS["water_dimer"]])).data.total_energies.data[0]
        )
        assert packed < 0.0, "the reference energy must be non-trivial"
        for n_pad in (1, 5):
            padded = float(
                potential(self._buffered_state(n_pad)).data.total_energies.data[0]
            )
            npt.assert_allclose(padded, packed, rtol=1e-12)

    @pytest.mark.parametrize("atomic_number", [104, 0, -1])
    def test_unsupported_atomic_numbers_are_rejected(self, atomic_number: int) -> None:
        """Out-of-range Z must fail, not silently alias a valid element.

        JAX clamps out-of-bounds gather indices, so without an explicit check
        ``Z = 104`` quietly borrows lawrencium's coefficients and returns a
        plausible number.
        """
        system = System(
            np.array([atomic_number, 6]),
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            None,
            (False, False, False),
        )
        state = _state_for([system])
        potential = _d3_potential()

        with pytest.raises(ValueError) as exc_info:
            potential(state)

        lo, hi = min(atomic_number, 6), max(atomic_number, 6)
        assert str(exc_info.value) == (
            f"D3 covers atomic numbers 1..103; got {lo}..{hi}."
        )

    @pytest.mark.parametrize("atomic_number", [104, 0, -1])
    def test_unsupported_atomic_numbers_are_captured_when_traced(
        self, atomic_number: int
    ) -> None:
        """The same validation must survive JAX tracing in standard evaluators."""
        from kups.core.assertion import with_runtime_assertions

        system = System(
            np.array([atomic_number, 6]),
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            None,
            (False, False, False),
        )
        state = _state_for([system])
        potential = _d3_potential()

        @with_runtime_assertions
        def evaluate(s: _State):
            return potential(s).data.total_energies.data

        _, assertions = evaluate(state)
        failed = [
            assertion
            for assertion in assertions
            if assertion.failed() and assertion.exception_type is ValueError
        ]
        assert failed, f"Z={atomic_number} was accepted silently"
        lo, hi = min(atomic_number, 6), max(atomic_number, 6)
        assert str(failed[0]).splitlines()[0] == (
            f"D3 covers atomic numbers 1..103; got {lo}..{hi}."
        )

    def test_supported_atomic_numbers_raise_nothing(self) -> None:
        from kups.core.assertion import with_runtime_assertions

        potential = _d3_potential()

        @with_runtime_assertions
        def evaluate(s: _State):
            return potential(s).data.total_energies.data

        _, assertions = evaluate(_state_for([SYSTEMS["water"]]))
        assert not [assertion for assertion in assertions if assertion.failed()]
