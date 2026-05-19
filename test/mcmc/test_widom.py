# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for [kups.mcmc.widom][kups.mcmc.widom]."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data.table import Table
from kups.core.lens import lens
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.propagator import (
    ChangesFn,
    LogProbabilityRatio,
    LogProbabilityRatioFn,
    PatchFn,
)
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass
from kups.mcmc.widom import (
    GhostProbe,
    WidomStatistics,
    finalize_widom,
    widom_test,
)


def _sys_table[T](values: T) -> Table[SystemId, T]:
    return Table.arange(values, label=SystemId)


# -- synthetic state + propose/patch/log_ratio stubs ---------------------


@dataclass
class _DummyState:
    energies: Array
    widom_statistics: Table[SystemId, WidomStatistics]


def _dummy_state(n_systems: int, energies: Array | None = None) -> _DummyState:
    if energies is None:
        energies = jnp.arange(n_systems, dtype=jnp.float64)
    return _DummyState(
        energies=energies,
        widom_statistics=_sys_table(WidomStatistics.zeros(n_systems)),
    )


def _propose_stub(
    n_systems: int, move_log_ratio: Array
) -> ChangesFn[_DummyState, None]:
    keys = tuple(SystemId(i) for i in range(n_systems))

    def propose(key: Array, state: _DummyState) -> tuple[None, LogProbabilityRatio]:
        del key, state
        return None, Table(keys, move_log_ratio)

    return propose


def _patch_stub() -> PatchFn[_DummyState, None, IdPatch[_DummyState]]:
    def patch_fn(
        key: Array, state: _DummyState, proposal: None
    ) -> IdPatch[_DummyState]:
        del key, state, proposal
        return IdPatch[_DummyState]()

    return patch_fn


def _ratio_stub(
    n_systems: int, density_log_ratio: Array
) -> LogProbabilityRatioFn[_DummyState, IdPatch[_DummyState]]:
    keys = tuple(SystemId(i) for i in range(n_systems))

    def ratio(
        state: _DummyState, patch: IdPatch[_DummyState]
    ) -> WithPatch[LogProbabilityRatio, Patch[_DummyState]]:
        del state, patch
        return WithPatch(Table(keys, density_log_ratio), IdPatch[_DummyState]())

    return ratio


# -- widom_test ----------------------------------------------------------


class TestWidomTest:
    def test_returns_sum_of_move_and_density_log_ratios(self):
        n_systems = 3
        move = jnp.array([0.1, -0.2, 0.3])
        density = jnp.array([0.5, 0.5, -0.1])
        state = _dummy_state(n_systems)

        result = widom_test(
            jax.random.key(0),
            state,
            _propose_stub(n_systems, move),
            _patch_stub(),
            _ratio_stub(n_systems, density),
        )
        npt.assert_allclose(result.data, move + density, rtol=1e-12)

    def test_does_not_modify_state(self):
        n_systems = 2
        state = _dummy_state(n_systems)
        old_leaves = jax.tree.leaves(state)

        _ = widom_test(
            jax.random.key(7),
            state,
            _propose_stub(n_systems, jnp.zeros(n_systems)),
            _patch_stub(),
            _ratio_stub(n_systems, jnp.array([1.23, -4.56])),
        )
        # IdPatch never fires; state is physically unchanged. Verify the
        # original leaves match in pytree order.
        new_leaves = jax.tree.leaves(state)
        assert len(old_leaves) == len(new_leaves)
        for a, b in zip(old_leaves, new_leaves, strict=True):
            npt.assert_array_equal(a, b)


# -- WidomStatistics + finalize_widom -----------------------------------


class TestWidomStatistics:
    def test_constant_boltzmann_factor_recovers_mu_excess_and_qst(self):
        """Single ΔU value across all insertions: μ_ex = ΔU and q_st = kT − ΔU."""
        n_samples = 10
        temperature = jnp.array([300.0])
        volume = jnp.array([100.0])
        delta_U = -0.05  # eV
        kT = float(BOLTZMANN_CONSTANT * 300.0)
        beta = 1.0 / kT
        ln_alpha = jnp.array([-beta * delta_U])
        delta_u_arr = jnp.array([delta_U])

        stats = WidomStatistics.zeros(1)
        for _ in range(n_samples):
            stats = stats.update(ln_alpha, delta_u_arr)

        result = finalize_widom(stats, temperature, volume)
        # ⟨W⟩ = exp(-β ΔU), μ_ex = -kT ln⟨W⟩ = ΔU
        npt.assert_allclose(result.excess_chemical_potential[0], delta_U, rtol=1e-9)
        # K_H = V ⟨W⟩ / kT = V exp(-β ΔU) / kT
        npt.assert_allclose(
            result.henry_coefficient[0],
            float(volume[0]) * float(jnp.exp(-beta * delta_U)) / kT,
            rtol=1e-9,
        )
        # q_st = kT − ⟨ΔU·W⟩/⟨W⟩ = kT − ΔU when ΔU is constant
        npt.assert_allclose(result.heat_of_adsorption[0], kT - delta_U, rtol=1e-9)

    def test_reset_clears_sums(self):
        stats = WidomStatistics.zeros(2)
        for _ in range(7):
            stats = stats.update(jnp.array([0.0, 0.5]), jnp.array([1.0, 2.0]))
        r = stats.reset()
        npt.assert_array_equal(r.sum_boltzmann, jnp.zeros(2))
        npt.assert_array_equal(r.sum_delta_u_boltzmann, jnp.zeros(2))
        npt.assert_array_equal(r.n_samples, jnp.zeros(2, dtype=jnp.int32))


# -- GhostProbe propagator ----------------------------------------------


class TestGhostProbe:
    def test_accumulates_via_stat_lens_and_update_fn(self):
        n_systems = 2
        energies = jnp.array([0.0, 1.5])
        state = _dummy_state(n_systems, energies=energies)

        def update(state_: _DummyState, stats: WidomStatistics, ln_a: Array):
            return stats.update(ln_a, state_.energies)

        stat_lens = lens(lambda s: s.widom_statistics.data, cls=_DummyState)

        ln_alpha = jnp.array([0.0, jnp.log(0.5)])
        probe = GhostProbe(
            propose_fn=_propose_stub(n_systems, jnp.zeros(n_systems)),
            patch_fn=_patch_stub(),
            log_probability_ratio_fn=_ratio_stub(n_systems, ln_alpha),
            stat_lens=stat_lens,
            update_fn=update,
        )

        new_state = probe(jax.random.key(3), state)
        stats = new_state.widom_statistics.data
        # One update with ln α = [0, ln 0.5] → boltzmann = [1, 0.5].
        npt.assert_allclose(stats.sum_boltzmann, jnp.array([1.0, 0.5]), rtol=1e-10)
        # Update fn passed ``state_.energies`` as ΔU, so ⟨ΔU·W⟩ = energies · W.
        npt.assert_allclose(
            stats.sum_delta_u_boltzmann,
            energies * jnp.array([1.0, 0.5]),
            rtol=1e-10,
        )
        npt.assert_array_equal(stats.n_samples, jnp.array([1, 1], dtype=jnp.int32))

    def test_does_not_mutate_non_stat_state(self):
        n_systems = 2
        state = _dummy_state(n_systems)

        def update(state_: _DummyState, stats: WidomStatistics, ln_a: Array):
            return stats.update(ln_a, state_.energies)

        probe = GhostProbe(
            propose_fn=_propose_stub(n_systems, jnp.zeros(n_systems)),
            patch_fn=_patch_stub(),
            log_probability_ratio_fn=_ratio_stub(n_systems, jnp.zeros(n_systems)),
            stat_lens=lens(lambda s: s.widom_statistics.data, cls=_DummyState),
            update_fn=update,
        )

        new_state = probe(jax.random.key(0), state)
        # `energies` is the non-accumulator field and must round-trip unchanged.
        npt.assert_array_equal(new_state.energies, state.energies)
