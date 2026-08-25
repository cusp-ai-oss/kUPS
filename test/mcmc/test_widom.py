# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for [kups.mcmc.widom][kups.mcmc.widom]."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array

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
    EnergyMoments,
    GhostProbe,
    TransitionStatistics,
    WidomStatistics,
    widom_test,
)


def _sys_table[T](values: T) -> Table[SystemId, T]:
    return Table.arange(values, label=SystemId)


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


class TestWidomStatistics:
    def test_reset_clears_sums(self):
        stats = WidomStatistics.zeros(2)
        for _ in range(7):
            stats = stats.update(jnp.array([0.0, 0.5]), jnp.array([1.0, 2.0]))
        r = stats.reset()
        npt.assert_array_equal(r.sum_boltzmann, jnp.zeros(2))
        npt.assert_array_equal(r.sum_delta_u_boltzmann, jnp.zeros(2))
        npt.assert_array_equal(r.n_samples, jnp.zeros(2, dtype=int))


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
        npt.assert_array_equal(stats.n_samples, jnp.array([1, 1], dtype=int))

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


# -- TransitionStatistics (TMMC C-matrix) --------------------------------


class TestTransitionStatistics:
    def test_zeros_has_correct_shape_and_dtype(self):
        stats = TransitionStatistics.zeros(4)
        assert stats.acceptance_insertion.shape == (4,)
        assert jnp.issubdtype(stats.n_trials_insertion.dtype, jnp.integer)
        npt.assert_array_equal(stats.acceptance_insertion, jnp.zeros(4))
        npt.assert_array_equal(stats.n_trials_deletion, jnp.zeros(4, dtype=int))

    def test_reset_restores_zeros(self):
        stats = TransitionStatistics(
            acceptance_insertion=jnp.array([1.0, 2.0]),
            acceptance_deletion=jnp.array([0.5, 1.5]),
            n_trials_insertion=jnp.array([7, 9], dtype=int),
            n_trials_deletion=jnp.array([7, 9], dtype=int),
        )
        reset = stats.reset()
        npt.assert_array_equal(reset.acceptance_insertion, jnp.zeros(2))
        npt.assert_array_equal(reset.n_trials_insertion, jnp.zeros(2, dtype=int))

    def test_update_insertion_clamps_and_increments(self):
        stats = TransitionStatistics.zeros(3)
        # ln α values: log(2) > 0 (clamps to 1), -1 (exp=0.368), -10 (exp≈0)
        ln_alpha = jnp.array([jnp.log(2.0), -1.0, -10.0])
        updated = stats.update_insertion(ln_alpha)
        expected_acc = jnp.minimum(1.0, jnp.exp(ln_alpha))
        npt.assert_allclose(updated.acceptance_insertion, expected_acc, rtol=1e-10)
        npt.assert_array_equal(updated.n_trials_insertion, jnp.ones(3, dtype=int))
        # Deletion side untouched
        npt.assert_array_equal(updated.acceptance_deletion, jnp.zeros(3))
        npt.assert_array_equal(updated.n_trials_deletion, jnp.zeros(3, dtype=int))

    def test_update_deletion_masks_at_N0_but_increments_trial(self):
        stats = TransitionStatistics.zeros(3)
        ln_alpha = jnp.array([0.0, -1.0, -2.0])  # exp = 1, 0.368, 0.135
        macrostate_n = jnp.array([0, 5, 3])  # system 0 has no particles
        updated = stats.update_deletion(ln_alpha, macrostate_n)

        expected_acc = jnp.minimum(1.0, jnp.exp(ln_alpha))
        expected_acc = jnp.where(macrostate_n > 0, expected_acc, 0.0)
        npt.assert_allclose(updated.acceptance_deletion, expected_acc, rtol=1e-10)
        # The trial count increments even at N = 0 so P(0 → 1) is not inflated.
        npt.assert_array_equal(updated.n_trials_deletion, jnp.ones(3, dtype=int))

    def test_repeated_updates_accumulate(self):
        stats = TransitionStatistics.zeros(2)
        for _ in range(5):
            stats = stats.update_insertion(jnp.array([0.0, 0.0]))  # acc=1 each
        npt.assert_allclose(stats.acceptance_insertion, jnp.array([5.0, 5.0]))
        npt.assert_array_equal(stats.n_trials_insertion, jnp.array([5, 5], dtype=int))


# -- EnergyMoments (Welford / Pébay) ------------------------------------


def _numpy_central_moments(samples: np.ndarray) -> tuple[float, float, float, float]:
    """Reference: compute mean + central moments 2-4 with NumPy (two-pass)."""
    mean = samples.mean()
    deviations = samples - mean
    m2 = np.mean(deviations**2)
    m3 = np.mean(deviations**3)
    m4 = np.mean(deviations**4)
    return float(mean), float(m2), float(m3), float(m4)


class TestEnergyMoments:
    @pytest.mark.parametrize("n_samples", [1, 2, 5, 20, 100])
    def test_welford_matches_numpy_reference(self, n_samples: int):
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=3.0, scale=2.5, size=n_samples)

        moments = EnergyMoments.zeros(1)
        for x in samples:
            moments = moments.update(jnp.array([x]))

        cumulants = moments.finalize()
        ref_mean, ref_m2, ref_m3, ref_m4 = _numpy_central_moments(samples)

        npt.assert_allclose(cumulants.mean[0], ref_mean, rtol=1e-10)
        npt.assert_allclose(cumulants.variance[0], ref_m2, rtol=1e-10)
        npt.assert_allclose(cumulants.cumulants[0, 1], ref_m3, rtol=1e-9, atol=1e-12)
        npt.assert_allclose(
            cumulants.cumulants[0, 2], ref_m4 - 3 * ref_m2**2, rtol=1e-9, atol=1e-12
        )

    def test_arbitrary_order_matches_numpy_reference(self):
        """max_order=6: M_p sums and the cumulant recursion (kappa_5, kappa_6)
        against two-pass NumPy references."""
        rng = np.random.default_rng(7)
        samples = rng.exponential(scale=1.5, size=200)  # skewed on purpose

        moments = EnergyMoments.zeros(1, max_order=6)
        for x in samples:
            moments = moments.update(jnp.array([x]))

        dev = samples - samples.mean()
        mu = {p: np.mean(dev**p) for p in range(2, 7)}
        for p in range(2, 7):
            npt.assert_allclose(
                moments.central_sums[0, p - 2], mu[p] * len(samples), rtol=1e-8
            )

        c = moments.finalize()
        npt.assert_allclose(c.cumulants[0, 3], mu[5] - 10 * mu[2] * mu[3], rtol=1e-8)
        npt.assert_allclose(
            c.cumulants[0, 4],
            mu[6] - 15 * mu[2] * mu[4] - 10 * mu[3] ** 2 + 30 * mu[2] ** 3,
            rtol=1e-8,
        )

    def test_multi_system_independence(self):
        """Two systems with different sample streams must stay independent."""
        rng = np.random.default_rng(0)
        s1 = rng.normal(size=50)
        s2 = rng.normal(loc=10.0, scale=0.1, size=50)

        moments = EnergyMoments.zeros(2)
        for x1, x2 in zip(s1, s2, strict=True):
            moments = moments.update(jnp.array([float(x1), float(x2)]))
        c = moments.finalize()

        npt.assert_allclose(c.mean[0], s1.mean(), rtol=1e-10)
        npt.assert_allclose(c.mean[1], s2.mean(), rtol=1e-10)
        npt.assert_allclose(c.variance[0], s1.var(), rtol=1e-10)
        npt.assert_allclose(c.variance[1], s2.var(), rtol=1e-10)

    def test_reset_restores_zeros(self):
        moments = EnergyMoments.zeros(3)
        for _ in range(10):
            moments = moments.update(jnp.array([1.0, 2.0, 3.0]))
        reset = moments.reset()
        npt.assert_array_equal(reset.count, jnp.zeros(3, dtype=int))
        npt.assert_array_equal(reset.mean, jnp.zeros(3))
        npt.assert_array_equal(reset.central_sums, jnp.zeros((3, 3)))
