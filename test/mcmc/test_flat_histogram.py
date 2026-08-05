# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for [kups.mcmc.flat_histogram][kups.mcmc.flat_histogram]."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.constants import BOLTZMANN_CONSTANT, KELVIN, PASCAL
from kups.mcmc.flat_histogram import (
    AdsorbateEOS,
    TMMCSummary,
    average_loading,
    extrapolate_log_partition_fn,
    isosteric_heat,
    isotherm,
    macrostate_distribution,
    reconstruct_log_partition_fn,
    transition_probabilities,
    working_capacity,
)
from kups.mcmc.fugacity import peng_robinson_log_fugacity
from kups.mcmc.widom import EnergyCumulants

# CO2-like EOS parameters for integration tests.
_CO2 = AdsorbateEOS(
    critical_pressure=7.38e6,  # Pa
    critical_temperature=304.2,  # K
    acentric_factor=0.228,
)


# -- transition_probabilities --------------------------------------------


class TestTransitionProbabilities:
    def test_normalises_by_total_trials(self):
        acc_ins = jnp.array([1.0, 2.0, 0.5])
        acc_del = jnp.array([0.0, 1.0, 0.3])
        n_ins = jnp.array([4, 4, 4], dtype=jnp.int32)
        n_del = jnp.array([4, 4, 4], dtype=jnp.int32)
        p_ins, p_del = transition_probabilities(acc_ins, acc_del, n_ins, n_del)
        npt.assert_allclose(p_ins, acc_ins / 8.0)
        npt.assert_allclose(p_del, acc_del / 8.0)

    def test_safe_at_zero_trials(self):
        """No divide-by-zero when a macrostate was never visited."""
        acc = jnp.zeros(3)
        n = jnp.zeros(3, dtype=jnp.int32)
        p_ins, p_del = transition_probabilities(acc, acc, n, n)
        npt.assert_array_equal(p_ins, jnp.zeros(3))
        npt.assert_array_equal(p_del, jnp.zeros(3))


# -- reconstruct_log_partition_fn ---------------------------------------


class TestReconstructLogPartitionFn:
    def test_roundtrip_from_known_log_qc(self):
        """Given ln Q_c, construct P_ins/P_del from detailed balance and recover it."""
        ln_qc_true = jnp.array([0.0, -1.0, -2.5, -4.0, -5.0])
        beta = jnp.asarray(1.5)
        beta_f = 0.01
        log_f = jnp.log(beta_f)

        delta_true = jnp.diff(ln_qc_true)
        p_del = jnp.array([0.5, 1.0, 1.0, 1.0, 1.0])
        # P_ins[N] = β f · exp(Δ ln Q_c(N)) · P_del[N+1]
        p_ins_core = beta * beta_f * jnp.exp(delta_true) * p_del[1:]
        p_ins_full = jnp.concatenate([p_ins_core, jnp.zeros(1)])

        reconstructed = reconstruct_log_partition_fn(p_ins_full, p_del, beta, log_f)
        npt.assert_allclose(reconstructed, ln_qc_true, rtol=1e-10, atol=1e-12)

    def test_anchors_to_zero_at_N0(self):
        p_ins = jnp.array([0.1, 0.2, 0.0])
        p_del = jnp.array([0.5, 0.3, 0.4])
        result = reconstruct_log_partition_fn(
            p_ins, p_del, jnp.asarray(1.0), jnp.asarray(0.0)
        )
        assert float(result[0]) == 0.0


# -- extrapolate_log_partition_fn ---------------------------------------


class TestExtrapolate:
    def test_cubic_ln_qc_is_exact_at_order_3(self):
        r"""For $\ln Q_c(\beta)$ cubic in $\beta$, the 3rd-order Taylor is exact."""
        beta_0 = 2.0
        c0, c1, c2, c3 = 0.5, -3.0, 1.2, -0.4

        def true_log_qc(beta: float) -> float:
            return c0 + c1 * beta + c2 * beta**2 + c3 * beta**3

        # Build derivatives ⟨E⟩=-dln Q/dβ, Var=d²ln Q/dβ², κ_3=d³ln Q/dβ³.
        dlnq_dbeta = c1 + 2 * c2 * beta_0 + 3 * c3 * beta_0**2
        cumulants = EnergyCumulants(
            mean=jnp.asarray(-dlnq_dbeta),
            variance=jnp.asarray(2 * c2 + 6 * c3 * beta_0),
            third=jnp.asarray(6 * c3),
            fourth=jnp.asarray(0.0),
        )
        log_qc_sim = jnp.asarray(true_log_qc(beta_0))
        for beta_target in [2.0, 2.1, 1.8, 3.0]:
            result = extrapolate_log_partition_fn(
                log_qc_sim,
                cumulants,
                jnp.asarray(beta_0),
                jnp.asarray(beta_target),
                order=3,
            )
            npt.assert_allclose(
                float(result), true_log_qc(beta_target), rtol=1e-10, atol=1e-10
            )

    def test_zero_displacement_returns_simulation_value(self):
        cumulants = EnergyCumulants(
            mean=jnp.array([1.0, 2.0, 3.0]),
            variance=jnp.array([0.1, 0.2, 0.3]),
            third=jnp.array([0.01, 0.02, 0.03]),
            fourth=jnp.array([0.001, 0.002, 0.003]),
        )
        log_qc = jnp.array([0.0, -1.0, -2.0])
        result = extrapolate_log_partition_fn(
            log_qc, cumulants, jnp.asarray(1.0), jnp.asarray(1.0), order=4
        )
        npt.assert_allclose(result, log_qc, rtol=1e-12)

    @pytest.mark.parametrize("order", [0, 5, -1])
    def test_invalid_order_raises(self, order: int):
        cumulants = EnergyCumulants(
            mean=jnp.array([0.0]),
            variance=jnp.array([0.0]),
            third=jnp.array([0.0]),
            fourth=jnp.array([0.0]),
        )
        with pytest.raises(ValueError):
            extrapolate_log_partition_fn(
                jnp.array([0.0]),
                cumulants,
                jnp.asarray(1.0),
                jnp.asarray(1.0),
                order=order,
            )


# -- macrostate_distribution + average_loading --------------------------


class TestMacrostateDistribution:
    def test_normalises_to_unity(self):
        log_qc = jnp.array([0.0, -1.0, -0.5, -2.0])
        dist = macrostate_distribution(log_qc, jnp.asarray(1.0), jnp.asarray(-1.0))
        npt.assert_allclose(float(jnp.sum(dist)), 1.0, rtol=1e-12)
        assert jnp.all(dist >= 0)

    def test_matches_analytical_weights(self):
        """For a 2-state system, the distribution follows the sigmoid form."""
        ln_qc = jnp.array([0.0, 0.7])
        beta = 2.0
        log_f = -1.5
        dist = macrostate_distribution(ln_qc, jnp.asarray(beta), jnp.asarray(log_f))
        # log-weights: N=0 → 0, N=1 → (log_f + log β)·1 + 0.7
        log_w = np.array([0.0, log_f + np.log(beta) + 0.7])
        expected = np.exp(log_w) / np.exp(log_w).sum()
        npt.assert_allclose(dist, expected, rtol=1e-10)

    def test_average_loading_matches_hand_calculation(self):
        dist = jnp.array([0.2, 0.3, 0.4, 0.1])
        expected = 0 * 0.2 + 1 * 0.3 + 2 * 0.4 + 3 * 0.1
        npt.assert_allclose(float(average_loading(dist)), expected, rtol=1e-12)


# -- isotherm -----------------------------------------------------------


class TestIsotherm:
    def test_shape_matches_pressure_grid(self):
        log_qc = jnp.array([0.0, -1.0, -2.0, -2.5])
        pressures = jnp.array([1e4, 1e5, 1e6])
        loadings = isotherm(
            log_qc,
            jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0)),
            pressures,
            jnp.asarray(300.0),
            _CO2.critical_pressure,
            _CO2.critical_temperature,
            _CO2.acentric_factor,
        )
        assert loadings.shape == (3,)
        assert jnp.all(loadings >= 0)
        assert jnp.all(loadings <= len(log_qc) - 1)

    def test_monotonic_in_pressure_for_attractive_site(self):
        """⟨N⟩ must be non-decreasing in P for a fixed-site attractive model."""
        log_qc = jnp.linspace(0.0, -10.0, 6)  # increasingly favourable binding
        pressures = jnp.logspace(3, 7, 10)
        loadings = isotherm(
            log_qc,
            jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0)),
            pressures,
            jnp.asarray(300.0),
            _CO2.critical_pressure,
            _CO2.critical_temperature,
            _CO2.acentric_factor,
        )
        diffs = jnp.diff(loadings)
        assert jnp.all(diffs >= -1e-10), f"loadings non-monotonic: {loadings}"


# -- isosteric heat (autodiff Clausius-Clapeyron) -----------------------


class TestIsostericHeat:
    def test_finite_and_has_correct_shape(self):
        log_qc_sim = jnp.array([0.0, -3.0, -5.0, -6.0])
        cumulants = EnergyCumulants(
            mean=jnp.array([0.0, -0.2, -0.35, -0.5]),
            variance=jnp.array([0.0, 0.01, 0.02, 0.03]),
            third=jnp.zeros(4),
            fourth=jnp.zeros(4),
        )
        beta_sim = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        pressures = jnp.array([1e4, 1e5, 1e6])

        q = isosteric_heat(
            log_qc_sim,
            cumulants,
            beta_sim,
            pressures,
            jnp.asarray(300.0),
            _CO2.critical_pressure,
            _CO2.critical_temperature,
            _CO2.acentric_factor,
            order=2,
        )
        assert q.shape == (3,)
        assert jnp.all(jnp.isfinite(q))

    def test_matches_central_difference(self):
        """Autodiff q_st must agree with a local central-difference oracle.

        Regression guard: autodiff through Peng-Robinson relies on the custom
        JVP attached to
        [`cubic_roots`][kups.core.utils.math.cubic_roots] and the double-where
        sanitisation in
        [`peng_robinson_log_fugacity`][kups.mcmc.fugacity.peng_robinson_log_fugacity].
        A regression in either would show up as a divergence between the
        autodiff Clausius--Clapeyron result and local finite differences.
        """
        from kups.mcmc.flat_histogram import _loading_from_T_logP

        log_qc_sim = jnp.array([0.0, -3.0, -5.0, -6.0])
        cumulants = EnergyCumulants(
            mean=jnp.array([0.0, -0.2, -0.35, -0.5]),
            variance=jnp.array([0.0, 0.01, 0.02, 0.03]),
            third=jnp.zeros(4),
            fourth=jnp.zeros(4),
        )
        beta_sim = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        T = 300.0
        P_test = 1e5
        log_P = float(jnp.log(P_test))

        def loading(t: float, lp: float) -> float:
            return float(
                _loading_from_T_logP(
                    jnp.asarray(t),
                    jnp.asarray(lp),
                    log_qc_sim,
                    cumulants,
                    beta_sim,
                    2,
                    _CO2.critical_pressure,
                    _CO2.critical_temperature,
                    _CO2.acentric_factor,
                )
            )

        h_T = 0.1
        h_lp = 1e-3
        dT_fd = (loading(T + h_T, log_P) - loading(T - h_T, log_P)) / (2.0 * h_T)
        dlogP_fd = (loading(T, log_P + h_lp) - loading(T, log_P - h_lp)) / (2.0 * h_lp)
        q_fd = BOLTZMANN_CONSTANT * T**2 * dT_fd / dlogP_fd

        q_ad = isosteric_heat(
            log_qc_sim,
            cumulants,
            beta_sim,
            jnp.array([P_test]),
            jnp.asarray(T),
            _CO2.critical_pressure,
            _CO2.critical_temperature,
            _CO2.acentric_factor,
            order=2,
        )
        npt.assert_allclose(float(q_ad[0]), q_fd, rtol=1e-3)


# -- working_capacity ---------------------------------------------------


class TestWorkingCapacity:
    def test_swing_loading_difference(self):
        """n_wc = ⟨N⟩_ads - ⟨N⟩_des; both terms come from the same summary."""
        log_qc_sim = jnp.linspace(0.0, -4.0, 5)
        cumulants = EnergyCumulants(
            mean=jnp.zeros(5),
            variance=jnp.zeros(5),
            third=jnp.zeros(5),
            fourth=jnp.zeros(5),
        )
        beta_sim = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        n_wc = working_capacity(
            log_qc_sim,
            cumulants,
            beta_sim,
            jnp.asarray(300.0),
            jnp.asarray(1e6),  # adsorption at high P
            jnp.asarray(300.0),
            jnp.asarray(1e3),  # desorption at low P
            _CO2.critical_pressure,
            _CO2.critical_temperature,
            _CO2.acentric_factor,
            order=1,
        )
        # Positive because ads P > des P and the material is attractive.
        assert float(n_wc) > 0


# -- end-to-end post-processing: analytical Langmuir -------------------


class TestAnalyticalLangmuirRoundTrip:
    """The full TMMC post-processing chain must reproduce a known closed-form
    isotherm to machine precision when fed consistent synthetic statistics.

    Given an $M$-site non-interacting Langmuir model with per-site binding
    energy $\\varepsilon$,

    $$\\ln Q_c(N) = \\ln\\binom{M}{N} + N \\beta \\varepsilon,$$

    we build transition-matrix counts that exactly satisfy Witman eq 8 and
    then run the full pipeline:

    C-matrix $\\to$ :func:`transition_probabilities` $\\to$
    :func:`reconstruct_log_partition_fn` $\\to$
    :class:`TMMCSummary` $\\to$ :func:`isotherm`.

    The resulting $\\langle N\\rangle(P)$ is compared against the closed-form
    Langmuir $\\langle N \\rangle = M\\,x/(1+x)$ with
    $x = \\beta f(P)\\,\\exp(\\beta\\varepsilon)$. Agreement is expected to
    float64 precision.
    """

    def test_isotherm_matches_langmuir_closed_form(self):
        M = 10
        eps_ev = 0.05
        T = 298.15
        beta = 1.0 / (BOLTZMANN_CONSTANT * T)
        P_sim = 1.0e5

        pr_sim = peng_robinson_log_fugacity(
            jnp.asarray(P_sim * PASCAL),
            jnp.asarray(T * KELVIN),
            jnp.atleast_1d(jnp.asarray(_CO2.critical_pressure) * PASCAL),
            jnp.atleast_1d(jnp.asarray(_CO2.critical_temperature) * KELVIN),
            jnp.atleast_1d(jnp.asarray(_CO2.acentric_factor)),
        )
        log_f_sim = jnp.asarray(float(pr_sim.log_fugacity[0]))

        n_values = jnp.arange(M + 1)
        log_choose = jnp.array(
            [float(np.log(math.comb(M, int(k)))) for k in range(M + 1)]
        )
        log_qc_true = log_choose + n_values * beta * eps_ev

        # Invert Witman eq 8 to construct consistent P_ins/P_del arrays.
        delta = jnp.diff(log_qc_true)
        p_del = jnp.full(M + 1, 0.5)
        p_ins_core = p_del[1:] * jnp.exp(delta + log_f_sim + jnp.log(beta))
        p_ins = jnp.concatenate([p_ins_core, jnp.zeros(1)])

        n_trials = jnp.full(M + 1, 10_000, dtype=jnp.int32)
        acc_ins = p_ins * 2 * n_trials.astype(jnp.float64)
        acc_del = p_del * 2 * n_trials.astype(jnp.float64)

        # Cumulants aren't exercised here (β_target == β_sim); pass zeros.
        cumulants = EnergyCumulants(
            mean=jnp.zeros(M + 1),
            variance=jnp.zeros(M + 1),
            third=jnp.zeros(M + 1),
            fourth=jnp.zeros(M + 1),
        )
        summary = TMMCSummary.from_transition_statistics(
            acc_ins,
            acc_del,
            n_trials,
            n_trials,
            cumulants=cumulants,
            beta_sim=jnp.asarray(beta),
            log_fugacity_sim=log_f_sim,
            adsorbate=_CO2,
            order=1,
        )

        # ln Q_c must round-trip exactly.
        npt.assert_allclose(
            summary.log_partition_fn_sim, log_qc_true, atol=1e-12, rtol=0
        )

        # Isotherm at several pressures must match the closed-form Langmuir.
        pressures = jnp.logspace(2, 7, 6)
        tmmc_loading = summary.isotherm(pressures, jnp.asarray(T))

        def analytical(P_pa: float) -> float:
            pr = peng_robinson_log_fugacity(
                jnp.asarray(P_pa * PASCAL),
                jnp.asarray(T * KELVIN),
                jnp.atleast_1d(jnp.asarray(_CO2.critical_pressure) * PASCAL),
                jnp.atleast_1d(jnp.asarray(_CO2.critical_temperature) * KELVIN),
                jnp.atleast_1d(jnp.asarray(_CO2.acentric_factor)),
            )
            log_f = float(pr.log_fugacity[0])
            x = np.exp(log_f + np.log(float(beta)) + float(beta) * eps_ev)
            return M * x / (1.0 + x)

        expected = np.array([analytical(float(p)) for p in pressures])
        npt.assert_allclose(tmmc_loading, expected, atol=1e-12, rtol=1e-10)


# -- TMMCSummary facade -------------------------------------------------


class TestTMMCSummary:
    def test_from_transition_statistics_roundtrip(self):
        ln_qc_true = jnp.array([0.0, -1.5, -3.0, -4.0])
        beta = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        log_f = jnp.asarray(-1.0)

        # Build consistent P_ins/P_del from true ln Q_c, then back out counts.
        delta = jnp.diff(ln_qc_true)
        p_del = jnp.array([0.5, 1.0, 1.0, 1.0])
        beta_f = float(jnp.exp(log_f) * beta)
        p_ins_core = beta_f * jnp.exp(delta) * p_del[1:]
        p_ins = jnp.concatenate([p_ins_core, jnp.zeros(1)])

        n_trials = jnp.full(4, 1000, dtype=jnp.int32)
        # P = acc / (n_ins + n_del) ⇒ acc = P · 2 · n.
        acc_ins = p_ins * (2 * n_trials.astype(jnp.float64))
        acc_del = p_del * (2 * n_trials.astype(jnp.float64))

        cumulants = EnergyCumulants(
            mean=jnp.zeros(4),
            variance=jnp.zeros(4),
            third=jnp.zeros(4),
            fourth=jnp.zeros(4),
        )
        summary = TMMCSummary.from_transition_statistics(
            acc_ins,
            acc_del,
            n_trials,
            n_trials,
            cumulants=cumulants,
            beta_sim=beta,
            log_fugacity_sim=log_f,
            adsorbate=_CO2,
            order=1,
        )
        npt.assert_allclose(
            summary.log_partition_fn_sim, ln_qc_true, rtol=1e-10, atol=1e-10
        )

    def test_methods_dispatch_to_module_level_functions(self):
        """Sanity: summary methods produce the same values as calling the functions."""
        ln_qc_sim = jnp.array([0.0, -2.0, -3.5, -4.5])
        cumulants = EnergyCumulants(
            mean=jnp.zeros(4),
            variance=jnp.zeros(4),
            third=jnp.zeros(4),
            fourth=jnp.zeros(4),
        )
        beta_sim = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        summary = TMMCSummary(
            log_partition_fn_sim=ln_qc_sim,
            cumulants=cumulants,
            beta_sim=beta_sim,
            adsorbate=_CO2,
            order=1,
        )
        pressures = jnp.array([1e4, 1e5])
        via_method = summary.isotherm(pressures, jnp.asarray(300.0))
        via_function = isotherm(
            summary.extrapolate(beta_sim),
            beta_sim,
            pressures,
            jnp.asarray(300.0),
            _CO2.critical_pressure,
            _CO2.critical_temperature,
            _CO2.acentric_factor,
        )
        npt.assert_allclose(via_method, via_function, rtol=1e-10)
