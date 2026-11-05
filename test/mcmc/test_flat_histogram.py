# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for [kups.mcmc.flat_histogram][kups.mcmc.flat_histogram]."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.constants import BOLTZMANN_CONSTANT, KELVIN, PASCAL
from kups.core.data import Table
from kups.core.typing import SystemId
from kups.mcmc.flat_histogram import (
    AdsorbateEOS,
    TMMCSummary,
    _isosteric_heat,
    _isotherm,
    _working_capacity,
    average_loading,
    extrapolate_log_partition,
    macrostate_distribution,
    reconstruct_log_partition,
)
from kups.mcmc.fugacity import peng_robinson_log_fugacity
from kups.mcmc.widom import EnergyCumulants, TransitionStatistics

# CO2-like EOS parameters for integration tests.
_CO2 = AdsorbateEOS.single_species(
    critical_pressure=7.38e6,  # Pa
    critical_temperature=304.2,  # K
    acentric_factor=0.228,
)


def _rows(tree):
    """Add a leading (length-1) system axis to every leaf of ``tree``."""
    return jax.tree.map(lambda x: jnp.asarray(x)[None], tree)


def _summary(
    log_partition_sim: jnp.ndarray,
    cumulants: EnergyCumulants,
    beta_sim: jnp.ndarray,
    order: int = 1,
) -> TMMCSummary:
    """Wrap one system's raw ladder into a keyed :class:`TMMCSummary`."""
    return TMMCSummary(
        log_partition_sim=Table.arange(_rows(log_partition_sim), label=SystemId),
        cumulants=Table.arange(_rows(cumulants), label=SystemId),
        beta_sim=Table.arange(_rows(beta_sim), label=SystemId),
        adsorbates=_CO2,
        order=order,
    )


# -- reconstruct_log_partition ---------------------------------------


class TestReconstructLogPartition:
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

        reconstructed = reconstruct_log_partition(p_ins_full, p_del, beta, log_f)
        npt.assert_allclose(reconstructed, ln_qc_true, rtol=1e-10, atol=1e-12)

    def test_anchors_to_zero_at_N0(self):
        p_ins = jnp.array([0.1, 0.2, 0.0])
        p_del = jnp.array([0.5, 0.3, 0.4])
        result = reconstruct_log_partition(
            p_ins, p_del, jnp.asarray(1.0), jnp.asarray(0.0)
        )
        assert float(result[0]) == 0.0

    def test_prefix_sum_runs_along_the_trailing_axis(self):
        """A stack of ladders reconstructs row-by-row, per-system β and ln f."""
        p_ins = jnp.array([[0.1, 0.2, 0.0], [0.3, 0.05, 0.0]])
        p_del = jnp.array([[0.5, 0.3, 0.4], [0.5, 0.2, 0.7]])
        beta = jnp.array([1.0, 2.0])
        log_f = jnp.array([0.0, -1.0])

        batched = reconstruct_log_partition(p_ins, p_del, beta, log_f)
        assert batched.shape == (2, 3)
        for row in range(2):
            single = reconstruct_log_partition(
                p_ins[row], p_del[row], beta[row], log_f[row]
            )
            npt.assert_array_equal(batched[row], single)


# -- extrapolate_log_partition ---------------------------------------


class TestExtrapolate:
    def test_cubic_ln_qc_is_exact_at_order_3(self):
        r"""For $\ln Q_c(\beta)$ cubic in $\beta$, the 3rd-order Taylor is exact."""
        beta_0 = 2.0
        c0, c1, c2, c3 = 0.5, -3.0, 1.2, -0.4

        def true_log_qc(beta: float) -> float:
            return c0 + c1 * beta + c2 * beta**2 + c3 * beta**3

        # Build derivatives ⟨E⟩=-dln Q/dβ, Var=d²ln Q/dβ², κ_3=d³ln Q/dβ³.
        dlnq_dbeta = c1 + 2 * c2 * beta_0 + 3 * c3 * beta_0**2
        # kappa_2 = d²lnQ/dβ², kappa_3 = -d³lnQ/dβ³ (standard cumulants).
        cumulants = EnergyCumulants(
            mean=jnp.asarray(-dlnq_dbeta),
            cumulants=jnp.stack(
                [
                    jnp.asarray(2 * c2 + 6 * c3 * beta_0),
                    jnp.asarray(-6 * c3),
                    jnp.asarray(0.0),
                ]
            ),
        )
        log_qc_sim = jnp.asarray(true_log_qc(beta_0))
        for beta_target in [2.0, 2.1, 1.8, 3.0]:
            result = extrapolate_log_partition(
                log_qc_sim,
                cumulants,
                jnp.asarray(beta_0),
                jnp.asarray(beta_target),
                order=3,
            )
            npt.assert_allclose(
                float(result), true_log_qc(beta_target), rtol=1e-10, atol=1e-10
            )

    def test_quintic_ln_qc_is_exact_at_order_5(self):
        r"""Arbitrary-degree extrapolation: a quintic $\ln Q_c(\beta)$ is exact
        at order 5 when the cumulants encode its derivatives."""
        beta_0 = 1.5
        coeffs = [0.3, -2.0, 0.9, -0.25, 0.05, -0.004]  # c_0 .. c_5

        def true_log_qc(beta: float) -> float:
            return sum(c * beta**j for j, c in enumerate(coeffs))

        def derivative(k: int) -> float:
            return sum(
                c * (math.factorial(j) // math.factorial(j - k)) * beta_0 ** (j - k)
                for j, c in enumerate(coeffs)
                if j >= k
            )

        # kappa_k = (-1)^k d^k lnQ/dbeta^k (standard cumulants).
        cumulants = EnergyCumulants(
            mean=jnp.asarray(-derivative(1)),
            cumulants=jnp.stack(
                [jnp.asarray((-1.0) ** k * derivative(k)) for k in range(2, 6)]
            ),
        )
        log_qc_sim = jnp.asarray(true_log_qc(beta_0))
        for beta_target in [1.5, 1.2, 2.0]:
            result = extrapolate_log_partition(
                log_qc_sim,
                cumulants,
                jnp.asarray(beta_0),
                jnp.asarray(beta_target),
                order=5,
            )
            npt.assert_allclose(
                float(result), true_log_qc(beta_target), rtol=1e-10, atol=1e-10
            )

    def test_zero_displacement_returns_simulation_value(self):
        cumulants = EnergyCumulants(
            mean=jnp.array([1.0, 2.0, 3.0]),
            cumulants=jnp.array(
                [
                    [0.1, 0.01, 0.001],
                    [0.2, 0.02, 0.002],
                    [0.3, 0.03, 0.003],
                ]
            ),
        )
        log_qc = jnp.array([0.0, -1.0, -2.0])
        result = extrapolate_log_partition(
            log_qc, cumulants, jnp.asarray(1.0), jnp.asarray(1.0), order=4
        )
        npt.assert_allclose(result, log_qc, rtol=1e-12)

    def test_batched_ladders_extrapolate_row_by_row(self):
        """The moment-order axis stays trailing; macrostates sit before it."""
        cumulants = EnergyCumulants(
            mean=jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]]),
            cumulants=jnp.stack(
                [
                    jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                    jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
                ],
                axis=-1,
            ),
        )
        log_qc = jnp.array([[0.0, -1.0, -2.0], [0.0, -0.5, -3.0]])
        beta_sim = jnp.array([1.0, 1.5])

        batched = extrapolate_log_partition(
            log_qc, cumulants, beta_sim, jnp.asarray(1.2), order=3
        )
        assert batched.shape == (2, 3)
        for row in range(2):
            single = extrapolate_log_partition(
                log_qc[row],
                jax.tree.map(lambda x: x[row], cumulants),  # noqa: B023
                beta_sim[row],
                jnp.asarray(1.2),
                order=3,
            )
            npt.assert_array_equal(batched[row], single)

    @pytest.mark.parametrize("order", [0, 5, -1])
    def test_invalid_order_raises(self, order: int):
        cumulants = EnergyCumulants(mean=jnp.array([0.0]), cumulants=jnp.zeros((1, 3)))
        with pytest.raises(ValueError):
            extrapolate_log_partition(
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

    def test_normalises_each_row_of_a_batch(self):
        """Softmax runs over the trailing macrostate axis, not the batch."""
        log_qc = jnp.array([[0.0, -1.0, -0.5], [0.0, 2.0, 1.0]])
        dist = macrostate_distribution(
            log_qc, jnp.array([1.0, 3.0]), jnp.array([-1.0, 0.5])
        )
        assert dist.shape == (2, 3)
        npt.assert_allclose(jnp.sum(dist, axis=-1), jnp.ones(2), rtol=1e-12)

    def test_average_loading_matches_hand_calculation(self):
        dist = jnp.array([0.2, 0.3, 0.4, 0.1])
        expected = 0 * 0.2 + 1 * 0.3 + 2 * 0.4 + 3 * 0.1
        npt.assert_allclose(float(average_loading(dist)), expected, rtol=1e-12)

    def test_average_loading_reduces_only_the_macrostate_axis(self):
        dist = jnp.array([[0.2, 0.3, 0.4, 0.1], [0.5, 0.5, 0.0, 0.0]])
        loadings = average_loading(dist)
        assert loadings.shape == (2,)
        npt.assert_allclose(loadings, jnp.array([1.4, 0.5]), rtol=1e-12)


# -- isotherm -----------------------------------------------------------


class TestIsotherm:
    def test_shape_matches_pressure_grid(self):
        log_qc = jnp.array([0.0, -1.0, -2.0, -2.5])
        pressures = jnp.array([1e4, 1e5, 1e6])
        loadings = _isotherm(
            log_qc,
            jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0)),
            pressures,
            jnp.asarray(300.0),
            _CO2,
        )
        assert loadings.shape == (3,)
        assert jnp.all(loadings >= 0)
        assert jnp.all(loadings <= len(log_qc) - 1)

    def test_monotonic_in_pressure_for_attractive_site(self):
        """⟨N⟩ must be non-decreasing in P for a fixed-site attractive model."""
        log_qc = jnp.linspace(0.0, -10.0, 6)  # increasingly favourable binding
        pressures = jnp.logspace(3, 7, 10)
        loadings = _isotherm(
            log_qc,
            jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0)),
            pressures,
            jnp.asarray(300.0),
            _CO2,
        )
        diffs = jnp.diff(loadings)
        assert jnp.all(diffs >= -1e-10), f"loadings non-monotonic: {loadings}"

    def test_pressure_axis_is_appended_to_per_system_axes(self):
        log_qc = jnp.array([[0.0, -1.0, -2.0, -2.5], [0.0, -3.0, -4.0, -4.5]])
        pressures = jnp.array([1e4, 1e5, 1e6])
        loadings = _isotherm(
            log_qc,
            jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0)),
            pressures,
            jnp.asarray(300.0),
            _CO2,
        )
        assert loadings.shape == (2, 3)

    def test_rejects_multicomponent_eos_table(self):
        two_species = Table.arange(
            AdsorbateEOS(
                critical_pressure=jnp.array([7.38e6, 4.86e6]),
                critical_temperature=jnp.array([304.2, 150.7]),
                acentric_factor=jnp.array([0.228, -0.002]),
            ),
            label=SystemId,
        )
        with pytest.raises(AssertionError, match="single adsorbate species"):
            _isotherm(
                jnp.array([0.0, -1.0]),
                jnp.asarray(1.0),
                jnp.array([1e5]),
                jnp.asarray(300.0),
                two_species,
            )


# -- isosteric heat (autodiff Clausius-Clapeyron) -----------------------


class TestIsostericHeat:
    def test_finite_and_has_correct_shape(self):
        log_qc_sim = jnp.array([0.0, -3.0, -5.0, -6.0])
        cumulants = EnergyCumulants(
            mean=jnp.array([0.0, -0.2, -0.35, -0.5]),
            cumulants=jnp.stack(
                [jnp.array([0.0, 0.01, 0.02, 0.03]), jnp.zeros(4), jnp.zeros(4)],
                axis=-1,
            ),
        )
        beta_sim = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        pressures = jnp.array([1e4, 1e5, 1e6])

        q = _isosteric_heat(
            log_qc_sim,
            cumulants,
            beta_sim,
            pressures,
            jnp.asarray(300.0),
            _CO2,
            order=2,
        )
        assert q.shape == (3,)
        assert jnp.all(jnp.isfinite(q))

    def test_matches_central_difference(self):
        """Autodiff q_st agrees with central differences through Peng-Robinson."""
        from kups.mcmc.flat_histogram import _loading_from_temperature_log_pressure

        log_qc_sim = jnp.array([0.0, -3.0, -5.0, -6.0])
        cumulants = EnergyCumulants(
            mean=jnp.array([0.0, -0.2, -0.35, -0.5]),
            cumulants=jnp.stack(
                [jnp.array([0.0, 0.01, 0.02, 0.03]), jnp.zeros(4), jnp.zeros(4)],
                axis=-1,
            ),
        )
        beta_sim = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        T = 300.0
        P_test = 1e5
        log_P = float(jnp.log(P_test))

        def loading(t: float, lp: float) -> float:
            return float(
                _loading_from_temperature_log_pressure(
                    jnp.asarray(t),
                    jnp.asarray(lp),
                    log_qc_sim,
                    cumulants,
                    beta_sim,
                    2,
                    _CO2,
                )
            )

        h_T = 0.1
        h_lp = 1e-3
        dT_fd = (loading(T + h_T, log_P) - loading(T - h_T, log_P)) / (2.0 * h_T)
        dlogP_fd = (loading(T, log_P + h_lp) - loading(T, log_P - h_lp)) / (2.0 * h_lp)
        q_fd = BOLTZMANN_CONSTANT * T**2 * dT_fd / dlogP_fd

        q_ad = _isosteric_heat(
            log_qc_sim,
            cumulants,
            beta_sim,
            jnp.array([P_test]),
            jnp.asarray(T),
            _CO2,
            order=2,
        )
        npt.assert_allclose(float(q_ad[0]), q_fd, rtol=1e-3)


# -- working_capacity ---------------------------------------------------


class TestWorkingCapacity:
    def test_swing_loading_difference(self):
        """n_wc = ⟨N⟩_ads - ⟨N⟩_des; both terms come from the same ladder."""
        log_qc_sim = jnp.linspace(0.0, -4.0, 5)
        cumulants = EnergyCumulants(mean=jnp.zeros(5), cumulants=jnp.zeros((5, 3)))
        beta_sim = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        n_wc = _working_capacity(
            log_qc_sim,
            cumulants,
            beta_sim,
            jnp.asarray(1e6),  # adsorption at high P
            jnp.asarray(1e3),  # desorption at low P
            jnp.asarray(300.0),
            _CO2,
            order=1,
        )
        # Positive because ads P > des P and the material is attractive.
        assert float(n_wc) > 0


# -- end-to-end post-processing: analytical Langmuir -------------------


class TestAnalyticalLangmuirRoundTrip:
    r"""Full pipeline reproduces an analytic $M$-site Langmuir isotherm.

    Synthetic C-matrix counts satisfying eq 8 for
    $\ln Q_c(N) = \ln\binom{M}{N} + N\beta\varepsilon$ must give
    $\langle N \rangle = M x / (1 + x)$, $x = \beta f e^{\beta\varepsilon}$.
    """

    def test_isotherm_matches_langmuir_closed_form(self):
        M = 10
        eps_ev = 0.05
        T = 298.15
        beta = 1.0 / (BOLTZMANN_CONSTANT * T)
        P_sim = 1.0e5

        eos = _CO2.data
        pr_sim = peng_robinson_log_fugacity(
            jnp.asarray(P_sim * PASCAL),
            jnp.asarray(T * KELVIN),
            eos.critical_pressure * PASCAL,
            eos.critical_temperature * KELVIN,
            eos.acentric_factor,
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

        n_trials = jnp.full(M + 1, 10_000, dtype=int)
        acc_ins = p_ins * 2 * n_trials.astype(float)
        acc_del = p_del * 2 * n_trials.astype(float)

        # Cumulants aren't exercised here (β_target == β_sim); pass zeros.
        cumulants = EnergyCumulants(
            mean=jnp.zeros(M + 1), cumulants=jnp.zeros((M + 1, 3))
        )
        summary = TMMCSummary.from_transition_statistics(
            Table.arange(
                _rows(TransitionStatistics(acc_ins, acc_del, n_trials, n_trials)),
                label=SystemId,
            ),
            cumulants=Table.arange(_rows(cumulants), label=SystemId),
            beta_sim=Table.arange(_rows(jnp.asarray(beta)), label=SystemId),
            log_fugacity_sim=Table.arange(_rows(log_f_sim), label=SystemId),
            adsorbates=_CO2,
            order=1,
        )

        # ln Q_c must round-trip exactly.
        npt.assert_allclose(
            summary.log_partition_sim.data[0], log_qc_true, atol=1e-12, rtol=0
        )

        # Isotherm at several pressures must match the closed-form Langmuir.
        pressures = jnp.logspace(2, 7, 6)
        tmmc_loading = summary.isotherm(pressures, jnp.asarray(T))

        def analytical(P_pa: float) -> float:
            pr = peng_robinson_log_fugacity(
                jnp.asarray(P_pa * PASCAL),
                jnp.asarray(T * KELVIN),
                eos.critical_pressure * PASCAL,
                eos.critical_temperature * KELVIN,
                eos.acentric_factor,
            )
            log_f = float(pr.log_fugacity[0])
            x = np.exp(log_f + np.log(float(beta)) + float(beta) * eps_ev)
            return M * x / (1.0 + x)

        expected = np.array([analytical(float(p)) for p in pressures])
        npt.assert_allclose(tmmc_loading.data[0], expected, atol=1e-12, rtol=1e-10)


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

        n_trials = jnp.full(4, 1000, dtype=int)
        # P = acc / (n_ins + n_del) ⇒ acc = P · 2 · n.
        acc_ins = p_ins * (2 * n_trials.astype(float))
        acc_del = p_del * (2 * n_trials.astype(float))

        cumulants = EnergyCumulants(mean=jnp.zeros(4), cumulants=jnp.zeros((4, 3)))
        summary = TMMCSummary.from_transition_statistics(
            Table.arange(
                _rows(TransitionStatistics(acc_ins, acc_del, n_trials, n_trials)),
                label=SystemId,
            ),
            cumulants=Table.arange(_rows(cumulants), label=SystemId),
            beta_sim=Table.arange(_rows(beta), label=SystemId),
            log_fugacity_sim=Table.arange(_rows(log_f), label=SystemId),
            adsorbates=_CO2,
            order=1,
        )
        assert summary.log_partition_sim.keys == (SystemId(0),)
        npt.assert_allclose(
            summary.log_partition_sim.data[0], ln_qc_true, rtol=1e-10, atol=1e-10
        )

    def test_methods_return_keyed_tables(self):
        """Every observable keeps the system keys and appends its own axis."""
        ln_qc_sim = jnp.array([0.0, -2.0, -3.5, -4.5])
        cumulants = EnergyCumulants(mean=jnp.zeros(4), cumulants=jnp.zeros((4, 3)))
        beta_sim = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        summary = _summary(ln_qc_sim, cumulants, beta_sim, order=1)
        pressures = jnp.array([1e4, 1e5])
        temperature = jnp.asarray(300.0)

        loading = summary.isotherm(pressures, temperature)
        heat = summary.isosteric_heat(pressures, temperature)
        capacity = summary.working_capacity(
            jnp.asarray(1e6), jnp.asarray(1e3), temperature
        )
        log_partition = summary.extrapolate(beta_sim)

        for result in (loading, heat, capacity, log_partition):
            assert isinstance(result, Table)
            assert result.keys == (SystemId(0),)
        assert loading.data.shape == (1, 2)
        assert heat.data.shape == (1, 2)
        assert capacity.data.shape == (1,)
        assert log_partition.data.shape == (1, 4)

    def test_methods_dispatch_to_module_level_helpers(self):
        """Sanity: summary methods produce the same values as the raw helpers."""
        ln_qc_sim = jnp.array([0.0, -2.0, -3.5, -4.5])
        cumulants = EnergyCumulants(mean=jnp.zeros(4), cumulants=jnp.zeros((4, 3)))
        beta_sim = jnp.asarray(1.0 / (BOLTZMANN_CONSTANT * 300.0))
        summary = _summary(ln_qc_sim, cumulants, beta_sim, order=1)
        pressures = jnp.array([1e4, 1e5])
        via_method = summary.isotherm(pressures, jnp.asarray(300.0))
        via_function = _isotherm(
            ln_qc_sim,
            beta_sim,
            pressures,
            jnp.asarray(300.0),
            _CO2,
        )
        npt.assert_allclose(via_method.data[0], via_function, rtol=1e-10)


# -- batched ladders ---------------------------------------------------


class TestBatchedLadders:
    """A stack of ladders is bitwise identical to looping over them."""

    @staticmethod
    def _inputs() -> tuple[
        TransitionStatistics, EnergyCumulants, jnp.ndarray, jnp.ndarray
    ]:
        acc_ins = jnp.array([[0.9, 0.6, 0.2, 0.0], [0.7, 0.45, 0.1, 0.0]])
        acc_del = jnp.array([[0.0, 0.8, 0.9, 0.95], [0.0, 0.5, 0.85, 0.9]])
        n_trials = jnp.full((2, 4), 500, dtype=int)
        statistics = TransitionStatistics(
            acceptance_insertion=acc_ins * 1000.0,
            acceptance_deletion=acc_del * 1000.0,
            n_trials_insertion=n_trials,
            n_trials_deletion=n_trials,
        )
        cumulants = EnergyCumulants(
            mean=jnp.array([[0.0, -0.2, -0.35, -0.5], [0.0, -0.15, -0.3, -0.44]]),
            cumulants=jnp.stack(
                [
                    jnp.array([[0.0, 0.01, 0.02, 0.03], [0.0, 0.02, 0.03, 0.04]]),
                    jnp.zeros((2, 4)),
                    jnp.zeros((2, 4)),
                ],
                axis=-1,
            ),
        )
        beta_sim = 1.0 / (BOLTZMANN_CONSTANT * jnp.array([300.0, 280.0]))
        log_fugacity_sim = jnp.array([-1.0, -0.5])
        return statistics, cumulants, beta_sim, log_fugacity_sim

    @classmethod
    def _summary(cls, system: int | None) -> TMMCSummary:
        """Summary over both systems (``None``) or over one system alone."""
        statistics, cumulants, beta_sim, log_fugacity_sim = cls._inputs()
        if system is not None:
            take = lambda x: x[system][None]  # noqa: E731
            statistics = jax.tree.map(take, statistics)
            cumulants = jax.tree.map(take, cumulants)
            beta_sim = take(beta_sim)
            log_fugacity_sim = take(log_fugacity_sim)
        return TMMCSummary.from_transition_statistics(
            Table.arange(statistics, label=SystemId),
            cumulants=Table.arange(cumulants, label=SystemId),
            beta_sim=Table.arange(beta_sim, label=SystemId),
            log_fugacity_sim=Table.arange(log_fugacity_sim, label=SystemId),
            adsorbates=_CO2,
            order=2,
        )

    def test_batched_equals_per_system_loop(self):
        pressures = jnp.array([1e4, 1e5, 1e6])
        temperature = jnp.asarray(310.0)

        batched = self._summary(None)
        assert len(batched.log_partition_sim) == 2
        batched_results = (
            batched.log_partition_sim,
            batched.isotherm(pressures, temperature),
            batched.isosteric_heat(pressures, temperature),
            batched.working_capacity(jnp.asarray(1e6), jnp.asarray(1e3), temperature),
        )
        for system in range(2):
            single = self._summary(system)
            single_results = (
                single.log_partition_sim,
                single.isotherm(pressures, temperature),
                single.isosteric_heat(pressures, temperature),
                single.working_capacity(
                    jnp.asarray(1e6), jnp.asarray(1e3), temperature
                ),
            )
            for batch, one in zip(batched_results, single_results, strict=True):
                npt.assert_array_equal(batch.data[system], one.data[0])
