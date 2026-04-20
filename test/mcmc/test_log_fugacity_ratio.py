# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy.testing as npt
from jax import Array

from kups.core.data.table import Table
from kups.core.patch import IdPatch
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass
from kups.core.utils.math import log_factorial_ratio
from kups.mcmc.probability import LogFugacityRatio


@dataclass
class MockState:
    log_fug: Table[SystemId, Array]
    volume: Table[SystemId, Array]
    counts: Table[SystemId, Array]


@dataclass
class MockPatch:
    delta: Array  # (n_sys, n_species)

    def __call__(self, state, accept):
        return state


def get_log_fugacity(state: MockState) -> Table[SystemId, Array]:
    return state.log_fug


def get_volume(state: MockState) -> Table[SystemId, Array]:
    return state.volume


def get_counts(state: MockState) -> Table[SystemId, Array]:
    return state.counts


def get_patched_counts(state: MockState, patch: MockPatch) -> Table[SystemId, Array]:
    return state.counts.set_data(state.counts.data + patch.delta)


class TestLogFugacityRatio:
    def setup_method(self):
        # 3 systems, 1 species; friendly values for exact checks
        self.log_fug = Table.arange(
            jnp.log(jnp.array([[2.0], [2.0], [2.0]])), label=SystemId
        )
        self.volume = Table.arange(jnp.array([1.0, 2.0, 3.0]), label=SystemId)
        self.counts = Table.arange(jnp.array([[0], [1], [2]]), label=SystemId)

        self.state = MockState(self.log_fug, self.volume, self.counts)

        # Construct probability object with functional views
        self.ln_ratio = LogFugacityRatio(
            log_activity=get_log_fugacity,
            volume=get_volume,
            particle_counts=get_counts,
            patched_particle_counts=get_patched_counts,
        )

    def test_insertion_single_species(self):
        """Delta +1: ln_ratio = ln fV - ln(N+1)."""
        patch = MockPatch(delta=jnp.ones_like(self.counts.data))
        res = self.ln_ratio(self.state, patch)
        assert isinstance(res.patch, IdPatch)
        out = res.data.data

        ln_f = self.log_fug.data[0]
        ln_V = jnp.log(self.volume.data)
        ln_fV = ln_f + ln_V
        counts = self.counts.data[:, 0]
        patched = counts + 1
        expected = ln_fV + log_factorial_ratio(counts, patched)
        npt.assert_allclose(out, expected, rtol=0, atol=1e-12)

    def test_deletion_single_species(self):
        """Delta -1: ln_ratio = -ln fV + ln N (only compare N>0)."""
        c = self.counts.data
        patch = MockPatch(delta=-jnp.where(c > 0, 1, 0))
        res = self.ln_ratio(self.state, patch)
        out = res.data.data

        ln_f = self.log_fug.data[0]
        ln_V = jnp.log(self.volume.data)
        counts = c[:, 0]
        patched = counts - 1
        mask = counts > 0
        expected_masked = -(ln_f + ln_V)[mask] + log_factorial_ratio(
            counts[mask], patched[mask]
        )
        npt.assert_allclose(out[mask], expected_masked, rtol=0, atol=1e-12)

    def test_multispecies_mixed_changes(self):
        """Two species; mixed +/- per system; sum over species."""
        log_fug = Table.arange(
            jnp.log(jnp.array([[2.0, 5.0], [2.0, 5.0]])), label=SystemId
        )
        volume = Table.arange(jnp.array([1.5, 0.5]), label=SystemId)
        counts = Table.arange(jnp.array([[2, 3], [4, 1]]), label=SystemId)
        state = MockState(log_fug, volume, counts)

        delta = jnp.array([[+1, -1], [-1, +1]])
        patch = MockPatch(delta=delta)

        ln_ratio = LogFugacityRatio(
            log_activity=get_log_fugacity,
            volume=get_volume,
            particle_counts=get_counts,
            patched_particle_counts=get_patched_counts,
        )

        res = ln_ratio(state, patch)
        out = res.data.data

        ln_V = jnp.log(volume.data)[:, None]
        ln_fV = log_fug.data + ln_V
        patched_counts = counts.data + delta
        expected_mat = delta * ln_fV + log_factorial_ratio(counts.data, patched_counts)
        expected = expected_mat.sum(axis=-1)
        npt.assert_allclose(out, expected, rtol=0, atol=1e-12)
