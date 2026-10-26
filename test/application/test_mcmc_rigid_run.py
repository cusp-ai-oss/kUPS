# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end runs of the rigid-body MCMC entry point."""

import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.application.mcmc.logging import MCMCLoggedData, MCMCStepData
from kups.application.simulations import mcmc_rigid
from kups.application.simulations.mcmc_rigid import MCMCState, init_state, run
from kups.core.data import Index
from kups.core.storage import HDF5StorageReader

from ._rigid_configs import assert_readable, batched_co2_config, co2_config


class TestRunNVT:
    """Canonical (fixed-N) MCMC: exchange disabled, host pre-loaded.

    Neutral CO2 keeps these runs cheap; the GCMC runs cover Ewald.
    """

    init_adsorbates = (2,)

    @pytest.fixture(scope="class")
    def run_result(self) -> tuple[MCMCState, str]:
        config = co2_config(
            exchange_prob=0.0, init_adsorbates=self.init_adsorbates, charged=False
        )
        return run(config), str(config.run.out_file)

    def test_loading_is_conserved(self, run_result):
        state, _ = run_result
        # exchange_prob=0 keeps the molecule count fixed at the initial loading.
        counts = state.groups.data.system.counts.data
        npt.assert_array_equal(counts, self.init_adsorbates)

    def test_analyzer_reads_back_physical_outputs(self, run_result):
        _, out_file = run_result
        assert_readable(out_file)

    def test_no_free_slots_without_insertions(self, run_result):
        state, _ = run_result
        # Without exchange moves no molecule can be inserted: buffers hold
        # exactly the initial molecules.
        assert len(state.groups) == sum(self.init_adsorbates)
        assert len(state.particles) == int(state.particles.occupation.sum())

    def test_empty_host_keeps_one_slot_and_runs(self):
        # At zero pressure the reservoir estimate is 0; one slot is still kept.
        base = co2_config(exchange_prob=0.0, init_adsorbates=(0,), charged=False)
        host = base.hosts[0].model_copy(update={"pressure": 0.0})
        config = base.model_copy(update={"hosts": (host,)})
        state = init_state(jax.random.key(7), config)
        assert len(state.groups) == 1
        assert int(state.groups.occupation.sum()) == 0
        final = run(config)
        assert int(final.groups.data.system.counts.data[0]) == 0

    @pytest.mark.parametrize(
        ("exchange_prob", "loadings", "free_groups"),
        [(0.0, (0, 2), 0), (0.0, (0, 0), 1), (0.5, (0, 2), None)],
    )
    def test_batched_free_slots(self, exchange_prob, loadings, free_groups):
        base = co2_config(
            exchange_prob=exchange_prob, init_adsorbates=(0,), charged=False
        )
        hosts = tuple(
            base.hosts[0].model_copy(update={"init_adsorbates": (n,)}) for n in loadings
        )
        state = init_state(jax.random.key(7), base.model_copy(update={"hosts": hosts}))
        npt.assert_array_equal(state.groups.data.system.counts.data, loadings)
        free = len(state.groups) - int(state.groups.occupation.sum())
        free_particles = len(state.particles) - int(state.particles.occupation.sum())
        if free_groups is None:
            assert free > 0
        else:
            assert free == free_groups
        # Each free molecule slot carries one CO2 worth of particle slots.
        assert free_particles == 3 * free


class TestRunGCMC:
    """Grand-canonical (µVT) MCMC with insertions/deletions and initial loading.

    The stress-logging run is repeated with one and with two cycles per
    compiled call; both must save exactly the same frames. Blocks are kept
    uniform so each run compiles one cycle program; partial blocks are covered
    by ``test_propagate``.
    """

    @pytest.fixture(scope="class")
    def runs(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> list[tuple[MCMCState, Path]]:
        directory = tmp_path_factory.mktemp("gcmc")
        config = co2_config(exchange_prob=0.5, init_adsorbates=(2,))
        results = []
        for block_size in (1, 2):
            output = directory / f"cycles-{block_size}.h5"
            run_config = config.run.model_copy(
                update={
                    "out_file": output,
                    "num_warmup_cycles": 2,
                    "num_cycles": 4,
                    "cycles_per_call": block_size,
                }
            )
            blocked = config.model_copy(
                update={"compute_stress": True, "run": run_config}
            )
            results.append((run(blocked), output))
        return results

    def test_state_has_finite_energy(self, runs):
        for state, _ in runs:
            assert jnp.isfinite(state.systems.data.potential_energy.total[0]).item()

    def test_analyzer_reads_back_physical_outputs(self, runs):
        for _, output in runs:
            assert_readable(str(output))

    def test_blocked_calls_preserve_every_saved_frame(self, runs):
        frames: list[MCMCStepData] = []
        for _, output in runs:
            with HDF5StorageReader[MCMCLoggedData[MCMCStepData]](output) as reader:
                sample = reader.focus_group(lambda cfg: cfg.per_step)[:]
                assert sample.systems.data.potential_energy.shape[0] == 4
                assert reader.file.attrs["actual_steps"] == 4
                frames.append(sample)
        for expected, actual in zip(
            jax.tree.leaves(frames[0]), jax.tree.leaves(frames[1]), strict=True
        ):
            npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_gcmc_with_int32_indices_matches_default() -> None:
    states = []
    for index_dtype in ("int64", "int32"):
        config = batched_co2_config(2, (2,))
        run_config = config.run.model_copy(update={"index_dtype": index_dtype})
        states.append(run(config.model_copy(update={"run": run_config})))
    leaves = jax.tree.leaves(states[1], is_leaf=lambda x: isinstance(x, Index))
    dtypes = {x.raw_indices.dtype for x in leaves if isinstance(x, Index)}
    assert dtypes == {jnp.dtype(jnp.int32)}
    assert states[1].cell_tables.table.slot_of_row.dtype == jnp.int32
    npt.assert_array_equal(
        states[1].groups.data.system.counts.data,
        states[0].groups.data.system.counts.data,
    )
    npt.assert_allclose(
        states[1].systems.data.potential_energy.total,
        states[0].systems.data.potential_energy.total,
        rtol=1e-10,
        atol=1e-10,
    )


def test_seed_zero_is_not_time_based(monkeypatch: pytest.MonkeyPatch) -> None:
    """``seed: 0`` is a fixed seed and does not read the clock."""
    clock = types.SimpleNamespace(time_ns=lambda: pytest.fail("seed 0 read the clock"))
    monkeypatch.setattr(mcmc_rigid, "time", clock)
    monkeypatch.setattr(
        mcmc_rigid, "run_mcmc", lambda key, propagator, state, *_: state
    )
    config = co2_config(exchange_prob=0.0, init_adsorbates=(2,), charged=False)
    run(config.model_copy(update={"run": config.run.model_copy(update={"seed": 0})}))
