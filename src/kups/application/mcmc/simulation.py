# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Generic simulation loop for rigid-body MCMC simulations."""

from __future__ import annotations

import logging

import jax
from jax import Array

from kups.application.mcmc.data import RunConfig
from kups.application.mcmc.logging import IsMCMCState, MCMCLoggedData, MCMCStepData
from kups.application.utils.propagate import (
    make_sampled_cycle_function,
    run_sampled_cycles,
)
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.propagator import Propagator
from kups.core.storage import HDF5StorageWriter, WriterGroupConfig
from kups.core.utils.jax import key_chain


def run_mcmc[State: IsMCMCState](
    key: Array,
    propagator: Propagator[State],
    state: State,
    config: RunConfig,
    logged_data: MCMCLoggedData[State],
) -> State:
    """Run a µVT MCMC simulation with warmup and production phases.

    Args:
        key: JAX PRNG key.
        propagator: Propagator, e.g. from :func:`~kups.application.simulations.mcmc_rigid.make_propagator`.
        state: Initial simulation state.
        config: Run configuration.
        logged_data: Logging configuration with host/adsorbate split.

    Returns:
        Final simulation state after production run.
    """

    def postfix(sample: MCMCStepData) -> dict[str, str]:
        return {"Loading": str(sample.particle_count.data.sum())}

    chain = key_chain(key)
    cycle_fn = make_sampled_cycle_function(propagator, logged_data.per_step.view)
    logging.info("Warming up (%d cycles)...", config.num_warmup_cycles)
    state = run_sampled_cycles(
        next(chain),
        cycle_fn,
        state,
        config.num_warmup_cycles,
        TqdmLogger(config.num_warmup_cycles),
        config.cycles_per_call,
    )
    logging.info("Production run (%d cycles)...", config.num_cycles)
    # The fixed snapshot must outlive donation of the simulation buffers.
    fixed = jax.tree.map(
        lambda array: array.copy(), jax.device_get(logged_data.fixed.view(state))
    )
    sample_config = MCMCLoggedData[MCMCStepData](
        fixed=WriterGroupConfig(
            lambda _: fixed,
            logged_data.fixed.logging_frequency,
            compression=logged_data.fixed.compression,
        ),
        per_step=WriterGroupConfig(
            lambda sample: sample,
            logged_data.per_step.logging_frequency,
            compression=logged_data.per_step.compression,
        ),
    )
    logger = CompositeLogger[MCMCStepData](
        HDF5StorageWriter(
            config.out_file,
            sample_config,
            logged_data.per_step.view(state),
            config.num_cycles,
            compile_views=False,
        ),
        TqdmLogger(config.num_cycles, postfix=postfix),
    )
    state = run_sampled_cycles(
        next(chain),
        cycle_fn,
        state,
        config.num_cycles,
        logger,
        config.cycles_per_call,
    )
    logging.info("Done.")
    return state
