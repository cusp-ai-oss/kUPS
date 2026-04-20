# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Generic simulation loop for rigid-body MCMC simulations."""

from __future__ import annotations

import logging

from jax import Array

from kups.application.mcmc.data import RunConfig
from kups.application.mcmc.logging import MCMCLoggedData
from kups.application.utils.propagate import run_simulation_cycles, run_warmup_cycles
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.propagator import Propagator
from kups.core.storage import HDF5StorageWriter
from kups.core.utils.jax import key_chain


def run_mcmc[State](
    key: Array,
    propagator: Propagator[State],
    state: State,
    config: RunConfig,
    logged_data: MCMCLoggedData,
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
    chain = key_chain(key)
    logging.info("Warming up (%d cycles)...", config.num_warmup_cycles)
    state = run_warmup_cycles(next(chain), propagator, state, config.num_warmup_cycles)
    logging.info("Production run (%d cycles)...", config.num_cycles)
    logger = CompositeLogger(
        HDF5StorageWriter(config.out_file, logged_data, state, config.num_cycles),
        TqdmLogger(config.num_cycles),
    )
    state = run_simulation_cycles(
        next(chain), propagator, state, config.num_cycles, logger
    )
    logging.info("Done.")
    return state
