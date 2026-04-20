# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared propagation utilities for simulation loops.

Provides warmup, sampling, and data-parallelism helpers used across
MD, MCMC, and relaxation application modules.
"""

import logging
from copy import deepcopy
from typing import Any, Callable, Generator

import jax
import numpy as np
import tqdm
from jax import Array

from kups.core.logging import Logger
from kups.core.propagator import (
    Propagator,
    propagate_and_fix,
    propagator_with_assertions,
)
from kups.core.result import Result, as_result_function
from kups.core.utils.jax import jit, key_chain, shard_map

__all__ = [
    "propagate_and_fix",
    "propagator_with_assertions",
    "warmup_and_sample",
    "data_parallelism_vmap",
    "data_parallelism_put",
]


def run_warmup_cycles[State](
    key: Array, propagator: Propagator[State], state: State, num_cycles: int
) -> State:
    """Run warmup propagation cycles without logging.

    Args:
        key: JAX PRNG key.
        propagator: Step propagator.
        state: Initial simulation state.
        num_cycles: Number of warmup steps.

    Returns:
        State after warmup.
    """
    chain = key_chain(key)
    propagator_with_assertion = jit(as_result_function(propagator), donate_argnums=(1,))
    for _ in tqdm.trange(num_cycles):
        state = propagate_and_fix(propagator_with_assertion, next(chain), state)
    return state


def run_simulation_cycles[State](
    key: Array,
    propagator: Propagator[State],
    state: State,
    num_cycles: int,
    logger: Logger[State],
    *,
    convergence_fn: Callable[[State], bool] | None = None,
) -> State:
    """Run simulation steps with logging and optional early stopping.

    Args:
        key: JAX PRNG key for stochastic propagators (e.g. MD thermostats).
        propagator: Step propagator.
        state: Initial state.
        num_cycles: Maximum number of steps.
        logger: Logger receiving state each step.
        convergence_fn: If provided, called after each step; stops early when
            it returns True.

    Returns:
        State after all steps or early convergence.
    """
    chain = key_chain(key)
    prop_with_assertions = jit(as_result_function(propagator), donate_argnums=(1,))
    with logger:
        for i in range(num_cycles):
            state = propagate_and_fix(prop_with_assertions, next(chain), state)
            logger.log(state, i)
            if convergence_fn is not None and convergence_fn(state):
                logging.info("Converged at step %d", i + 1)
                break
    return state


def warmup_and_sample[State](
    chain: Generator[Array, Any, Any],
    propagator: Callable[[Array, State], Result[State, State]],
    state: State,
    warmup_cycles: int,
    num_samples: int,
    *,
    print_progress: bool = True,
) -> list[State]:
    """Warm up and collect evenly-spaced state snapshots.

    Samples ``num_samples`` states from the last 20% of warmup steps.
    Falls back to the final ``num_samples`` steps when 20% is too few.

    Args:
        chain: PRNG key generator.
        propagator: JIT-compiled propagator returning ``Result``.
        state: Initial state.
        warmup_cycles: Total number of propagation steps.
        num_samples: Number of snapshots to collect.
        print_progress: Show a tqdm progress bar.

    Returns:
        List of ``num_samples`` deep-copied state snapshots.

    Raises:
        ValueError: If ``warmup_cycles`` is too small for ``num_samples``.
    """
    # Spread samples over the last 20% of steps; fall back to the tail.
    if warmup_cycles * 0.2 < num_samples:
        take_from = set(range(warmup_cycles - num_samples, warmup_cycles))
    else:
        take_from = set(
            np.linspace(
                warmup_cycles * 0.8,
                warmup_cycles - 1,
                num_samples + 1,
                dtype=int,
            )[1:]
        )
    if len(take_from) < num_samples:
        raise ValueError(
            "Not enough warmup cycles to take the requested number of states."
        )

    states = []
    with tqdm.trange(warmup_cycles, disable=not print_progress) as pbar:
        for i in pbar:
            state = propagate_and_fix(propagator, next(chain), state)
            if i in take_from:
                states.append(deepcopy(state))
    assert len(states) == num_samples
    return states


BATCH_MESH = jax.sharding.Mesh(jax.devices(), ("batch",))
BATCH_P = jax.P("batch")
BATCH_SHARDING = jax.NamedSharding(BATCH_MESH, BATCH_P)


def data_parallelism_vmap[C: Callable](f: C) -> C:
    """Vmap ``f`` with multi-device sharding when available.

    On a single device the function is simply vmapped and JIT-compiled.
    With multiple devices it is additionally shard-mapped across the
    ``"batch"`` mesh axis.
    """
    vmapped_fn = jax.vmap(f)
    if jax.device_count() > 1:
        return jit(
            shard_map(
                vmapped_fn,
                out_specs=BATCH_P,
                in_specs=BATCH_P,
                mesh=BATCH_MESH,
            ),
            donate_argnums=(1,),
        )
    return jit(vmapped_fn, donate_argnums=(1,))


def data_parallelism_put[S](state: S) -> S:
    """Place ``state`` on the batch-sharded mesh."""
    return jax.device_put(state, BATCH_SHARDING)
