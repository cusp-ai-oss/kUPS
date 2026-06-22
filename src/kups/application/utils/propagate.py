# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared propagation utilities for simulation loops.

Provides warmup, sampling, and data-parallelism helpers used across
MD, MCMC, and relaxation application modules.
"""

import logging
from operator import itemgetter
from typing import Any, Callable, Protocol

import tqdm
from jax import Array

from kups.core.logging import Logger
from kups.core.propagator import (
    LoopPropagator,
    Propagator,
    propagate_and_fix,
    propagator_with_assertions,
)
from kups.core.result import Result, as_result_function
from kups.core.utils.jax import jit, key_chain

__all__ = [
    "propagate_and_fix",
    "propagator_with_assertions",
    "make_cycle_function",
    "make_block_function",
]


class CycleFunction[State](Protocol):
    def __call__(self, key: Array, state: State, /) -> Result[State, State]: ...


class BlockedCycleFunction[State](Protocol):
    def __call__(
        self, key: Array, state: State, /
    ) -> Result[State, tuple[State, list[Array]]]: ...


def make_cycle_function[State](propagator: Propagator[State]) -> CycleFunction[State]:
    """JIT a propagator into a reusable per-cycle function with state donation.

    Pass the result as ``cycle_fn`` to both :func:`run_warmup_cycles` and
    :func:`run_simulation_cycles` so a single traced-and-compiled program is
    shared across the warmup and sampling phases.

    Args:
        propagator: Step propagator to compile.

    Returns:
        A jitted ``(key, state) -> Result`` cycle function.
    """
    return jit(as_result_function(propagator), donate_argnums=(1,))


def make_block_function[State](
    propagator: Propagator[State],
    block_size: int,
    view: Callable[[Any], Any],
) -> BlockedCycleFunction[State]:
    """JIT ``block_size`` propagator steps fused into one dispatch, capturing each step's
    ``view``. Pass to :func:`run_simulation_cycles` with ``block_size`` set.
    """
    runner = LoopPropagator(propagator, block_size).scan_with(view)
    return jit(as_result_function(runner), donate_argnums=(1,))


def run_warmup_cycles[State](
    key: Array, cycle_fn: CycleFunction[State], state: State, num_cycles: int
) -> State:
    """Run warmup propagation cycles without logging.

    Args:
        key: JAX PRNG key.
        cycle_fn: Compiled per-cycle function from :func:`make_cycle_function`.
        state: Initial simulation state.
        num_cycles: Number of warmup steps.

    Returns:
        State after warmup.
    """
    chain = key_chain(key)
    for _ in tqdm.trange(num_cycles):
        state = propagate_and_fix(cycle_fn, next(chain), state)
    return state


def run_simulation_cycles[State](
    key: Array,
    cycle_fn: Callable[[Array, State], Result[State, Any]],
    state: State,
    num_cycles: int,
    logger: Logger[State],
    *,
    block_size: int = 1,
    convergence_fn: Callable[[State], bool] | None = None,
) -> State:
    """Run ``num_cycles`` propagation steps with logging and optional early stopping.

    ``block_size`` steps are fused per device dispatch (``1`` = per-step), but every step
    is still logged, so the saved trajectory is identical and ``block_size`` is a pure
    performance knob. ``num_cycles`` must be a multiple of ``block_size``.

    Args:
        key: JAX PRNG key for stochastic propagators (e.g. MD thermostats).
        cycle_fn: Per-step (:func:`make_cycle_function`) or blocked
            (:func:`make_block_function`) cycle function.
        state: Initial state.
        num_cycles: Total number of steps (a multiple of ``block_size``).
        logger: Receives each step via ``log``, or each fused block via ``log_block``.
        block_size: Steps fused per dispatch; ``1`` selects the per-step path.
        convergence_fn: If provided, checked after each step/block; stops early when True.

    Returns:
        State after all steps or early convergence.
    """
    if block_size < 1:
        raise ValueError(
            f"block_size ({block_size}) must be a strictly positive integer."
        )
    if num_cycles % block_size != 0:
        raise ValueError(
            f"num_cycles ({num_cycles}) must be a multiple of block_size ({block_size})"
        )
    chain = key_chain(key)
    # range step widens to block_size; a blocked cycle_fn returns (state, frames) and
    # itemgetter(0) feeds the state back into the fix loop, a per-step one returns state.
    blocked = block_size > 1
    with logger:
        for start in range(0, num_cycles, block_size):
            if blocked:
                state, frames = propagate_and_fix(
                    cycle_fn, next(chain), state, state_of=itemgetter(0)
                )
                logger.log_block(frames, start)
            else:
                state = propagate_and_fix(cycle_fn, next(chain), state)
                logger.log(state, start)
            if convergence_fn is not None and convergence_fn(state):
                logging.info("Converged at step %d", start + block_size)
                break
    return state
