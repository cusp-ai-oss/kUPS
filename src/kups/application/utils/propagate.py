# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared propagation utilities for simulation loops.

Provides warmup, sampling, and data-parallelism helpers used across
MD, MCMC, and relaxation application modules.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

import jax
import jax.numpy as jnp
import tqdm
from jax import Array

from kups.core.assertion import check_assertions
from kups.core.lens import View
from kups.core.logging import Logger
from kups.core.propagator import (
    Propagator,
    propagate_and_fix,
    propagator_with_assertions,
)
from kups.core.result import Result, as_result_function
from kups.core.utils.jax import (
    dataclass,
    field,
    jit,
    key_chain,
    no_post_init,
    tree_where_broadcast_last,
)

__all__ = [
    "propagate_and_fix",
    "propagator_with_assertions",
    "make_cycle_function",
]


class CycleFunction[State](Protocol):
    def __call__(self, key: Array, state: State, /) -> Result[State, State]: ...


@dataclass
class CycleSamples[Sample]:
    """Stack sample leaves without adding a time axis to indexed table metadata."""

    arrays: tuple[Array, ...]
    structure: jax.tree_util.PyTreeDef = field(static=True)

    @classmethod
    def from_sample[T](cls, sample: T) -> CycleSamples[T]:
        arrays, structure = jax.tree.flatten(sample)
        return cls(tuple(arrays), structure)

    def __getitem__(self, index: int) -> Sample:
        with no_post_init():
            # Keep zero-dimensional NumPy leaves as arrays when slicing host samples.
            return self.structure.unflatten(
                [array[index, ...] for array in self.arrays]
            )


class SampledCycleFunction[State, Sample](Protocol):
    def __call__(
        self, keys: Array, state: State, /
    ) -> Result[State, tuple[State, CycleSamples[Sample]]]: ...


def make_sampled_cycle_function[State, Sample](
    propagator: Propagator[State], sample: View[State, Sample]
) -> SampledCycleFunction[State, Sample]:
    """Compile consecutive cycles, retaining samples instead of full state histories."""

    def block(keys: Array, state: State) -> tuple[State, CycleSamples[Sample]]:
        def step(current: State, key: Array) -> tuple[State, CycleSamples[Sample]]:
            current = propagator(key, current)
            return current, CycleSamples.from_sample(sample(current))

        updated, samples = jax.lax.scan(step, state, keys)
        # A capacity repair must retry the entire block with the same keys.
        valid = check_assertions(jax.tree.leaves(updated)[0])
        updated = tree_where_broadcast_last(valid, updated, state)
        return updated, samples

    return jit(as_result_function(block), donate_argnums=(1,))


def run_sampled_cycles[State, Sample](
    key: Array,
    cycle_fn: SampledCycleFunction[State, Sample],
    state: State,
    num_cycles: int,
    logger: Logger[Sample],
    cycles_per_call: int,
) -> State:
    """Dispatch blocks while preserving every cycle's key, sample and logging index."""
    if cycles_per_call < 1:
        raise ValueError("cycles_per_call must be positive")
    chain = key_chain(key)
    with logger:
        for start in range(0, num_cycles, cycles_per_call):
            size = min(cycles_per_call, num_cycles - start)
            keys = jnp.stack([next(chain) for _ in range(size)])
            for _ in range(10):
                result = cycle_fn(keys, state)
                state, samples = result.value
                if not result.failed_assertions:
                    break
                state = result.fix_or_raise(state)
            else:
                raise RuntimeError(
                    "Failed to resolve sampled cycles after multiple attempts"
                )
            samples = jax.device_get(samples)
            for offset in range(size):
                logger.log(samples[offset], start + offset)
    return state


def make_cycle_function[State](propagator: Propagator[State]) -> CycleFunction[State]:
    """JIT a propagator into a reusable per-cycle function with state donation.

    Pass the result as ``cycle_fn`` to both :func:`run_warmup_cycles` and
    :func:`run_simulation_cycles` so a single traced-and-compiled program is shared
    across the warmup and sampling phases. For blocked stepping, compose the propagator
    with :class:`~kups.core.propagator.LoopPropagator` before passing it in.

    Args:
        propagator: Step propagator to compile.

    Returns:
        A jitted ``(key, state) -> Result`` cycle function.
    """
    return jit(as_result_function(propagator), donate_argnums=(1,))


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
    cycle_fn: CycleFunction[State],
    state: State,
    num_cycles: int,
    logger: Logger[State],
    *,
    convergence_fn: Callable[[State], bool] | None = None,
) -> State:
    """Run simulation steps with logging and optional early stopping.

    Args:
        key: JAX PRNG key for stochastic propagators (e.g. MD thermostats).
        cycle_fn: Compiled per-cycle function from :func:`make_cycle_function`.
        state: Initial state.
        num_cycles: Maximum number of steps.
        logger: Logger receiving state each step.
        convergence_fn: If provided, called after each step; stops early when
            it returns True.

    Returns:
        State after all steps or early convergence.
    """
    chain = key_chain(key)
    with logger:
        for i in range(num_cycles):
            state = propagate_and_fix(cycle_fn, next(chain), state)
            logger.log(state, i)
            if convergence_fn is not None and convergence_fn(state):
                logging.info("Converged at step %d", i + 1)
                break
    return state
