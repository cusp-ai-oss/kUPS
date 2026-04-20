# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Generic logging interface for simulation frameworks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, Self

import jax.profiler
import tqdm

from kups.core.lens import View


class Logger[State](Protocol):
    """Unified logging interface for simulation state.

    Implementations control their own filtering (step-based, time-based, etc.).
    State extraction must happen synchronously inside `log()` because JAX
    may donate the state buffer immediately after the call returns.
    """

    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> None: ...
    def log(self, state: State, step: int) -> None: ...


class CompositeLogger[State]:
    """Combines multiple loggers into one.

    Args:
        loggers: Loggers to compose.
    """

    def __init__(self, *loggers: Logger[State]) -> None:
        self._loggers = loggers

    def __enter__(self) -> Self:
        for logger in self._loggers:
            logger.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        for logger in reversed(self._loggers):
            logger.__exit__(*exc)

    def log(self, state: State, step: int) -> None:
        for logger in self._loggers:
            logger.log(state, step)


class NullLogger[State]:
    """No-op logger. Useful for warmup or when logging is disabled."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def log(self, state: State, step: int) -> None:
        pass


class TqdmLogger[State]:
    """Progress bar logger using tqdm.

    Args:
        num_steps: Total number of steps for the progress bar.
        postfix: Optional view extracting a dict from state to display as tqdm postfix.
    """

    def __init__(
        self,
        num_steps: int,
        postfix: View[State, dict[str, Any]] | None = None,
    ) -> None:
        self._num_steps = num_steps
        self._postfix = postfix
        self._pbar: tqdm.tqdm[int] | None = None

    def __enter__(self) -> Self:
        self._pbar = tqdm.trange(self._num_steps)
        self._pbar.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._pbar is not None:
            self._pbar.__exit__(None, None, None)
            self._pbar = None

    def log(self, state: State, step: int) -> None:
        if self._pbar is not None:
            self._pbar.update(1)
            if self._postfix is not None:
                self._pbar.set_postfix(self._postfix(state))


class ProfileLogger[State]:
    """Logger that captures a JAX profiler trace over a range of steps.

    Since ``log(state, step)`` is called *after* step ``step`` has executed,
    the trace is started one step early: the call at ``start_step - 1``
    arms the profiler so that ``start_step`` through ``end_step`` are
    captured.  When ``start_step == 0``, the trace begins in ``__enter__``.

    The resulting trace can be viewed in TensorBoard or Perfetto.

    Args:
        log_dir: Directory to write the profiler trace.
        start_step: First step to trace (inclusive).
        end_step: Last step to trace (inclusive).
    """

    def __init__(
        self,
        log_dir: str | Path,
        start_step: int,
        end_step: int,
    ) -> None:
        self._log_dir = str(log_dir)
        self._start_step = start_step
        self._end_step = end_step
        self._trace: Any | None = None

    def __enter__(self) -> Self:
        if self._start_step == 0:
            self._start_trace()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop_trace()

    def log(self, state: State, step: int) -> None:
        if step == self._start_step - 1:
            self._start_trace()
        elif step == self._end_step - 1:
            self._stop_trace()

    def _start_trace(self) -> None:
        if self._trace is not None:
            return
        logging.info("Starting JAX profiler trace at %s", self._log_dir)
        self._trace = jax.profiler.trace(self._log_dir)
        self._trace.__enter__()

    def _stop_trace(self) -> None:
        if self._trace is None:
            return
        self._trace.__exit__(None, None, None)
        self._trace = None
        logging.info("JAX profiler trace saved to %s", self._log_dir)
