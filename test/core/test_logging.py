# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the logging module."""

from __future__ import annotations

import tempfile
from typing import Self
from unittest.mock import MagicMock, patch

from kups.core.lens import view
from kups.core.logging import CompositeLogger, NullLogger, ProfileLogger, TqdmLogger

from ..clear_cache import clear_cache  # noqa: F401


class CounterLogger:
    """Simple logger that records calls for testing."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.logs: list[tuple[str, int]] = []

    def __enter__(self) -> Self:
        self.entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.exited = True

    def log(self, state: str, step: int) -> None:
        self.logs.append((state, step))


class TestNullLogger:
    def test_context_manager(self) -> None:
        logger: NullLogger[str] = NullLogger()
        with logger as l:
            assert l is logger
            l.log("state", 0)


class TestCompositeLogger:
    def test_enters_all_loggers(self) -> None:
        a, b = CounterLogger(), CounterLogger()
        with CompositeLogger(a, b):
            assert a.entered and b.entered

    def test_exits_in_reverse_order(self) -> None:
        order: list[str] = []

        class OrderLogger(CounterLogger):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

            def __exit__(self, *exc: object) -> None:
                order.append(self.name)
                super().__exit__(*exc)

        a, b, c = OrderLogger("a"), OrderLogger("b"), OrderLogger("c")
        with CompositeLogger(a, b, c):
            pass
        assert order == ["c", "b", "a"]

    def test_fans_out_log_calls(self) -> None:
        a, b = CounterLogger(), CounterLogger()
        composite = CompositeLogger(a, b)
        composite.log("x", 0)
        composite.log("y", 1)
        assert a.logs == [("x", 0), ("y", 1)]
        assert b.logs == [("x", 0), ("y", 1)]

    def test_mixed_null_and_counter(self) -> None:
        null: NullLogger[str] = NullLogger()
        counter = CounterLogger()
        with CompositeLogger(null, counter) as composite:
            composite.log("s", 5)
        assert counter.entered and counter.exited
        assert counter.logs == [("s", 5)]


class TestTqdmLogger:
    def test_progress_bar_updates(self) -> None:
        logger: TqdmLogger[str] = TqdmLogger(num_steps=3)
        with logger:
            assert logger._pbar is not None
            for i in range(3):
                logger.log("state", i)
            assert logger._pbar.n == 3
        assert logger._pbar is None

    def test_postfix_view(self) -> None:
        postfix = view(lambda s: {"val": s})
        logger: TqdmLogger[float] = TqdmLogger(num_steps=2, postfix=postfix)
        with logger:
            assert logger._pbar is not None
            with patch.object(logger._pbar, "set_postfix") as mock_postfix:
                logger.log(3.14, 0)
                mock_postfix.assert_called_once_with({"val": 3.14})

    def test_log_without_enter_is_noop(self) -> None:
        logger: TqdmLogger[str] = TqdmLogger(num_steps=5)
        logger.log("state", 0)


class TestProfileLogger:
    def test_traces_between_start_and_end(self) -> None:
        """Trace starts at start_step-1 and stops at end_step."""
        with tempfile.TemporaryDirectory() as tmp:
            logger: ProfileLogger[str] = ProfileLogger(tmp, start_step=3, end_step=5)
            with patch("kups.core.logging.jax.profiler.trace") as mock_trace:
                ctx = MagicMock()
                mock_trace.return_value = ctx
                with logger:
                    for i in range(8):
                        logger.log("s", i)
                mock_trace.assert_called_once_with(tmp)
                ctx.__enter__.assert_called_once()
                ctx.__exit__.assert_called_once()

    def test_step_zero_starts_in_enter(self) -> None:
        """start_step=0 begins the trace in __enter__."""
        with tempfile.TemporaryDirectory() as tmp:
            logger: ProfileLogger[str] = ProfileLogger(tmp, start_step=0, end_step=2)
            with patch("kups.core.logging.jax.profiler.trace") as mock_trace:
                ctx = MagicMock()
                mock_trace.return_value = ctx
                with logger:
                    # Trace should already be active before any log call.
                    ctx.__enter__.assert_called_once()
                    for i in range(4):
                        logger.log("s", i)
                ctx.__exit__.assert_called_once()

    def test_exit_stops_trace_if_still_running(self) -> None:
        """__exit__ stops trace even if end_step was never reached."""
        with tempfile.TemporaryDirectory() as tmp:
            logger: ProfileLogger[str] = ProfileLogger(tmp, start_step=1, end_step=100)
            with patch("kups.core.logging.jax.profiler.trace") as mock_trace:
                ctx = MagicMock()
                mock_trace.return_value = ctx
                with logger:
                    logger.log("s", 0)  # arms the trace
                    logger.log("s", 1)  # trace is running
                # __exit__ should have stopped the trace.
                ctx.__exit__.assert_called_once()

    def test_noop_when_steps_not_reached(self) -> None:
        """No trace is started if the step range is never reached."""
        with tempfile.TemporaryDirectory() as tmp:
            logger: ProfileLogger[str] = ProfileLogger(tmp, start_step=10, end_step=20)
            with patch("kups.core.logging.jax.profiler.trace") as mock_trace:
                with logger:
                    for i in range(5):
                        logger.log("s", i)
                mock_trace.assert_not_called()
