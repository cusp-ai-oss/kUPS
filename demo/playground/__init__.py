"""kUPS booth playground: live, hot-swappable simulations in the browser."""

import os
from pathlib import Path
from typing import Any, Callable

CACHE = Path(
    os.environ.get("KUPS_PLAYGROUND_CACHE", Path.home() / ".cache" / "kups-playground")
)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # share GPU with torch
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(CACHE / "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(CACHE / "triton"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # XLA:CPU's cache-load warnings

import jax  # noqa: E402  # reads the environment above

EXAMPLES = Path(__file__).parents[2] / "examples"
type Status = Callable[[str], None]


class Refused(Exception):
    """A command the playground declines, e.g. a block still warming up."""


def arg[T](cmd: dict[str, Any], key: str, kind: type[T]) -> T:
    """``cmd[key]``, refusing a malformed command."""
    value = cmd.get(key)
    if not isinstance(value, kind):
        raise Refused("Bad command")
    return value


def number(cmd: dict[str, Any], low: float, high: float, unit: str) -> float:
    """``cmd["value"]`` as a number within the UI slider's range."""
    value = cmd.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not low <= value <= high
    ):
        raise Refused(f"Choose {low:g}–{high:g} {unit}")
    return float(value)


def configure_jax() -> None:
    """Cache every compilation persistently.

    Call after importing kUPS applications, which point the cache at ``/tmp``.
    """
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(CACHE / "jax"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
