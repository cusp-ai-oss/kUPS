"""Compiled cycle functions kept on disk, so a restart skips tracing them.

Tracing a kUPS propagator takes 10-30 s even when XLA's cache holds the binary.
``cycle_function`` saves each executable with the pytree structures around it,
keyed by the code, the program's name and the argument shapes; later starts
load it in about a second. Programs calling back into PyTorch can't be saved
and are traced as usual.
"""

from __future__ import annotations

import copyreg
import functools
import hashlib
import logging
import pickle
import re
import threading
import typing
from pathlib import Path
from typing import Any, Callable

import cloudpickle
import jax
from jax.experimental import serialize_executable

import kups
from kups.core.propagator import Propagator
from kups.core.result import Result, as_result_function
from kups.core.utils.jax import enable_post_init, no_post_init

from . import CACHE, EXAMPLES

log = logging.getLogger(__name__)
# Assertion fixes in the output structure close over generic functions.
for _type in (typing.TypeVar, typing.ParamSpec, typing.TypeVarTuple):
    copyreg.pickle(_type, lambda t, cls=_type: (cls, (t.__name__,)))


@functools.cache
def _dir() -> Path:
    """Changes with kUPS, the playground, their data, JAX and the device."""
    digest = hashlib.sha256(
        f"{jax.__version__} {jax.devices()[0].device_kind}".encode()
    )
    for root in (Path(kups.__file__).parent, Path(__file__).parent, EXAMPLES):
        for path in sorted(root.rglob("*")):
            if path.suffix in (".py", ".yaml", ".cif"):
                digest.update(path.read_bytes())
    return CACHE / "aot" / digest.hexdigest()[:16]


# An executable XLA's own cache served doesn't serialize again on CPU. Naming
# the traced function after the code version makes new programs miss that cache.
_hits = threading.local()


def _note_hit(event: str, **_: Any) -> None:
    if event.endswith("cache_hits"):
        _hits.hit = True


jax.monitoring.register_event_listener(_note_hit)


def cycle_function[S](propagator: Propagator[S], name: str) -> Any:
    """``make_cycle_function(propagator)``, loaded from disk when possible.

    ``name`` must tell apart every propagator built for the same state shapes.
    """
    traced = _post_init(as_result_function(propagator), f"cycle_{_dir().name}")
    jitted = jax.jit(traced, donate_argnums=(1,))
    programs: dict[Any, Any] = {}
    lock = threading.Lock()

    def program(key: jax.Array, state: S) -> Any:
        leaves, tree = jax.tree.flatten((key, state))
        signature = tree, tuple(map(jax.typeof, leaves))
        with lock:
            if signature not in programs:
                programs[signature] = _load_or_compile(
                    jitted, name, signature, key, state
                )
            return programs[signature]

    def run(key: jax.Array, state: S) -> Result[S, S]:
        with no_post_init():
            return program(key, state)(key, state)

    return run


def _load_or_compile(jitted: Any, name: str, signature: Any, *args: Any) -> Any:
    # Functions in static fields print their address: drop it.
    text = re.sub(r" at 0x[0-9a-f]+", "", f"{name} {signature}")
    path = _dir() / f"{hashlib.sha256(text.encode()).hexdigest()[:24]}.pkl"
    with no_post_init():
        if path.exists():
            try:
                return serialize_executable.deserialize_and_load(
                    *pickle.loads(path.read_bytes())
                )
            except Exception as e:  # stale or truncated: compile anew
                log.warning("Recompiling %s: %s", name, e)
        lowered = jitted.trace(*args).lower()
        _hits.hit = False
        compiled = lowered.compile()
        if _hits.hit:
            return compiled
        try:
            blob = cloudpickle.dumps(serialize_executable.serialize(compiled))
        except Exception as e:  # e.g. host callbacks into PyTorch
            log.info("Not saving %s: %s", name, type(e).__name__)
            return compiled
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{threading.get_ident()}")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return compiled


def jit[F: Callable[..., Any]](fn: F) -> F:
    """``jax.jit`` that validates while tracing, so it can be called under
    ``no_post_init``."""
    return jax.jit(_post_init(fn, fn.__name__))  # type: ignore[return-value]


def _post_init(fn: Any, name: str) -> Any:
    def traced(*args: Any) -> Any:
        with enable_post_init():
            return fn(*args)

    traced.__name__ = name
    return traced
