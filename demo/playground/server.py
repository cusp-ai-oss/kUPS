"""FastAPI server streaming live kUPS simulations to the playground UI.

Run from ``demo/``: ``uv run python -m playground.server`` (see README).
"""

from __future__ import annotations

import os

# Models come from the cache filled by ``playground.prepare``: never touch the
# network at the booth.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import asyncio
import gzip
import json
import logging
import queue
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, override

import jax
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from . import Refused, arg, configure_jax
from .gcmc import GrandCanonicalMC
from .hybrid import Hybrid
from .md import (
    COLD,
    ENSEMBLES,
    FORCEFIELDS,
    INTEGRATORS,
    POTENTIALS,
    Blocks,
    MolecularDynamics,
    potential_cards,
    use_forcefield,
)

METHODS = {"md": "MD", "gcmc": "GCMC", "hybrid": "Hybrid"}
MAX_FPS = 30
BUILD = str(time.time_ns())  # pages from an earlier server run reload on connect
log = logging.getLogger("playground")


def warming_text(
    warming: frozenset[str], cold: frozenset[Blocks], md: Blocks
) -> str | None:
    """What is still compiling: GCMC, potentials whose cards still say warming
    for MD's integrator, then how many MD setups are left."""
    names = ["GCMC"] * ("gcmc" in warming) + [
        card["label"]
        for p, card in POTENTIALS.items()
        if p in warming or replace(md, potential=p) in cold
    ]
    n = len(cold)
    names += [f"{n} MD setup{'s' * (n > 1)}"] * (n > 0)
    return f"Warming {', '.join(names)}" if names else None


class Recorder:
    """Writes every broadcast and command as timestamped gzip JSONL, for replay."""

    def __init__(self, path: Path):
        self.file = gzip.open(path, "wt")
        self.lock = threading.Lock()  # warm-up threads broadcast too
        self.t0 = time.perf_counter()
        self.write("meta", {"device": jax.devices()[0].device_kind})

    def write(self, kind: str, body: dict[str, Any]) -> None:
        line = json.dumps({"t": round(time.perf_counter() - self.t0, 3), kind: body})
        with self.lock:
            if not self.file.closed:
                self.file.write(line + "\n")

    def close(self) -> None:
        with self.lock:
            self.file.close()


class Runner(threading.Thread):
    """Owns the live simulations; the only thread that steps them.

    MD and GCMC share one UiO-66 + CO₂ structure: switching hands it over, so
    each method continues from where the other stopped. Goes live with MD on
    the default potential; background threads build GCMC and compile every MD
    block combination. Combinations still warming can't be selected. Hybrid
    alternates the very same MD and GCMC objects.
    """

    def __init__(self, prewarm: bool, forcefield: str):
        super().__init__(daemon=True)
        self.prewarm = prewarm
        self.forcefield = forcefield
        self.commands: queue.Queue[Any] = queue.Queue()  # JSON, refused unless a dict
        self.clients: set[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]
        ] = set()
        self.config: dict[str, Any] | None = None
        self.status: str | None = "Starting…"
        self.background: str | None = None
        self.paused = False
        self.methods: dict[str, Any] = {}
        self.active = "md"
        # MD goes live on the first potential; the others warm in the background.
        self.warming = {"gcmc", *list(POTENTIALS)[1:]} if prewarm else {"gcmc"}
        if prewarm:
            COLD.update(Blocks(p, *k) for p in POTENTIALS for k in INTEGRATORS)
        # What the pages last heard is warming; the warm-up threads change it,
        # this thread publishes it.
        self.published: tuple[frozenset[str], frozenset[Blocks], Blocks] | None = None
        self.recorder: Recorder | None = None

    def record(self, path: Path) -> None:
        self.recorder = Recorder(path)
        for message in self.snapshot():
            self.recorder.write("msg", message)

    def snapshot(self) -> list[dict[str, Any]]:
        """What a page needs to catch up: the config and both status lines."""
        status = {"type": "status", "busy": self.status, "background": self.background}
        return [self.config, status] if self.config else [status]

    def send(self, message: dict[str, Any], droppable: bool = False) -> None:
        if self.recorder:
            self.recorder.write("msg", message)
        for loop, q in list(self.clients):
            if droppable and q.qsize() > 2:
                continue  # slow client: skip frames, never commands
            loop.call_soon_threadsafe(q.put_nowait, message)

    def set_status(self, status: str | None) -> None:
        self.status = status
        self.send({"type": "status", "busy": status})

    def set_background(self, text: str | None) -> None:
        self.background = text
        self.send({"type": "status", "background": text})

    def publish_config(self) -> None:
        self.config = {
            "type": "config",
            "build": BUILD,
            "method": self.active,
            "paused": self.paused,
            "warming": sorted(self.warming),
            "warming_blocks": [vars(b) for b in sorted(COLD, key=str)],
            "potentials": potential_cards(),
        } | self.methods[self.active].ui()
        self.send(self.config)

    def handle(self, cmd: Any) -> None:
        if self.recorder:
            self.recorder.write("cmd", cmd)
        if not isinstance(cmd, dict):
            raise Refused("Bad command")
        cmd.pop("say", None)  # the tour's caption, kept for the recording
        if not cmd.pop("tour", False):  # a person took over: stop every page's tour
            self.send({"type": "event", "kind": "takeover"})
        match arg(cmd, "type", str):
            case "method":
                method = arg(cmd, "value", str)
                if method not in METHODS:
                    raise Refused(f"Unknown method {method!r}")
                if method not in self.methods:
                    raise Refused(f"{METHODS[method]} is still warming up")
                if method != self.active:
                    self.switch(method)
            case "pause":
                self.paused = arg(cmd, "value", bool)
            case _:
                label = self.methods[self.active].apply(cmd, self.set_status)
                self.set_status(None)
                if label:
                    self.send({"type": "event", "kind": "swap", "text": label})
                if cmd["type"] == "reset":
                    self.send({"type": "event", "kind": "reset", "text": "Reset"})
        self.publish_config()

    def switch(self, method: str) -> None:
        """Hand the current structure to ``method``, which continues from it;
        the toast names the temperature if it differs or Hybrid retargeted MD,
        and the thermostat if Hybrid replaced NVE."""
        before, md = self.methods[self.active], self.methods["md"]
        temperatures = [m.temperature for m in self.methods.values()]
        thermostat = md.blocks.thermostat
        structure = before.structure()
        n = len(structure.molecules)
        self.set_status(f"Handing {n} CO₂ to {METHODS[method]}…")
        after = self.methods[method]
        after.load(structure)
        self.set_status(None)
        self.active = method
        text = f"{METHODS[method]} · {n} CO₂ kept"
        if after.temperature != before.temperature or temperatures != [
            m.temperature for m in self.methods.values()
        ]:
            text += f" · {after.temperature:g} K"
        if md.blocks.thermostat != thermostat:
            text += f" · {ENSEMBLES[md.blocks.integrator].split(' · ')[-1]}"
        self.send({"type": "event", "kind": "method", "text": text})

    def ready(self, name: str) -> None:
        log.info("%s ready after %.0f s", name, time.perf_counter() - self.started)
        self.warming.discard(name)

    def publish_warming(self) -> None:
        """Republish the config and warming line when warm-up made progress."""
        warming = frozenset(self.warming), frozenset(COLD), self.methods["md"].blocks
        if warming != self.published:
            self.published = warming
            self.publish_config()
            self.set_background(warming_text(*warming))

    def warm_up(self) -> None:
        """Build GCMC and compile MD's blocks, one thread each so loads and XLA
        compiles overlap."""
        threads = [
            threading.Thread(target=self.warm_gcmc, daemon=True),
            *(
                threading.Thread(target=self.warm_md, args=(p,), daemon=True)
                for p in (POTENTIALS if self.prewarm else ())
            ),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.warming.clear()
        log.info("Everything warm after %.0f s", time.perf_counter() - self.started)

    def warm_gcmc(self) -> None:
        gcmc = GrandCanonicalMC(self.forcefield)
        if self.prewarm:
            gcmc.warm_up(lambda _: None)
        self.methods["gcmc"] = gcmc
        self.methods["hybrid"] = Hybrid(self.methods["md"], gcmc)
        self.ready("gcmc")

    def warm_md(self, potential: str) -> None:
        MolecularDynamics().warm_up(lambda _: None, self.ready, [potential])

    @override
    def run(self) -> None:
        self.started = time.perf_counter()
        self.set_status("Loading molecular dynamics…")
        self.methods["md"] = MolecularDynamics()
        self.set_status(None)
        self.publish_warming()
        log.info("Live after %.0f s", time.perf_counter() - self.started)
        threading.Thread(target=self.warm_up, daemon=True).start()
        while True:
            self.publish_warming()
            try:
                while not self.commands.empty() or self.paused:
                    self.handle(self.commands.get(timeout=0.25))
            except queue.Empty:
                continue
            except Exception as e:
                self.recover(e, str(e)[:300])
                continue
            t0 = time.perf_counter()
            try:
                frame = self.methods[self.active].advance()
            except Exception as e:  # diverged, …: start over, keep running
                self.recover(e, f"{str(e)[:200]} · structure reset", reset=True)
                continue
            if frame.pop("republish", False):  # hybrid changed phase: new scene
                self.publish_config()
            self.send({"type": "frame", "method": self.active} | frame, droppable=True)
            time.sleep(max(0.0, 1 / MAX_FPS - (time.perf_counter() - t0)))

    def recover(self, error: Exception, text: str, reset: bool = False) -> None:
        """Report ``error`` and carry on; nothing here may end the thread."""
        if isinstance(error, Refused):
            log.info("Refused: %s", error)
        else:
            log.error("%s", text, exc_info=error)
        self.send({"type": "event", "kind": "error", "text": text})
        try:
            if reset:
                self.methods[self.active].apply({"type": "reset"}, self.set_status)
            self.set_status(None)
            self.publish_config()
        except Exception:
            log.exception("Recovery failed; back to MD")
            self.active, self.published = "md", None
            time.sleep(1)


def make_app(runner: Runner) -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        client = (asyncio.get_running_loop(), outbox)
        runner.clients.add(client)
        for message in runner.snapshot():
            outbox.put_nowait(message)

        async def pump() -> None:
            while True:
                await socket.send_json(await outbox.get())

        sender = asyncio.create_task(pump())
        try:
            while True:
                text = await socket.receive_text()
                try:
                    runner.commands.put(json.loads(text))
                except ValueError:
                    runner.commands.put(None)  # refused as a bad command
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            runner.clients.discard(client)

    @app.get("/live")
    async def live() -> dict[str, str]:  # tells the page a server is behind it
        return {"build": BUILD}

    @app.middleware("http")
    async def revalidate(request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"  # never run a stale app.js
        return response

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help="compile MD blocks on first click instead of in the background",
    )
    parser.add_argument(
        "--forcefield",
        choices=FORCEFIELDS,
        default="ml",
        help="MD potentials: ML models, or a classical force field that runs on any CPU",
    )
    parser.add_argument(
        "--record",
        type=Path,
        metavar="PATH",
        help="write every message to a gzip JSONL file (see playground.recording)",
    )
    args = parser.parse_args()
    # Bind before any JAX/torch thread exists, so a taken port fails fast.
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        raise SystemExit(
            f"Cannot listen on {args.host}:{args.port}: {e.strerror}. Try --port."
        )
    logging.basicConfig(level=logging.INFO)
    configure_jax()
    use_forcefield(args.forcefield)
    runner = Runner(prewarm=not args.no_prewarm, forcefield=args.forcefield)
    if args.record:
        runner.record(args.record)
    runner.start()
    print(f"kUPS playground on http://{args.host}:{args.port}", flush=True)
    # No pings: a page starved by rendering would miss them; its /live watchdog
    # reconnects instead.
    config = uvicorn.Config(
        make_app(runner), log_level="warning", ws_ping_interval=None
    )
    # websockets' legacy protocol trips an assertion when a client leaves mid-send;
    # the connection closes cleanly anyway.
    logging.getLogger("uvicorn.error").addFilter(
        lambda record: record.msg != "data transfer failed"
    )
    uvicorn.Server(config).run(sockets=[sock])
    if runner.recorder:
        runner.recorder.close()
    os._exit(0)  # don't wait on JAX/torch worker threads mid-compile


if __name__ == "__main__":
    main()
