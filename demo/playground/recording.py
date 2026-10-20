"""Cut one tour loop from a ``server --record`` file into a replayable recording.

Run from ``demo/``::

    uv run python -m playground.recording raw.jsonl.gz

The result, ``static/recordings/tour.json`` and its gzip copy for hosts that
serve it, is what the page replays when the live server is out of reach (or
with ``?replay``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import zlib
from pathlib import Path
from typing import Any

OUT = Path(__file__).parent / "static" / "recordings" / "tour.json"
ROUNDED = {"positions", "batch"}  # Å: three decimals are plenty to draw


def read(path: Path) -> list[dict[str, Any]]:
    """Every complete line, also from a file whose writer was killed."""
    lines: list[str] = []
    with gzip.open(path, "rt") as f:
        try:
            lines.extend(f)
        except (EOFError, zlib.error):
            pass
    return [json.loads(line) for line in lines if line.endswith("\n")]


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def compact(msg: dict[str, Any]) -> dict[str, Any]:
    return {
        k: compact(v) if isinstance(v, dict) else rounded(v) if k in ROUNDED else v
        for k, v in msg.items()
    }


def tour_loop(raw: list[dict[str, Any]], fps: float) -> tuple[list[list[Any]], float]:
    """``[t, "msg" | "cmd", body]`` from one tour start to the next, starting
    with the config and status in force at the cut, and the loop's duration."""
    starts = [i for i, e in enumerate(raw) if "say" in e.get("cmd", {})]
    if not starts:
        raise SystemExit("No tour in this recording: play it with the tour button.")
    first = raw[starts[0]]["cmd"]["say"]
    again = [i for i in starts[1:] if raw[i]["cmd"]["say"] == first]
    if not again:
        raise SystemExit("The tour never restarted: record one full loop.")
    begin, end = starts[0], again[0]

    config, status = None, {"type": "status"}
    for e in raw[:begin]:
        msg = e.get("msg", {})
        if msg.get("type") == "config":
            config = msg
        elif msg.get("type") == "status":
            status |= msg
    t0 = raw[begin]["t"]
    events: list[list[Any]] = [[0.0, "msg", m] for m in (config, status) if m]
    shown: dict[str, float] = {}
    for e in raw[begin:end]:
        t = round(e["t"] - t0, 3)
        if "cmd" in e:
            events.append(
                [t, "cmd", {k: v for k, v in e["cmd"].items() if k != "tour"}]
            )
            continue
        msg = e["msg"]
        if msg["type"] == "frame":
            if t - shown.get(msg["method"], -1.0) < 1 / fps - 1e-3:
                continue
            shown[msg["method"]] = t
        events.append([t, "msg", msg])
    for _, _, body in events:
        if body.get("type") == "config":
            body["build"] = "recording"
    duration = round(raw[end]["t"] - t0, 3)
    return [[t, kind, compact(body)] for t, kind, body in events], duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="file written by server --record")
    parser.add_argument("-o", "--out", type=Path, default=OUT)
    parser.add_argument("--fps", type=float, default=15, help="max frames per second")
    args = parser.parse_args()
    raw = read(args.raw)
    device = next((e["meta"]["device"] for e in raw if "meta" in e), None)
    events, duration = tour_loop(raw, args.fps)
    recording = {"device": device, "duration": duration, "events": events}
    text = json.dumps(recording, separators=(",", ":"))
    packed = args.out.with_name(args.out.name + ".gz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    packed.write_bytes(gzip.compress(text.encode(), compresslevel=9, mtime=0))
    frames = sum(body.get("type") == "frame" for _, _, body in events)
    print(
        f"{args.out}(.gz): {duration:.0f} s, {frames} frames, "
        f"{args.out.stat().st_size / 1e6:.1f} MB ({packed.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
