"""Hybrid MD/GCMC: flexible-framework adsorption from two unchanged kUPS methods.

GCMC exchanges CO₂ with the reservoir in a rigid snapshot of the framework;
MD then lets framework and guests relax together. The
loop is plain Python around the same MD and GCMC objects the other modes use.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from . import Refused, Status
from .gcmc import GrandCanonicalMC, Isotherm
from .md import ENSEMBLES, POTENTIALS, MolecularDynamics
from .structure import Structure

GCMC_CYCLES = 20  # per round
MD_STEPS = 100  # per round: 50 fs
EQUILIBRATION_ROUNDS = (
    2  # after a condition change, before rounds count towards the isotherm
)
CHARTS = [
    {
        "key": "loading",
        "title": "CO₂ loading",
        "unit": "mmol/g",
        "digits": 2,
        "series": ["Loading"],
    },
    {
        "key": "volume",
        "title": "Cell volume",
        "unit": "Å³",
        "digits": 0,
        "series": ["V"],
    },
    {
        "key": "temperature",
        "title": "Temperature",
        "unit": "K",
        "digits": 0,
        "series": ["T"],
    },
    {
        "key": "isotherm",
        "title": "Isotherm",
        "unit": "mmol/g",
        "digits": 2,
        "series": ["Rigid", "Flexible"],
        "kind": "scatter",
        "x_unit": "bar",
    },
]


class Hybrid:
    """Alternate GCMC cycles and MD steps on one structure; drive it from a single thread."""

    def __init__(self, md: MolecularDynamics, gcmc: GrandCanonicalMC):
        self.md, self.gcmc = md, gcmc
        self.isotherm = Isotherm(EQUILIBRATION_ROUNDS * GCMC_CYCLES)
        self.rounds = 0.0
        self.phase = "gcmc"
        self.frames = 0

    @property
    def current(self) -> MolecularDynamics | GrandCanonicalMC:
        return self.md if self.phase == "md" else self.gcmc

    @property
    def md_frames(self) -> int:
        return math.ceil(MD_STEPS / self.md.steps_per_frame)

    @property
    def md_steps(self) -> int:
        return self.md_frames * self.md.steps_per_frame

    @property
    def temperature(self) -> float:
        return self.gcmc.temperature

    # -- structure hand-over -----------------------------------------------
    def load(self, structure: Structure) -> None:
        """Start a round from ``structure``: GCMC first, at GCMC's conditions."""
        if self.md.blocks.thermostat == "none":
            self.md.update(replace(self.md.blocks, thermostat="langevin"))
        self.md.set_temperature(self.gcmc.temperature)
        self.phase, self.frames = "gcmc", 0
        self.gcmc.load(structure)

    def structure(self) -> Structure:
        return self.current.structure()

    def _hand_over(self) -> None:
        structure = self.current.structure()
        self.phase, self.frames = ("md" if self.phase == "gcmc" else "gcmc"), 0
        self.current.load(structure)

    # -- method interface ------------------------------------------------
    def apply(self, cmd: dict[str, Any], status: Status) -> str | None:
        match cmd["type"]:
            case "blocks":
                if cmd.get("thermostat") == "none":
                    raise Refused("Hybrid needs a thermostat")
                return self.md.apply(cmd, status)
            case "moves":
                return self.gcmc.apply(cmd, status)
            case "pressure":
                label = self.gcmc.apply(cmd, status)
                if label:
                    self.isotherm.since_change = 0
                return label
            case "temperature":
                label = self.gcmc.apply(cmd, status)
                if label:
                    self.isotherm = Isotherm(self.isotherm.equilibration)
                    self.md.set_temperature(self.gcmc.temperature)
                return label
            case "reset":
                self.md.apply(cmd, status)
                self.gcmc.apply(cmd, status)
                self.isotherm = Isotherm(self.isotherm.equilibration)
                self.rounds = 0.0
                self.phase, self.frames = "gcmc", 0
        return None

    def ui(self) -> dict[str, Any]:
        md, gcmc = self.md.ui(), self.gcmc.ui()
        scene = md if self.phase == "md" else gcmc
        blocks = self.md.blocks
        potential = POTENTIALS[blocks.potential]["label"]
        ensemble = ENSEMBLES[blocks.integrator].split(" · ")[0]
        return (
            md
            | gcmc
            | {k: scene[k] for k in ("symbols", "static_atoms")}
            | {
                "phase": self.phase,
                "pipeline": [
                    {
                        "name": "GCMC",
                        "detail": f"{GCMC_CYCLES} cycles · {self.gcmc.pressure_bar:g} bar · rigid host",
                        "active": self.phase == "gcmc",
                    },
                    {
                        "name": "MD",
                        "detail": f"{self.md_steps} steps · {potential} · {ensemble}",
                        "active": self.phase == "md",
                    },
                ],
                "pipeline_head": "while True: gcmc → md, one shared structure",
                "listing": [
                    f'md_step = make_md_propagator(lens, "{blocks.integrator}", {blocks.potential})',
                    "gcmc_cycle = make_gcmc_propagator(lens, moves, …)",
                    f"# {self.gcmc.pressure_bar:g} bar reservoir, {self.gcmc.temperature:g} K",
                    "",
                    "while booth_is_open:  # flexible-framework GCMC",
                    "    gcmc.load(structure)       # host frozen",
                    f"    for _ in range({GCMC_CYCLES}):         # CO₂ in and out",
                    "        gcmc.state = gcmc_cycle(key, gcmc.state)",
                    "    md.load(gcmc.structure())  # all atoms move",
                    f"    for _ in range({self.md_steps}):        # host relaxes",
                    "        md.state = md_step(key, md.state)",
                    "    structure = md.structure()",
                ],
                "charts": CHARTS,
            }
        )

    def advance(self) -> dict[str, Any]:
        """One frame of the current phase; hand over first when the phase is done."""
        republish = False
        length = GCMC_CYCLES if self.phase == "gcmc" else self.md_frames
        if self.frames >= length:
            self._hand_over()
            republish = True
            length = GCMC_CYCLES if self.phase == "gcmc" else self.md_frames
        # Each chart follows the phase that changes it, on that phase's own clock:
        # loading per GCMC cycle, volume and temperature per MD picosecond.
        frame = (
            self.gcmc.advance(self.isotherm)
            if self.phase == "gcmc"
            else self.md.advance()
        )
        charts, md_time = frame["charts"], [self.md.time_fs / 1000, "ps"]
        self.frames += 1
        self.rounds += 0.5 / length
        rigid, flexible = self.gcmc.rigid_isotherm(), self.isotherm.means()
        return frame | {
            "republish": republish,
            "x": self.rounds,
            "x_unit": "rounds",
            "charts": {
                "loading": charts.get("loading"),
                "volume": charts.get("volume"),
                "temperature": charts.get("temperature"),
                "isotherm": [
                    [p, rigid.get(p), flexible.get(p)]
                    for p in sorted(rigid.keys() | flexible.keys())
                ],
            },
            "chart_x": {
                "loading": [self.gcmc.cycles, "cycles"],
                "volume": md_time,
                "temperature": md_time,
            },
            "reference": {"temperature": self.gcmc.temperature},
        }
