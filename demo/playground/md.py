"""Molecular dynamics of UiO-66 + CO₂ whose building blocks swap while it runs.

Every block (potential, thermostat, barostat) maps onto a plain kUPS object. A
swap rebuilds only the affected object; the state carries over.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal

import ase
import jax
import jax.numpy as jnp
import numpy as np

from kups.application.md.data import (
    MdParameters,
    MdState,
    _build_integrator_params,
    md_state_from_ase,
)
from kups.application.md.simulation import make_md_propagator
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.simulations.potentials import TojaxPotentialConfig
from kups.application.utils.path import get_model_path
from kups.core.constants import BOLTZMANN_CONSTANT, PASCAL
from kups.core.data import Index
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.propagator import LoopPropagator, Propagator, propagate_and_fix
from kups.md.observables import instantaneous_pressure, system_kinetic_energy
from kups.observables.stress import stress_via_virial_theorem

from . import Refused, Status, arg, number
from .aot import cycle_function
from .structure import MAX_CO2, Structure, pristine

log = logging.getLogger(__name__)
STATE = identity_lens(MdState)
TIME_STEP = 0.5  # fs
TEMPERATURE = 300.0  # K
COMPRESSIBILITY = 2.5e-11  # 1/Pa, UiO-66
MAX_TEMPERATURE = 5000.0  # K; beyond this a run counts as diverged
SETTLE_FRAMES = 10  # warm-up frames before compiling, so capacities have grown
THERMALISE_FS = 10.0  # a structure without momenta runs hot this long: not charted
TEMPERATURES = (10.0, 600.0)  # K, the slider's range
MACE_PATH = "hf://CuspAI/kUPS-mace-jax/mace-mpa-0-medium_32.zip"
UMA_PATH = "hf://facebook/UMA/checkpoints/uma-s-1p2.pt"

Thermostat = Literal["none", "csvr", "langevin"]
# In warm-up order: the tour's first, NPT leading as it grows the neighbor lists most.
INTEGRATORS: dict[tuple[Thermostat, bool], str] = {
    ("langevin", True): "baoab_npt_langevin",
    ("none", False): "verlet",
    ("langevin", False): "baoab_langevin",
    ("csvr", False): "csvr",
    ("csvr", True): "csvr_npt",
}
ENSEMBLES = {
    "verlet": "NVE",
    "csvr": "NVT · CSVR",
    "baoab_langevin": "NVT · Langevin",
    "csvr_npt": "NPT · CSVR",
    "baoab_npt_langevin": "NPT · Langevin",
}
# MD potentials per ``--forcefield``; ``say`` names one in the tour's captions.
FORCEFIELDS = {
    "ml": {
        "mace": {
            "label": "MACE-MPA-0",
            "say": "MACE in JAX",
            "tag": "JAX · tojax",
            "steps_per_frame": 2,
            "search_headroom": 4,
        },
        "uma": {
            "label": "UMA-S 1.2",
            "say": "Meta's UMA via PyTorch",
            "tag": "PyTorch · Meta FAIR",
            "steps_per_frame": 1,
            "search_headroom": 4,
        },
    },
    "classical": {
        "uff4mof": {
            "label": "UFF4MOF",
            "say": "UFF4MOF in JAX",
            "tag": "JAX",
            "steps_per_frame": 2,
            "search_headroom": 1,
        },
        "uff4mof_lj": {
            "label": "UFF4MOF · neutral",
            "say": "UFF4MOF without charges",
            "tag": "JAX",
            "steps_per_frame": 8,
            "search_headroom": 1,
        },
    },
}
POTENTIALS = dict(FORCEFIELDS["ml"])  # the active force field's; the first is default
POTENTIAL_CODE = {
    "mace": [
        "model = TojaxedMliap.from_zip_file(  # MACE exported to JAX",
        '    "hf://CuspAI/kUPS-mace-jax/mace-mpa-0-medium_32.zip"',
        ")",
        "potential = make_tojaxed_from_state(",
        "    state_lens, parameters=model, gradient=POSITIONS_AND_CELL",
        ")",
    ],
    "uma": [
        "model = load_uma(  # Meta FAIR Chemistry, PyTorch",
        '    "uma-s-1p2.pt", task_name="odac", compute_cell_gradients=True',
        ")",
        "potential = make_torch_mliap_from_state(",
        "    state_lens, parameters=model, gradient=POSITIONS_AND_CELL",
        ")",
    ],
    "uff4mof": [
        "lj, ewald = gcmc_force_field()  # GCMC's",
        "bonded = uff4mof_terms(framework)  # flexible",
        "potential = sum_potentials(",
        "    make_lennard_jones_from_state(state_lens, lj, …),",
        "    make_ewald_from_state(state_lens, ewald, …),",
        "    *bonded,",
        ")",
    ],
    "uff4mof_lj": [
        "lj, _ = gcmc_force_field()  # GCMC's, no charges",
        "bonded = uff4mof_terms(framework)  # flexible",
        "potential = sum_potentials(",
        "    make_lennard_jones_from_state(state_lens, lj, …),",
        "    *bonded,",
        ")",
    ],
}
CHARTS = [
    {
        "key": "temperature",
        "title": "Temperature",
        "unit": "K",
        "digits": 0,
        "series": ["T"],
    },
    {
        "key": "energy",
        "title": "Energy",
        "unit": "eV",
        "digits": 2,
        "series": ["Total", "Potential"],
    },
    {"key": "volume", "title": "Volume", "unit": "Å³", "digits": 0, "series": ["V"]},
    {
        "key": "pressure",
        "title": "Pressure",
        "unit": "bar",
        "digits": 0,
        "series": ["P"],
    },
]


@dataclass(frozen=True)
class Blocks:
    potential: str = field(default_factory=lambda: next(iter(POTENTIALS)))
    thermostat: Thermostat = "langevin"
    barostat: bool = False

    @property
    def integrator(self) -> str:
        return INTEGRATORS[self.thermostat, self.barostat]

    def valid(self) -> bool:
        return (
            self.potential in POTENTIALS
            and (self.thermostat, self.barostat) in INTEGRATORS
        )


# Shared by every MolecularDynamics instance (and thread), so warming one warms all.
_potentials: dict[str, tuple[Any, Any]] = {}
_frame_functions: dict[Blocks, tuple[Any, int, list[str]]] = {}
COLD: set[Blocks] = set()  # combinations warming in the background; can't be selected
UNAVAILABLE: dict[str, str] = {}  # potential -> why its model failed to load
# Capacities only grow; reusing the settled ones keeps array shapes, and so
# compiled programs, identical across instances, swaps and box sizes.
_neighborlists: dict[str, UniversalNeighborlistParameters] = {}
_locks_lock = threading.Lock()
_locks: dict[Any, threading.Lock] = {}


def _once[K, V](cache: dict[K, V], key: K, build: Callable[[], V]) -> V:
    """``cache[key]``, built at most once even when threads race for it."""
    with _locks_lock:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if key not in cache:
            cache[key] = build()
    return cache[key]


def build_potential(name: str) -> tuple[Any, Any]:
    """Return ``(potential, cutoff)``; models load once and are reused."""
    return _once(_potentials, name, lambda: _load_potential(name))


def use_forcefield(name: str) -> None:
    """Offer the potentials of ``FORCEFIELDS[name]`` from now on."""
    POTENTIALS.clear()
    POTENTIALS.update(FORCEFIELDS[name])


def potential_cards() -> dict[str, dict[str, Any]]:
    """The offered potentials, for the UI; ``unavailable`` says why one failed to load."""
    return {k: v | {"unavailable": UNAVAILABLE.get(k)} for k, v in POTENTIALS.items()}


def _load_potential(name: str) -> tuple[Any, Any]:
    match name:
        case "uff4mof" | "uff4mof_lj":
            from .classical import make_potential

            return make_potential(STATE, charged=name == "uff4mof")
        case "mace":
            return TojaxPotentialConfig(model_path=MACE_PATH).build(
                STATE, POSITIONS_AND_CELL
            )
        case "uma":
            from kups.application.potential.mliap.torch import (
                make_torch_mliap_from_state,
            )
            from kups.potential.mliap.torch import TorchMliap, load_uma

            from .padding import DropPadding

            # Default inference: turbo freezes the composition, but GCMC changes it.
            uma = load_uma(
                get_model_path(UMA_PATH), task_name="odac", compute_cell_gradients=True
            )
            model = TorchMliap.from_module(
                DropPadding(uma.wrapper.module), cutoff=6.0, compute_cell_gradients=True
            )
            potential = make_torch_mliap_from_state(
                STATE, parameters=model, gradient=POSITIONS_AND_CELL
            )
            return potential, model.cutoff
    raise ValueError(f"Unknown potential {name!r}")


def frame_function(blocks: Blocks) -> tuple[Any, int, list[str]]:
    """Jitted frame function, steps per frame and pipeline for a block combination."""

    def build() -> tuple[Any, int, list[str]]:
        potential, _ = build_potential(blocks.potential)
        step = make_md_propagator(STATE, blocks.integrator, potential)  # type: ignore[arg-type]
        n = POTENTIALS[blocks.potential]["steps_per_frame"]
        frame = cycle_function(LoopPropagator(step, n), f"md {blocks} x{n}")
        return frame, n, pipeline(step)

    return _once(_frame_functions, blocks, build)


def pipeline(propagator: Propagator[Any]) -> list[str]:
    """Class names of the integrator's sub-propagators, in execution order."""
    md_step = propagator.propagator.propagators[0]  # type: ignore[attr-defined]
    return [type(p).__name__ for p in md_step.propagators]


def md_parameters(integrator: str, temperature: float) -> MdParameters:
    return MdParameters(
        temperature=temperature,
        time_step=TIME_STEP,
        friction_coefficient=0.01,
        thermostat_time_constant=100.0,
        target_pressure=1e5,
        pressure_coupling_time=500.0,
        compressibility=COMPRESSIBILITY,
        minimum_scale_factor=0.99,
        integrator=integrator,  # type: ignore[arg-type]
    )


class MolecularDynamics:
    """One running MD simulation; drive it from a single thread."""

    def __init__(self, seed: int = 0):
        self.key = jax.random.key(seed)
        self.blocks = Blocks()
        self.temperature = TEMPERATURE
        self.time_fs = 0.0  # keeps running across hand-overs; only a reset restarts it
        self.load(pristine())

    def _next_key(self) -> jax.Array:
        self.key, key = jax.random.split(self.key)
        return key

    @property
    def params(self) -> MdParameters:
        return md_parameters(self.blocks.integrator, self.temperature)

    # -- structure hand-over -----------------------------------------------
    def load(self, structure: Structure) -> None:
        """Continue from ``structure``: framework momenta carry over, new CO₂ is thermal.

        A structure without momenta sits at its potential minimum, where
        equipartition halves the temperature: its momenta are drawn at twice it.
        The arrays always hold ``MAX_CO2`` molecules; unused slots are padding
        outside every system, so any loading runs the same compiled program.
        """
        if len(structure.molecules) > MAX_CO2:
            raise ValueError(
                f"MD holds at most {MAX_CO2} CO₂, got {len(structure.molecules)}"
            )
        self.framework = structure.framework
        self.symbols = structure.symbols
        capacity = len(self.framework) + 3 * MAX_CO2
        n_pad = capacity - len(self.symbols)
        atoms = ase.Atoms(
            self.symbols + ["O"] * n_pad,
            np.concatenate([structure.positions, np.zeros((n_pad, 3))]),
            cell=structure.cell,
            pbc=True,
        )
        heat = 1 if structure.framework_momenta is not None else 2
        self.charted_from = self.time_fs + (heat - 1) * THERMALISE_FS
        params = md_parameters(self.blocks.integrator, heat * self.temperature)
        particles, systems = md_state_from_ase(atoms, params, key=self._next_key())
        padding = jnp.arange(capacity) >= len(self.symbols)
        momenta = jnp.where(padding[:, None], 0.0, particles.data.momenta)
        if structure.framework_momenta is not None:
            momenta = momenta.at[: len(self.framework)].set(structure.framework_momenta)
        system = particles.data.system
        system = Index(
            system.keys, jnp.where(padding, len(system.keys), system.indices)
        )
        particles = bind(particles, lambda t: (t.data.momenta, t.data.system)).set(
            (momenta, system)
        )
        self.state = MdState(
            particles=particles,
            systems=systems,
            neighborlist_params=None,  # type: ignore[arg-type]  # set by _rebind
            step=jnp.array([0]),
        )
        self._rebind()

    def structure(self) -> Structure:
        p = jax.device_get(self.state.particles.data)
        n_framework, n_real = len(self.framework), len(self.symbols)
        positions = np.asarray(p.positions)
        return Structure(
            framework=self.framework,
            framework_positions=positions[:n_framework],
            molecules=positions[n_framework:n_real].reshape(-1, 3, 3),
            cell=np.asarray(self.state.systems.data.cell.vectors[0]),
            framework_momenta=np.asarray(p.momenta)[:n_framework],
        )

    # -- blocks ------------------------------------------------------------
    def update(self, blocks: Blocks) -> None:
        """Swap blocks in place: the trajectory continues with the new physics."""
        if not blocks.valid():
            raise ValueError(f"Invalid combination {blocks}")
        self.blocks = blocks
        self._rebind()

    def set_temperature(self, temperature: float) -> None:
        """Change the thermostat target: data, not code, so nothing recompiles."""
        self.temperature = temperature
        if self.blocks.thermostat == "none":
            return  # NVE has no target; the next thermostat swap picks it up
        self.state = bind(
            self.state, lambda s: s.systems.data.integrator_params.temperature
        ).set(jax.device_put(jnp.array([temperature]), jax.devices()[0]))

    def available(self, potential: str) -> bool:
        """Load the potential's model, remembering why it is unavailable if it fails."""
        try:
            build_potential(potential)
            return True
        except Exception as e:  # missing weights, gated repo, no torch, …
            log.warning("Potential %s unavailable: %s", potential, e)
            UNAVAILABLE[potential] = f"{type(e).__name__}: {e}"[:300]
            return False

    def _rebind(self) -> None:
        nl = _neighborlists.get(self.blocks.potential) or self._estimate_neighborlist()
        integrator_params = _build_integrator_params(
            self.params, self.state.systems.data.cell
        )
        state = bind(
            replace(self.state, neighborlist_params=nl),
            lambda s: s.systems.data.integrator_params,
        ).set(integrator_params)
        # Commit every array to the device: jit caches on committedness, and a
        # mix of fresh and computed arrays would miss programs warmed elsewhere.
        self.state = jax.device_put(state, jax.devices()[0])
        self._cycle_fn, self.steps_per_frame, _ = frame_function(self.blocks)
        self._refresh_forces()
        COLD.discard(self.blocks)

    def _estimate_neighborlist(self) -> UniversalNeighborlistParameters:
        """Edges sized for every CO₂ slot filled (they set MACE's cost); the cheap
        search buffers get the potential's headroom: cells near twice its cutoff
        pull in images."""
        _, cutoff = build_potential(self.blocks.potential)
        headroom = POTENTIALS[self.blocks.potential]["search_headroom"]
        counts = self.state.particles.data.system.counts
        full = counts.set_data(jnp.array([len(self.state.particles)]))
        nl = UniversalNeighborlistParameters.estimate(full, self.state.systems, cutoff)
        return replace(
            nl,
            avg_candidates=nl.avg_candidates * headroom,
            avg_image_candidates=nl.avg_image_candidates * headroom,
        )

    def _refresh_forces(self) -> None:
        """Zero-length steps re-evaluate forces without moving or heating anything."""
        dt = lambda s: s.systems.data.integrator_params.time_step  # noqa: E731
        time_step = bind(self.state, dt).get()
        self.state = bind(self.state, dt).set(jnp.zeros_like(time_step))
        self.state = propagate_and_fix(self._cycle_fn, self._next_key(), self.state)
        self.state = bind(self.state, dt).set(time_step)

    # -- method interface ------------------------------------------------
    def apply(self, cmd: dict[str, Any], status: Status) -> str | None:
        """Handle a UI command; return a short label for a chart marker, if any."""
        match cmd["type"]:
            case "blocks":
                fields: dict[str, Any] = {
                    k: arg(cmd, k, kind)
                    for k, kind in (
                        ("potential", str),
                        ("thermostat", str),
                        ("barostat", bool),
                    )
                    if k in cmd
                }
                blocks = replace(self.blocks, **fields)
                if blocks == self.blocks:
                    return None
                if blocks.potential not in POTENTIALS:
                    raise Refused(f"Unknown potential {blocks.potential!r}")
                if not blocks.valid():
                    raise Refused(
                        "NPT needs a thermostat"
                        if blocks.thermostat == "none"
                        else f"Unknown thermostat {blocks.thermostat!r}"
                    )
                label = POTENTIALS[blocks.potential]["label"]
                if blocks in COLD:
                    ensemble = ENSEMBLES[blocks.integrator]
                    raise Refused(f"{label} · {ensemble} is still warming up")
                if not self.available(blocks.potential):
                    raise Refused(
                        f"{label} unavailable: {UNAVAILABLE[blocks.potential]}"
                    )
                label = self._describe(blocks)
                status(f"Swapping in {label}…")
                self.update(blocks)
                return label
            case "temperature":
                temperature = number(cmd, *TEMPERATURES, "K")
                if temperature == self.temperature:
                    return None
                self.set_temperature(temperature)
                return f"{temperature:g} K"
            case "reset":
                status("Back to the pristine framework…")
                self.time_fs = 0.0
                self.load(pristine())
        return None

    def _describe(self, blocks: Blocks) -> str:
        if blocks.potential != self.blocks.potential:
            return POTENTIALS[blocks.potential]["label"]
        return ENSEMBLES[blocks.integrator]

    def warm_up(
        self,
        status: Status,
        ready: Callable[[str], None] = lambda _: None,
        potentials: Iterable[str] = POTENTIALS,
    ) -> None:
        """Compile every block combination of ``potentials``.

        Run it on a spare instance in a background thread: the frame functions
        it compiles are shared with the live instance.
        """
        for potential in potentials:
            status(f"Warming {POTENTIALS[potential]['label']}")
            if not self.available(potential):
                COLD.difference_update(Blocks(potential, *k) for k in INTEGRATORS)
                ready(potential)
                continue
            self.blocks = Blocks(potential, *next(iter(INTEGRATORS)))
            self.load(pristine())
            for _ in range(SETTLE_FRAMES):  # grow neighbor-list capacities first
                self._try_advance()
            ready(potential)
            settled = None
            while settled != _neighborlists[potential]:
                settled = _neighborlists[potential]
                for thermostat, barostat in INTEGRATORS:
                    self.update(Blocks(potential, thermostat, barostat))
                    self._try_advance()

    def _try_advance(self) -> None:
        try:
            self.advance()
        except RuntimeError:  # diverged: start over and keep warming
            self.load(pristine())

    def ui(self) -> dict[str, Any]:
        b = self.blocks
        n_co2 = (len(self.symbols) - len(self.framework)) // 3
        return {
            "system_label": f"UiO-66 + {n_co2} CO₂ · {len(self.symbols)} atoms",
            "blocks": vars(b),
            "temperature": self.temperature,
            "symbols": self.symbols,
            "static_atoms": None,
            "n_framework": len(self.framework),
            "positions": self.structure().positions.round(3).tolist(),
            "pipeline": [{"name": name} for name in frame_function(b)[2]],
            "pipeline_head": "step = make_md_propagator(…)",
            "listing": [
                "atoms = structure_from_gcmc_or_cif()  # UiO-66 + CO₂, shared with GCMC",
                "particles, systems = md_state_from_ase(atoms, params)",
                "",
                *POTENTIAL_CODE[b.potential],
                "",
                f'step = make_md_propagator(state_lens, "{b.integrator}", potential)',
                "frame = make_cycle_function(LoopPropagator(step, n))",
                "",
                "while booth_is_open:",
                "    state = propagate_and_fix(frame, next(keys), state)",
            ],
            "charts": CHARTS,
        }

    def advance(self) -> dict[str, Any]:
        """Run one frame's worth of steps; reset the structure if the run diverged."""
        t0 = time.perf_counter()
        self.state = propagate_and_fix(self._cycle_fn, self._next_key(), self.state)
        _neighborlists[self.blocks.potential] = self.state.neighborlist_params
        obs = jax.device_get(_observables(self.state))
        elapsed = time.perf_counter() - t0
        if (
            not math.isfinite(obs["temperature"])
            or obs["temperature"] > MAX_TEMPERATURE
        ):
            raise RuntimeError("Simulation diverged")  # the caller resets
        self.time_fs += self.steps_per_frame * TIME_STEP
        return {
            "positions": obs["positions"][: len(self.symbols)].tolist(),
            "cell": obs["cell"].tolist(),
            "x": self.time_fs / 1000,
            "x_unit": "ps",
            "rate": self.steps_per_frame / elapsed,
            "rate_unit": "steps/s",
            "charts": {
                "temperature": [float(obs["temperature"])]
                if self.time_fs > self.charted_from
                else None,
                "energy": [float(obs["total_energy"]), float(obs["potential_energy"])],
                "volume": [float(obs["volume"])],
                "pressure": [float(obs["pressure_bar"])],
            },
            "reference": {
                "temperature": None
                if self.blocks.thermostat == "none"
                else self.temperature
            },
        }


@jax.jit
def _observables(state: MdState) -> dict[str, Any]:
    p, systems = state.particles, state.systems
    n = p.data.system.counts.data[0]  # real atoms only, not padding
    internal_ke = system_kinetic_energy(
        p.data.momenta, p.data.masses, p.data.system, remove_com=True
    )
    kinetic_energy = system_kinetic_energy(p.data.momenta, p.data.masses, p.data.system)
    volume = systems.data.cell.volume
    stress = stress_via_virial_theorem(p, systems).data
    potential_energy = systems.data.potential_energy
    return {
        "positions": p.data.positions.astype(jnp.float32),
        "cell": systems.data.cell.vectors[0].astype(jnp.float32),
        "temperature": (2 * internal_ke / ((3 * n - 3) * BOLTZMANN_CONSTANT))[0],
        "potential_energy": potential_energy[0],
        "total_energy": (potential_energy + kinetic_energy)[0],
        "volume": volume[0],
        "pressure_bar": instantaneous_pressure(internal_ke, stress, volume)[0]
        / PASCAL
        / 1e5,
    }
