"""Grand-canonical Monte Carlo of CO₂ in UiO-66, driven by kUPS' rigid-body GCMC app.

The framework is a rigid snapshot of whatever MD handed over; moves are the
swappable blocks; temperature and pressure are state data.
"""

from __future__ import annotations

import time
from dataclasses import fields, replace
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from kups.application.mcmc.data import RunConfig
from kups.application.simulations import mcmc_rigid
from kups.application.simulations.mcmc_rigid import (
    Config,
    MCMCState,
    MCMCStateUpdate,
    init_state,
    make_propagator,
)
from kups.core.constants import BAR, KELVIN, PASCAL
from kups.core.data import Buffered, Index, Table, WithIndices
from kups.core.lens import bind
from kups.core.patch import Patch
from kups.core.propagator import propagate_and_fix
from kups.core.typing import GroupId, MotifId, MotifParticleId, ParticleId, SystemId
from kups.core.utils.jax import dataclass, no_post_init
from kups.mcmc.fugacity import peng_robinson_log_fugacity
from kups.mcmc.moves import ExchangeChanges, ExchangeGroupData, ExchangeParticleData
from kups.mcmc.probability import LogFugacityRatio, make_muvt_probability_ratio
from kups.potential.classical.ewald import ewald_short_range_pair
from kups.potential.classical.lennard_jones import lennard_jones_pair
from kups.potential.common.fused import FusedPotentialCache
from kups.potential.common.graph import PointCloud

from . import EXAMPLES, Refused, Status, arg, classical, number
from .aot import cycle_function, jit
from .structure import MAX_CO2, Structure, pristine

MOVES = ("translation", "rotation", "reinsertion", "exchange")
MOVE_LABELS = dict(zip(MOVES, ("Translate", "Rotate", "Reinsert", "Exchange")))
SCHEDULERS = tuple(f"{m}_params" for m in MOVES)  # step sizes and acceptance history
WEIGHTS = {"translation": 1.0, "rotation": 1.0, "reinsertion": 1.0, "exchange": 3.0}
PRESSURES = (0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0)  # bar: the batched isotherm
TEMPERATURES = (200.0, 450.0)  # K, the slider's range
# Å, per MD force field: just under half the primitive cell's width, so MD and
# GCMC share one cell; the classical one's, whose NPT needs a margin.
CUTOFFS = {"ml": 6.0, "classical": classical.CUTOFF}
EQUILIBRATION_FRAMES = 60  # frames after a pressure change excluded from the isotherm
ISOTHERM_WINDOW = 300  # latest equilibrated frames averaged per pressure
CO2 = yaml.safe_load(open(EXAMPLES / "adsorbate" / "co2.yaml"))
CHARTS = [
    {
        "key": "loading",
        "title": "CO₂ loading",
        "unit": "mmol/g",
        "digits": 2,
        "series": ["Loading"],
    },
    {
        "key": "energy",
        "title": "Energy",
        "unit": "eV",
        "digits": 2,
        "series": ["CO₂"],
    },
    {
        "key": "acceptance",
        "title": "Acceptance",
        "unit": "%",
        "digits": 0,
        "series": list(MOVE_LABELS.values()),
    },
    {
        "key": "isotherm",
        "title": f"Isotherm ×{len(PRESSURES)}",
        "unit": "mmol/g",
        "digits": 2,
        "series": ["Rigid"],
        "kind": "scatter",
        "x_unit": "bar",
    },
]


def gcmc_config(temperature: float, pressure_bar: float, cutoff: float) -> Config:
    """Every move enabled: which ones run is decided by ``GcmcState.move_weights``."""
    probs = {f"{m}_prob": w for m, w in WEIGHTS.items()}
    lj = yaml.safe_load(open(EXAMPLES / "lennard_jones" / "trappe.yaml"))
    return Config(
        adsorbates=(CO2,),
        hosts=(
            {
                "cif_file": str(EXAMPLES / "host" / "RUBTAK.cif"),
                "pressure": pressure_bar * 1e5,
                "temperature": temperature,
                "init_adsorbates": (0,),
            },
        ),
        run=RunConfig(
            out_file="unused.h5",
            num_cycles=0,
            num_warmup_cycles=0,
            min_cycle_length=20,
            **probs,
        ),
        lj=lj | {"cutoff": cutoff},
        ewald={"real_cutoff": cutoff, "precision": 1e-6},
    )


def log_fugacity(pressure_bar: float, temperature: float) -> jax.Array:
    """CO₂'s Peng–Robinson log fugacity, one entry per adsorbate species."""
    return peng_robinson_log_fugacity(
        jnp.asarray(pressure_bar * BAR),
        jnp.asarray(temperature * KELVIN),
        jnp.asarray([CO2["critical_pressure"] * PASCAL]),
        jnp.asarray([CO2["critical_temperature"] * KELVIN]),
        jnp.asarray([CO2["acentric_factor"]]),
    ).log_fugacity


def co2_slots() -> Any:
    """Size every system's CO₂ buffer to MD's capacity plus one, the slot an
    insertion is proposed into (kUPS' saturation estimate is smaller than what
    the short cutoff packs in)."""
    return mock.patch.object(
        mcmc_rigid,
        "estimate_max_adsorbates",
        lambda particles, motifs, systems: systems.set_data(
            jnp.full(len(systems), MAX_CO2 + 1)
        ),
    )


@dataclass
class CappedFugacityRatio[State, Move: Patch[Any]](LogFugacityRatio[State, Move]):
    """kUPS' fugacity ratio, rejecting any insertion beyond ``MAX_CO2`` molecules:
    a hard wall at the loading MD can hold."""

    def __call__(self, state: State, patch: Move) -> Any:
        ratio = super().__call__(state, patch)
        full = self.patched_particle_counts(state, patch).data.sum(-1) > MAX_CO2
        return replace(
            ratio, data=ratio.data.set_data(jnp.where(full, -jnp.inf, ratio.data.data))
        )


def capped_muvt_probability_ratio(state: Any, potential: Any) -> Any:
    potential, ratio = make_muvt_probability_ratio(state, potential)
    capped = CappedFugacityRatio(**vars(ratio.log_fugacity_ratio))
    return potential, replace(ratio, log_fugacity_ratio=capped)


def deterministic_propagator(state: MCMCState, config: Config) -> Any:
    """kUPS' GCMC propagators, set up on the CPU, capped at ``MAX_CO2``.

    The setup sums energy coefficients that end up as constants in the compiled
    program. GPU sums vary in the last bits from run to run, which would change
    the program, and so its key in the persistent compile cache, on every start.
    """
    with (
        jax.default_device(jax.devices("cpu")[0]),
        mock.patch.object(
            mcmc_rigid, "make_muvt_probability_ratio", capped_muvt_probability_ratio
        ),
    ):
        return make_propagator(state, config.run)


class BatchedIsotherm:
    """A rigid isotherm from one GCMC state holding one UiO-66 system per pressure.

    Every pressure advances in the same compiled cycle: batching is data.
    """

    def __init__(self, temperature: float, cutoff: float, key: jax.Array):
        self.key = key
        config = gcmc_config(temperature, 1.0, cutoff)
        host = config.hosts[0]
        config = config.model_copy(
            update={
                "hosts": tuple(
                    host.model_copy(update={"pressure": p * 1e5}) for p in PRESSURES
                )
            }
        )
        with co2_slots():
            state = init_state(self._next_key(), config)
        init, cycle = deterministic_propagator(state, config)
        self._init = cycle_function(init, f"isotherm init {cutoff}")
        self._cycle_fn = cycle_function(cycle, f"isotherm cycle {cutoff}")
        self._empty = jax.device_put(state, jax.devices()[0])
        self.set_temperature(temperature)
        p = jax.device_get(state.particles.data)
        host = (np.asarray(p.system.indices) == 0) & ~np.asarray(p.group.valid_mask)
        keys = state.particles.data.labels.keys
        self.framework = {
            "symbols": [
                keys[i].split("_")[0] for i in np.asarray(p.labels.indices)[host]
            ],
            "positions": np.asarray(p.positions)[host].round(3).tolist(),
            "cell": np.asarray(state.systems.data.cell.vectors[0]).tolist(),
        }

    def _next_key(self) -> jax.Array:
        self.key, key = jax.random.split(self.key)
        return key

    def set_temperature(self, temperature: float) -> None:
        """Start over at ``temperature``: an isotherm is at one temperature."""
        self.temperature = temperature
        state = bind(
            jax.tree.map(jnp.copy, self._empty),
            lambda s: (s.systems.data.temperature, s.systems.data.log_fugacity),
        ).set(
            (
                jnp.full(len(PRESSURES), temperature),
                jnp.stack([log_fugacity(p, temperature) for p in PRESSURES]),
            )
        )
        state = jax.device_put(state, jax.devices()[0])
        self.state = propagate_and_fix(self._init, self._next_key(), state)
        self.cycles = 0
        self.samples: list[np.ndarray] = []

    def advance(self) -> None:
        self.state = propagate_and_fix(self._cycle_fn, self._next_key(), self.state)
        self.cycles += 1
        if self.cycles > EQUILIBRATION_FRAMES:
            self.samples.append(np.asarray(self.state.groups.data.system.counts.data))
            del self.samples[:-ISOTHERM_WINDOW]

    def guests(self) -> list[list[list[float]]]:
        """Guest atom positions per pressure, molecule by molecule (C, O, O), each
        molecule shifted into the framework's cell."""
        p = jax.device_get(self.state.particles.data)
        system, guest = np.asarray(p.system.indices), np.asarray(p.group.valid_mask)
        cell = np.asarray(self.framework["cell"])
        out: list[list[list[float]]] = []
        for k in range(len(PRESSURES)):
            atoms = _guest_atoms(
                p.positions,
                p.labels.indices,
                p.group.indices,
                p.motif.indices,
                guest & (system == k),
                self.state.particles.data.labels.keys,
            )
            molecules = np.reshape(atoms["positions"], (-1, 3, 3))
            shift = np.floor(molecules[:, 0] @ np.linalg.inv(cell)) @ cell
            out.append((molecules - shift[:, None]).reshape(-1, 3).round(3).tolist())
        return out

    def molecules(self) -> dict[float, float]:
        """Mean CO₂ per cell at each pressure; empty while equilibrating."""
        if not self.samples:
            return {}
        return dict(zip(PRESSURES, np.mean(self.samples, axis=0).tolist()))


class Isotherm:
    """Mean CO₂ loading per pressure, from frames well after the last change."""

    def __init__(self, equilibration: int):
        self.equilibration = equilibration
        self.samples: dict[float, list[float]] = {}
        self.since_change = 0

    def add(self, pressure_bar: float, loading: float) -> None:
        self.since_change += 1
        if self.since_change > self.equilibration:
            samples = self.samples.setdefault(pressure_bar, [])
            samples.append(loading)
            del samples[:-ISOTHERM_WINDOW]

    def means(self) -> dict[float, float]:
        return {p: float(np.mean(v)) for p, v in sorted(self.samples.items())}


@dataclass
class GcmcState(MCMCState):
    """kUPS' GCMC state plus the move weights, as data: toggling a move recompiles nothing."""

    move_weights: jax.Array


def _weights_from_state(cycle: Any) -> Any:
    """Point the MCMC propagator inside kUPS' cycle at ``state.move_weights``."""
    loop = cycle.propagator  # ResetOnErrorPropagator(LoopPropagator(MCMCPropagator))
    mcmc = replace(loop.propagator, weights=lambda s: s.move_weights)
    return replace(cycle, propagator=replace(loop, propagator=mcmc))


class GrandCanonicalMC:
    """CO₂ adsorbing into a rigid UiO-66 snapshot; drive it from a single thread."""

    def __init__(self, forcefield: str = "ml", seed: int = 1):
        self.key = jax.random.key(seed)
        self.cutoff = CUTOFFS[forcefield]
        self.temperature = 298.0
        self.pressure_bar = 1.0
        self.moves = frozenset(MOVES)
        self.cycles = 0  # keeps running across hand-overs; only a reset restarts it
        config = self._config()
        with co2_slots():
            mcmc = init_state(self._next_key(), config)
        empty = GcmcState(
            **{f.name: getattr(mcmc, f.name) for f in fields(mcmc)},
            move_weights=jnp.array(list(self.weights.values())),
        )
        init, cycle = deterministic_propagator(empty, config)
        self._init = cycle_function(init, f"gcmc init {self.cutoff}")  # full energy
        self._cycle_fn = cycle_function(
            _weights_from_state(cycle), f"gcmc cycle {self.cutoff}"
        )
        self._empty = jax.device_put(empty, jax.devices()[0])
        self.state = self._empty
        self.load(pristine())
        self.batch = BatchedIsotherm(self.temperature, self.cutoff, self._next_key())

    def _next_key(self) -> jax.Array:
        self.key, key = jax.random.split(self.key)
        return key

    def _config(self) -> Config:
        return gcmc_config(self.temperature, self.pressure_bar, self.cutoff)

    @property
    def weights(self) -> dict[str, float]:
        """Each move's selection weight; a disabled move weighs nothing."""
        return {m: WEIGHTS[m] if m in self.moves else 0.0 for m in MOVES}

    # -- structure hand-over -----------------------------------------------
    def load(self, structure: Structure) -> None:
        """Freeze ``structure``'s framework and adopt its CO₂ molecules.

        The moves keep their tuned step sizes and acceptance history. The
        framework's own energy is measured first: charts show the CO₂'s.
        """
        cell = np.asarray(self._empty.systems.data.cell.vectors[0])
        self.handed_over = structure  # MD's cell and momenta, returned unchanged
        local = structure.in_cell(cell)  # MD may have strained the cell slightly
        n_framework = len(structure.framework)
        schedulers = {k: getattr(self.state, k) for k in SCHEDULERS}
        state = jax.tree.map(jnp.copy, replace(self._empty, **schedulers))
        state = bind(state, lambda s: s.particles.data.positions).apply(
            lambda r: r.at[:n_framework].set(local.framework_positions)
        )
        state = self._evaluate(state)
        self.host_energy = float(state.systems.data.potential_energy.value[0])
        self.state = self._evaluate(_insert(state, local.molecules, self._next_key()))
        self.set_conditions(self.temperature, self.pressure_bar)
        p = jax.device_get(self.state.particles.data)
        self.static_atoms = {
            "symbols": list(structure.framework),
            "positions": np.asarray(p.positions)[:n_framework].round(3).tolist(),
        }
        # g/mol per cell, to report loading in mmol/g
        self.framework_mass = float(np.asarray(p.masses)[:n_framework].sum())

    def _evaluate(self, state: MCMCState) -> Any:
        state = bind(state, lambda s: s.cell_tables).set(_cell_tables(state))
        state = jax.device_put(state, jax.devices()[0])
        return propagate_and_fix(self._init, self._next_key(), state)

    def structure(self) -> Structure:
        p = jax.device_get(self.state.particles.data)
        guests = np.asarray(p.group.valid_mask)
        order = np.lexsort(
            (np.asarray(p.motif.indices)[guests], np.asarray(p.group.indices)[guests])
        )
        n_framework = len(self.handed_over.framework)
        local = replace(
            self.handed_over,
            framework_positions=np.asarray(p.positions)[:n_framework],
            molecules=np.asarray(p.positions)[guests][order].reshape(-1, 3, 3),
            cell=np.asarray(self.state.systems.data.cell.vectors[0]),
        )
        return local.in_cell(self.handed_over.cell)

    def loading(self, molecules: float) -> float:
        """CO₂ loading in mmol/g of framework."""
        return molecules / self.framework_mass * 1000

    def rigid_isotherm(self) -> dict[float, float]:
        """Loading (mmol/g) per pressure (bar), from the batched run."""
        return {p: self.loading(n) for p, n in self.batch.molecules().items()}

    def _guests(self) -> dict[str, list[Any]]:
        p = jax.device_get(self.state.particles.data)
        return _guest_atoms(
            p.positions,
            p.labels.indices,
            p.group.indices,
            p.motif.indices,
            p.group.valid_mask,
            self.state.particles.data.labels.keys,
        )

    # -- blocks ------------------------------------------------------------
    def set_conditions(self, temperature: float, pressure_bar: float) -> None:
        """Temperature and fugacity are state data: no recompilation."""
        self.temperature, self.pressure_bar = temperature, pressure_bar
        log_f = log_fugacity(pressure_bar, temperature)
        self.state = bind(
            self.state,
            lambda s: (
                s.systems.data.temperature,
                s.systems.data.log_fugacity,
                s.move_weights,
            ),
        ).set(
            jax.device_put(
                (
                    jnp.array([temperature]),
                    log_f[None],
                    jnp.array(list(self.weights.values())),
                ),
                jax.devices()[0],
            )
        )

    # -- method interface ------------------------------------------------
    def apply(self, cmd: dict[str, Any], status: Status) -> str | None:
        match cmd["type"]:
            case "moves":
                move, on = arg(cmd, "move", str), arg(cmd, "value", bool)
                if move not in MOVES:
                    raise Refused(f"Unknown move {move!r}")
                moves = self.moves | {move} if on else self.moves - {move}
                if not moves:
                    raise Refused("Monte Carlo needs at least one move")
                if moves == self.moves:
                    return None
                self.moves = moves
                self.set_conditions(
                    self.temperature, self.pressure_bar
                )  # writes the weights
                return f"{'+' if on else '−'} {MOVE_LABELS[move]}"
            case "temperature":
                temperature = number(cmd, *TEMPERATURES, "K")
                if temperature == self.temperature:
                    return None
                self.set_conditions(temperature, self.pressure_bar)
                self.batch.set_temperature(temperature)
                return f"{temperature:g} K"
            case "pressure":
                pressure = number(cmd, PRESSURES[0], PRESSURES[-1], "bar")
                if pressure == self.pressure_bar:
                    return None
                self.set_conditions(self.temperature, pressure)
                return f"{pressure:g} bar"
            case "reset":
                status("Back to the pristine framework…")
                self.cycles = 0
                self.batch.set_temperature(self.temperature)
                self.state = self._empty  # fresh move statistics too
                self.load(pristine())
        return None

    def warm_up(self, status: Status) -> None:
        status("GCMC · compiling")
        self.advance()
        for _ in range(EQUILIBRATION_FRAMES):  # grow the batch's capacities now
            self.batch.advance()
        self.load(pristine())

    def ui(self) -> dict[str, Any]:
        weights = self.weights
        total = sum(weights.values())
        return {
            "moves": {m: m in self.moves for m in MOVES},
            "temperature": self.temperature,
            "pressure": self.pressure_bar,
            "symbols": None,
            "static_atoms": self.static_atoms,
            "guests": self._guests(),
            "batch": {"pressures": PRESSURES, "framework": self.batch.framework},
            "pipeline": [
                {
                    "name": f"{m.capitalize()}Move",
                    "detail": f"p = {weights[m] / total:.0%}",
                }
                for m in MOVES
                if weights[m]
            ]
            + [
                {"name": "LennardJones", "detail": "TraPPE / UFF"},
                {"name": "LJTailCorrection", "detail": "analytic"},
                {"name": "Ewald", "detail": "incremental k-space"},
            ],
            "pipeline_head": "MCMCPropagator(moves) · sum_potentials(LJ, tail, Ewald)",
            "listing": [
                "host = framework_snapshot_from_md()  # rigid while GCMC runs",
                "state = init_state(key, Config(adsorbates=(co2,), hosts=(host,), …))",
                "",
                f"log_f = peng_robinson_log_fugacity({self.pressure_bar:g} * BAR, {self.temperature:g} * KELVIN, *co2)",
                "state = set_fugacity(state, log_f)  # data: no recompile",
                "",
                "potential = sum_potentials(lennard_jones, lj_tail, ewald)",
                "step = make_gcmc_mcmc_propagator(state_lens, update, muvt_ratio)",
                "step.weights = lambda s: s.move_weights  # read from the state",
                f"state.move_weights = [{', '.join(f'{w:g}' for w in weights.values())}]"
                "  # no recompile",
                "frame = make_cycle_function(LoopPropagator(step, n_molecules))",
                "",
                "",
                f"# isotherm: {len(PRESSURES)} pressures = {len(PRESSURES)} systems in one state",
                "hosts = tuple(host.at(p) for p in PRESSURES)",
                "batch = init_state(key, Config(hosts=hosts, …))",
                "",
                "while booth_is_open:",
                "    state = propagate_and_fix(frame, next(keys), state)",
                "    batch = propagate_and_fix(batch_frame, next(keys), batch)",
            ],
            "charts": CHARTS,
        }

    def advance(self, isotherm: Isotherm | None = None) -> dict[str, Any]:
        """One cycle here and one in the batched isotherm; the loading also goes
        into ``isotherm``, if given."""
        t0 = time.perf_counter()
        n_before = int(self.state.groups.data.system.counts.data[0])
        self.state = propagate_and_fix(self._cycle_fn, self._next_key(), self.state)
        obs = jax.device_get(_observables(self.state))
        elapsed = time.perf_counter() - t0
        self.cycles += 1
        loading = self.loading(int(obs["molecules"]))
        if isotherm is not None:
            isotherm.add(self.pressure_bar, loading)
        self.batch.advance()
        guests = _guest_atoms(
            obs["positions"],
            obs["labels"],
            obs["groups"],
            obs["motifs"],
            obs["guest_mask"],
            self.state.particles.data.labels.keys,
        )
        return guests | {
            "cell": np.asarray(obs["cell"]).tolist(),
            "batch": self.batch.guests(),
            "x": self.cycles,
            "x_unit": "cycles",
            "rate": max(n_before, 20) / elapsed,
            "rate_unit": "moves/s",
            "charts": {
                "loading": [loading],
                "energy": [float(obs["energy"]) - self.host_energy],
                "acceptance": [100 * float(a) for a in obs["acceptance"]],
                "isotherm": [[p, y] for p, y in self.rigid_isotherm().items()],
            },
            "reference": dict[str, float | None](),
        }


def _insert(state: MCMCState, molecules: np.ndarray, key: jax.Array) -> MCMCState:
    """Place CO₂ molecules at given positions, one at a time so shapes never change.

    Validation is off: each step's jitted output is already valid.
    """
    with no_post_init():
        for molecule, k in zip(molecules, jax.random.split(key, len(molecules))):
            state = _insert_one(state, jnp.asarray(molecule), k)
    return state


@jit
def _insert_one(state: MCMCState, molecule: jax.Array, key: jax.Array) -> MCMCState:
    """kUPS' own insertion move, at a given position instead of a random one."""
    n_atoms = len(CO2["symbols"])
    n_sys = len(state.systems)
    group = state.groups.select_free(1)
    particles = ExchangeParticleData(
        new_positions=molecule,
        system=Index.integer(
            jnp.zeros(n_atoms, int),
            n=n_sys,
            label=SystemId,
            max_count=state.particles.data.system.max_count,
        ),
        group=Index.integer(
            jnp.repeat(group.indices, n_atoms),
            n=len(state.groups),
            label=GroupId,
            max_count=n_atoms,
        ),
        motif=Index.integer(
            jnp.arange(n_atoms), n=len(state.motifs), label=MotifParticleId
        ),
    )
    group_data = ExchangeGroupData(
        motif=Index.integer(
            jnp.zeros(1, int), n=state.motifs.data.motif.num_labels, label=MotifId
        ),
        system=Index.integer(
            jnp.zeros(1, int),
            n=n_sys,
            label=SystemId,
            max_count=state.groups.data.system.max_count,
        ),
    )
    changes = ExchangeChanges(
        WithIndices(
            state.particles.select_free(n_atoms),
            Buffered.arange(particles, label=ParticleId),
        ),
        WithIndices(group, Buffered.arange(group_data, label=GroupId)),
    )
    update = MCMCStateUpdate.from_changes(key, state, changes)
    return update(state, Table(state.systems.keys, jnp.ones(n_sys, bool)))


@jit
def _cell_tables(state: MCMCState) -> FusedPotentialCache[Any]:
    """kUPS' ``init_cell_tables`` for a charged system, jitted for fast hand-overs."""
    lj = lennard_jones_pair.with_particles(lambda p: p).with_parameters(lambda p: p[0])
    coulomb = ewald_short_range_pair.with_particles(lambda p: p).with_parameters(
        lambda p: p[1]
    )
    return FusedPotentialCache.create(
        lj + coulomb,
        (state.lj_parameters.data, state.ewald_parameters.data),
        PointCloud(state.particles, state.systems),
        layout=state.cell_tables.layout,  # sized once, so the rebuild can be jitted
    )


def _guest_atoms(
    positions: Any,
    labels: Any,
    groups: Any,
    motifs: Any,
    mask: Any,
    keys: tuple[str, ...],
) -> dict[str, list[Any]]:
    """Guest atoms molecule by molecule, each in motif order (C, O, O)."""
    mask = np.asarray(mask)
    order = np.lexsort((np.asarray(motifs)[mask], np.asarray(groups)[mask]))
    return {
        "symbols": [keys[i].split("_")[0] for i in np.asarray(labels)[mask][order]],
        "positions": np.asarray(positions)[mask][order].round(3).tolist(),
    }


@jax.jit
def _observables(state: MCMCState) -> dict[str, Any]:
    p = state.particles.data
    schedulers = (
        state.translation_params,
        state.rotation_params,
        state.reinsertion_params,
        state.exchange_params,
    )
    return {
        "positions": p.positions.astype(jnp.float32),
        "labels": p.labels.indices,
        "guest_mask": p.group.valid_mask,
        "groups": p.group.indices,
        "motifs": p.motif.indices,
        "cell": state.systems.data.cell.vectors[0].astype(jnp.float32),
        "molecules": state.groups.data.system.counts.data[0],
        "energy": state.systems.data.potential_energy.value[0],
        "acceptance": jnp.stack([s.data.history.values[0].mean() for s in schedulers]),
    }
