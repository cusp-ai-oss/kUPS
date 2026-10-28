# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Builders shared by the application tests: CI argon settings, input files,
potentials over the skin list, MD states with their step, and a small exported
model."""

import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import ase
import ase.build
import jax
import jax.numpy as jnp
import numpy as np
from jax import export

from kups.application.md.data import (
    MdParameters,
    MDParticles,
    MdState,
    MDSystems,
    md_state_from_ase,
)
from kups.application.md.simulation import make_md_propagator
from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.simulations.potentials import BuiltPotential, LjPotentialConfig
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.propagator import Propagator
from kups.core.typing import DTypeLike, ParticleId, SystemId
from kups.core.utils.msgpack import serialize as msgpack_serialize
from kups.potential.classical.lennard_jones import LennardJonesParameters

ROWS = LjPotentialConfig(
    cutoff=8.0, parameters={"Ar": (3.405, 0.010326)}, mixing_rule="lorentz_berthelot"
)
"""Argon Lennard-Jones over the skin list's neighbor rows."""

_LJ = LennardJonesParameters.from_dict(ROWS.cutoff, ROWS.parameters, ROWS.mixing_rule)

type Build = Callable[[Table, Table], BuiltPotential]
"""Builds a potential for the loaded particle and system tables."""


def md_parameters(integrator: str, skin: float = 0.3) -> MdParameters:
    """NPT settings of the CI argon runs (compressibility in 1/Pa)."""
    return MdParameters(
        temperature=120.0,
        time_step=4.0,
        friction_coefficient=0.01,
        thermostat_time_constant=100.0,
        target_pressure=4.2e7,
        pressure_coupling_time=1000.0,
        compressibility=5.4e-10,
        minimum_scale_factor=0.9,
        integrator=integrator,
        initialize_momenta=True,
        skin=skin,
    )


def argon(n: int, seed: int) -> ase.Atoms:
    """Rattled fcc argon of ``n`` cubic cells per axis."""
    atoms = ase.build.bulk("Ar", "fcc", a=5.3, cubic=True) * (n, n, n)
    atoms.rattle(0.1, seed=seed)
    return atoms


def write_cif(atoms: ase.Atoms, path: Path) -> Path:
    """Write ``atoms`` as a P1 CIF labeled by element, as LJ parameters are keyed,
    with ``atoms.info["_atom_site_charge"]`` as a charge column if set.

    ``particles_from_ase`` caches by path and kups_md donates the cached arrays, so
    every run needs new files.
    """
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    rows = [
        f"{symbol} {symbol} {x:.12f} {y:.12f} {z:.12f}"
        for symbol, (x, y, z) in zip(
            atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=False)
        )
    ]
    columns = "_atom_site_fract_z"
    charges = atoms.info.get("_atom_site_charge")
    if charges is not None:
        rows = [f"{row} {q:.6f}" for row, q in zip(rows, charges, strict=True)]
        columns += "\n_atom_site_charge"
    lines = "\n".join(rows)
    path.write_text(
        f"""data_input
_cell_length_a {a:.12f}
_cell_length_b {b:.12f}
_cell_length_c {c:.12f}
_cell_angle_alpha {alpha:.12f}
_cell_angle_beta {beta:.12f}
_cell_angle_gamma {gamma:.12f}
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
{columns}
{lines}
"""
    )
    return path


def graph_lj(factory) -> Build:
    """Graph LJ over ``factory``'s pairs, as the MLFF backends evaluate them."""
    potential = make_lennard_jones_from_state(
        identity_lens(MdState),
        parameters=_LJ,
        gradient=POSITIONS_AND_CELL,
        neighborlist_factory=factory,
    )
    return lambda *_: (potential, _LJ.cutoff)


def md_tables(
    md: MdParameters, atoms: list[ase.Atoms], index_dtype: DTypeLike = int
) -> tuple[Table[ParticleId, MDParticles], Table[SystemId, MDSystems]]:
    """One system per entry of ``atoms``, each with its own momenta key."""
    tables = [
        md_state_from_ase(a, md, key=jax.random.key(i), index_dtype=index_dtype)
        for i, a in enumerate(atoms)
    ]
    return Table.union([p for p, _ in tables], [s for _, s in tables])


def state_and_step(
    md: MdParameters,
    build: Build,
    particles: Table[ParticleId, MDParticles],
    systems: Table[SystemId, MDSystems],
) -> tuple[MdState, Propagator[MdState]]:
    """State with a skin list at the cutoff of ``build``'s potential, and its step."""
    potential, cutoff = build(particles, systems)
    step = make_md_propagator(identity_lens(MdState), md.integrator, potential)
    return MdState.create(particles, systems, cutoff, md.skin), step


def write_pair_model(path: Path) -> None:
    """Exported model with energy sum_ij (rc^2 - r_ij^2)^2 over its edges, rc 5 Å."""

    def energy(params, x):
        i, j = x["edge_index"]
        cell = x["cell"][x["batch"][i]]
        shift = jnp.einsum("ek,ekl->el", x["cell_offsets"], cell)
        r = x["pos"][j] - x["pos"][i] + shift
        pair = params[0] * (25.0 - (r**2).sum(-1)) ** 2
        return jnp.zeros_like(x["charge"]).at[x["batch"][i]].add(pair)

    f64, i64 = np.float64, np.int64
    concrete = {
        "pos": np.zeros((4, 3), f64),
        "atomic_numbers": np.zeros(4, i64),
        "cell": np.zeros((2, 3, 3), f64),
        "pbc": np.ones((2, 3), bool),
        "edge_index": np.zeros((2, 6), i64),
        "cell_offsets": np.zeros((6, 3), f64),
        "charge": np.zeros(2, f64),
        "spin": np.zeros(2, f64),
        "batch": np.zeros(4, i64),
    }
    shapes = {
        "pos": "n, _",
        "atomic_numbers": "n",
        "cell": "b, _, _",
        "pbc": "b, _",
        "edge_index": "_, e",
        "cell_offsets": "e, _",
        "charge": "b",
        "spin": "b",
        "batch": "n",
    }
    params = [np.array(1e-4, f64)]
    specs = export.symbolic_args_specs((params, concrete), shapes_specs=(None, shapes))
    exported = export.export(jax.jit(energy))(*specs)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("model.jax", exported.serialize(vjp_order=1))
        zf.writestr("metadata.json", json.dumps({"cutoff": 5.0}))
        zf.writestr("params.msgpack", msgpack_serialize(params))
