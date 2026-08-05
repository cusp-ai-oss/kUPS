# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Domain decomposition stays correct as atoms move: the SHIPPED DD apps match single-device.

`test_domain.py` pins the single-step DD == single-device energy+forces. This
pins the dynamic case through the real entry points (`md_lj_dd.run`,
`relax_lj_dd.run`): many steps of the stock run loops under `shard_map`, the
owned-incident edge shard rebuilt every step, HDF5 output written and read
back — bit-identical to the same run on a one-device mesh (where one shard
owns all atoms, so the owned-incident graph is the whole graph). Also pins
that an undersized neighbor-list estimate RESIZES through the assertion
machinery instead of silently corrupting the trajectory.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from packaging.version import Version

from kups.application.md.analysis import analyze_md_file
from kups.application.md.data import MdParameters, MdRunConfig
from kups.application.relaxation.data import RelaxRunConfig
from kups.application.simulations import md_lj_dd, relax_lj_dd
from kups.application.simulations.potentials import LjPotentialConfig
from kups.core.sharding import shard_axis
from kups.core.typing import OriginDeviceId

_AXIS = shard_axis(OriginDeviceId)
_LJ = LjPotentialConfig(
    cutoff=3.0,
    parameters={"Ar": (1.0, 0.5)},
    mixing_rule="lorentz_berthelot",
)
_MD = MdParameters(
    temperature=100.0,
    time_step=1.0,
    friction_coefficient=0.01,
    thermostat_time_constant=100.0,
    target_pressure=0.0,
    pressure_coupling_time=1000.0,
    compressibility=1e-10,
    minimum_scale_factor=0.9,
    integrator="verlet",  # deterministic NVE so D=1 and D=N must match exactly
    initialize_momenta=True,
)
_FIRE = [{"transform": "scale_by_fire", "dt_start": 0.1}]


def _mesh(n_devices: int) -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.array(jax.devices()[:n_devices]), axis_names=(_AXIS,))


@pytest.fixture(scope="module")
def ar_crystal(tmp_path_factory: pytest.TempPathFactory):
    """Factory for jittered simple-cubic Ar crystal P1 CIFs (one per run).

    Hand-written CIF keeps the labels uniform (``Ar``) so they match the LJ
    parameter table (ASE's CIF writer would uniquify them to Ar1, Ar2, ...).
    Each run gets its OWN copy: ``particles_from_ase`` caches per path, and the
    run loop donates state buffers, so a second run reading the same cached
    arrays would hit deleted buffers.
    """
    n_per_axis, spacing = 4, 2.0
    box = spacing * n_per_axis
    rng = np.random.default_rng(0)
    grid = (
        np.stack(np.meshgrid(*[np.arange(n_per_axis)] * 3, indexing="ij"), -1).reshape(
            -1, 3
        )
        * spacing
    )
    frac = ((grid + rng.uniform(-0.3, 0.3, size=grid.shape)) % box) / box
    rows = "\n".join(f"Ar Ar {x:.8f} {y:.8f} {z:.8f}" for x, y, z in frac)
    cif = f"""data_ar
_cell_length_a {box:.6f}
_cell_length_b {box:.6f}
_cell_length_c {box:.6f}
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{rows}
"""
    base = tmp_path_factory.mktemp("dd")

    def make(name: str) -> Path:
        path = base / f"ar_crystal_{name}.cif"
        path.write_text(cif)
        return path

    return make


def _md_config(out_file: Path, inp_file: Path, steps: int = 10) -> md_lj_dd.Config:
    return md_lj_dd.Config(
        run=MdRunConfig(
            out_file=out_file, num_steps=steps, num_warmup_steps=2, seed=42
        ),
        md=_MD,
        potential=_LJ,
        inp_files=(inp_file,),
    )


def _relax_config(out_file: Path, inp_file: Path, steps: int = 8) -> relax_lj_dd.Config:
    return relax_lj_dd.Config(
        run=RelaxRunConfig(
            out_file=out_file,
            max_steps=steps,
            seed=42,
            force_tolerance=1e-10,  # never converges within `steps`
            optimizer=_FIRE,
            optimize_cell=False,
        ),
        potential=_LJ,
        inp_files=(inp_file,),
    )


@pytest.mark.skipif(
    len(jax.devices()) < 2, reason="DD trajectory gate needs a multi-device mesh"
)
def test_dd_md_trajectory_matches_single_device(
    ar_crystal: Callable[[str], Path], tmp_path: Path
) -> None:
    # D=1: one shard owns all atoms -> owned-incident graph == whole graph.
    final_1 = md_lj_dd.run(
        _md_config(tmp_path / "d1.h5", ar_crystal("md_d1")), mesh=_mesh(1)
    )
    final_d = md_lj_dd.run(
        _md_config(tmp_path / "dn.h5", ar_crystal("md_dn")),
        mesh=_mesh(len(jax.devices())),
    )
    # To host: the D=1 and D=N results live on different device meshes.
    pos1 = np.asarray(final_1.particles.data.positions)
    posd = np.asarray(final_d.particles.data.positions)
    mom1 = np.asarray(final_1.particles.data.momenta)
    momd = np.asarray(final_d.particles.data.momenta)
    assert np.allclose(posd, pos1, rtol=1e-9, atol=1e-9), (
        f"max position drift {float(np.abs(posd - pos1).max()):.2e} over 10 steps"
    )
    assert np.allclose(momd, mom1, rtol=1e-9, atol=1e-9), (
        f"max momentum drift {float(np.abs(momd - mom1).max()):.2e}"
    )
    e1 = float(jnp.asarray(final_1.systems.data.potential_energy).sum())
    ed = float(jnp.asarray(final_d.systems.data.potential_energy).sum())
    assert np.allclose(ed, e1, rtol=1e-10), f"energy {ed} != {e1}"
    # The DD run writes a trajectory the stock analyzer can read back.
    assert analyze_md_file(tmp_path / "dn.h5") is not None


@pytest.mark.skipif(
    len(jax.devices()) < 2, reason="DD trajectory gate needs a multi-device mesh"
)
def test_dd_relax_trajectory_matches_single_device(
    ar_crystal: Callable[[str], Path], tmp_path: Path
) -> None:
    final_1 = relax_lj_dd.run(
        _relax_config(tmp_path / "d1.h5", ar_crystal("rx_d1")), mesh=_mesh(1)
    )
    final_d = relax_lj_dd.run(
        _relax_config(tmp_path / "dn.h5", ar_crystal("rx_dn")),
        mesh=_mesh(len(jax.devices())),
    )
    # To host: the D=1 and D=N results live on different device meshes.
    pos1 = np.asarray(final_1.particles.data.positions)
    posd = np.asarray(final_d.particles.data.positions)
    f1 = np.asarray(final_1.particles.data.forces)
    fd = np.asarray(final_d.particles.data.forces)
    assert np.allclose(posd, pos1, rtol=1e-9, atol=1e-9), (
        f"max position drift {float(np.abs(posd - pos1).max()):.2e} over 8 FIRE steps"
    )
    assert np.allclose(fd, f1, rtol=1e-8, atol=1e-8), (
        f"max force diff {float(np.abs(fd - f1).max()):.2e}"
    )
    e1 = float(jnp.asarray(final_1.systems.data.potential_energy).sum())
    ed = float(jnp.asarray(final_d.systems.data.potential_energy).sum())
    assert np.allclose(ed, e1, rtol=1e-10), f"energy {ed} != {e1}"


@pytest.mark.skipif(
    len(jax.devices()) < 2, reason="DD resize gate needs a multi-device mesh"
)
@pytest.mark.skipif(
    Version(jax.__version__) < Version("0.8.0"),
    reason="jax 0.7.x vma bug: searchsorted under shard_map rejects mixed manual "
    "axes (jnp.searchsorted in _generate_image_offsets); fixed upstream",
)
def test_dd_md_undersized_estimate_resizes_and_stays_correct(
    ar_crystal: Callable[[str], Path], tmp_path: Path
) -> None:
    """An undersized neighbor-list estimate must RESIZE through the assertion
    machinery (LensCapacity fix under the shard-mapped step), not silently
    drop edges: the trajectory still matches the well-sized run."""
    from kups.core.sharding import device_put_replicated
    from kups.core.utils.jax import key_chain

    mesh = _mesh(len(jax.devices()))
    config = _md_config(tmp_path / "resized.h5", ar_crystal("resize"), steps=3)
    chain = key_chain(jax.random.key(config.run.seed or 0))
    mb_key = next(chain)
    state, cap_owned, lj = md_lj_dd.init_state(mb_key, config, mesh.size)
    # Sabotage the estimate: 1 edge/candidate per particle is far too small.
    state = dataclasses.replace(
        state,
        neighborlist_params=dataclasses.replace(
            state.neighborlist_params, avg_edges=1, avg_candidates=1
        ),
    )

    from kups.application.md.simulation import make_md_propagator, run_md
    from kups.application.potential.filter import POSITIONS_AND_CELL
    from kups.application.simulations._domain_decomposition import (
        ShardMappedPropagator,
        make_sharded_lj_potential,
        mesh_max_cell_list_view,
    )
    from kups.core.lens import identity_lens

    state_lens = identity_lens(md_lj_dd.LjMdStateDD)
    neighborlist = mesh_max_cell_list_view(
        state_lens.focus(lambda s: s.neighborlist_params), lj.cutoff
    )
    potential = make_sharded_lj_potential(
        state_lens, lj, neighborlist, cap_owned, POSITIONS_AND_CELL
    )
    propagator = ShardMappedPropagator(
        make_md_propagator(state_lens, config.md.integrator, potential), mesh
    )
    final = run_md(
        next(chain), propagator, device_put_replicated(state, mesh), config.run
    )
    assert int(final.neighborlist_params.avg_edges) > 1, "resize did not happen"

    reference = md_lj_dd.run(
        _md_config(tmp_path / "reference.h5", ar_crystal("resize_ref"), steps=3),
        mesh=mesh,
    )
    assert np.allclose(
        np.asarray(final.particles.data.positions),
        np.asarray(reference.particles.data.positions),
        rtol=1e-9,
        atol=1e-9,
    )
