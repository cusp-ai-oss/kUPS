# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Lennard-Jones plus Ewald or PME in one pair pass equals the potentials summed."""

from collections.abc import Iterator
from contextlib import contextmanager

import ase
import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.application.md.data import (
    MdParameters,
    MDParticles,
    MDSystems,
    md_state_from_ase,
)
from kups.application.potential.classical.ewald import make_ewald_from_state
from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.classical.lennard_jones_ewald import (
    make_lennard_jones_ewald_from_state,
)
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.core.data import Table
from kups.core.geometry import position_and_cell_idx_view
from kups.core.lens import const_lens, identity_lens
from kups.core.neighborlist import AdaptiveNeighborList, UniversalNeighborlistParameters
from kups.core.potential import EMPTY_LENS, sum_potentials
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.units import catalog as u
from kups.core.utils.jax import astype_floating, dataclass
from kups.potential.classical.ewald import (
    EwaldParameters,
    PMEParameters,
    make_pme_potential,
)
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.potential.common.graph import POINTCLOUD_GEOMETRY
from test.units import EV, FORCE, A


@contextmanager
def _x64(enabled: bool) -> Iterator[None]:
    """Scoped x64 toggle; `jax.enable_x64` only exists from jax 0.8."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@dataclass
class _State:
    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters


def _salt(cells: int, seed: int) -> ase.Atoms:
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.6, cubic=True) * ((cells,) * 3)
    atoms.rattle(0.1, seed=seed)
    charges = np.where(atoms.numbers == 11, 0.4, -0.4)
    atoms.info["_atom_site_charge"] = charges
    return atoms


def _state(dtype: type, n_sys: int) -> _State:
    md = MdParameters(
        temperature="300 K",
        time_step="1 fs",
        friction_coefficient="1 1/fs",
        thermostat_time_constant="100 fs",
        target_pressure="1 Pa",
        pressure_coupling_time="1.0e10 fs",
        compressibility="4.5e-5 1/Pa",
        minimum_scale_factor=1.0,
        integrator="verlet",
    )
    parts = [md_state_from_ase(_salt(2, seed), md) for seed in range(n_sys)]
    particles, systems = Table.union([p for p, _ in parts], [s for _, s in parts])
    particles, systems = astype_floating((particles, systems), dtype)
    capacities = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, systems.set_data(jnp.ones(n_sys) * 5 * A)
    )
    return _State(particles, systems, capacities)


@pytest.mark.parametrize("mesh", [False, True], ids=["ewald", "pme"])
@pytest.mark.parametrize(("dtype", "n_sys"), [(jnp.float64, 1), (jnp.float32, 2)])
def test_one_pass_equals_the_sum_of_both_potentials(
    dtype: type, n_sys: int, mesh: bool
):
    with _x64(dtype == jnp.float64):
        state = _state(dtype, n_sys)
        lj = LennardJonesParameters.from_dict(
            cutoff=4.0 * A,
            parameters={
                "Na": (2.4 * A, 0.005 * u.eV),
                "Cl": (4.4 * A, 0.004 * u.eV),
            },
            mixing_rule="lorentz_berthelot",
        )
        ones = jnp.ones(n_sys, dtype)
        ewald = EwaldParameters.from_cutoffs(
            state.systems,
            alpha=state.systems.set_data(ones * 0.6 / A),
            cutoff=state.systems.set_data(ones * 5.0 * A),
            k_max=state.systems.set_data(ones * 3.0 / A),
        )
        lens = identity_lens(_State)
        coulomb = make_ewald_from_state(
            lens, parameters=ewald, gradient=POSITIONS_AND_CELL
        )
        if mesh:
            ewald = PMEParameters(ewald.alpha, ewald.cutoff, mesh=(16, 16, 16), order=4)
            coulomb = make_pme_potential(
                lens.focus(lambda s: s.particles),
                lens.focus(lambda s: s.systems),
                lambda s: AdaptiveNeighborList.from_state(s, ewald.cutoff),
                const_lens(ewald),
                POINTCLOUD_GEOMETRY.nest(POSITIONS_AND_CELL),
                EMPTY_LENS,
                EMPTY_LENS,
                patch_idx_view=position_and_cell_idx_view,
            )
        lj, ewald = astype_floating((lj, ewald), dtype)
        one_pass = make_lennard_jones_ewald_from_state(
            lens, lennard_jones=lj, ewald=ewald, gradient=POSITIONS_AND_CELL
        )
        summed = sum_potentials(
            make_lennard_jones_from_state(
                lens, parameters=lj, gradient=POSITIONS_AND_CELL
            ),
            coulomb,
        )
        got, want = (
            jax.jit(as_result_function(lambda s, p=p: p(s).data))(state)
            for p in (one_pass, summed)
        )
        got.raise_assertion()
        want.raise_assertion()
        rtol = 1e-12 if dtype == jnp.float64 else 2e-5
        npt.assert_allclose(
            got.value.total_energies.data.to(EV),
            want.value.total_energies.data.to(EV),
            rtol=rtol,
        )
        forces = want.value.gradients.positions.data.to(FORCE)
        npt.assert_allclose(
            got.value.gradients.positions.data.to(FORCE),
            forces,
            rtol=rtol,
            atol=rtol * float(jnp.abs(forces).max()),
        )
        cell = want.value.gradients.cell.data.vectors.to(FORCE)
        npt.assert_allclose(
            got.value.gradients.cell.data.vectors.to(FORCE),
            cell,
            rtol=rtol,
            atol=rtol * float(jnp.abs(cell).max()),
        )
        assert float(jnp.abs(forces).max()) > 0.01
