# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the smooth-PME reciprocal-space potential (``kups.potential.classical.pme``).

PME must reproduce the direct-Ewald reciprocal term (``ewald_long_range_energy``)
for energy, forces (``dE/dr``) and the NPT stress (``dE/dcell``), and handle a
batched multi-system axis. The trusted oracle is the existing direct-Ewald path
(validated against the NaCl Madelung constant in ``test_ewald.py``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.cell import PeriodicCell, TriclinicFrame, make_supercell
from kups.core.data.table import Table
from kups.core.lens import lens
from kups.core.typing import ParticleId, SystemId
from kups.potential.classical.ewald import (
    EwaldLongRangeInput,
    EwaldParameters,
    estimate_ewald_parameters,
    ewald_long_range_energy,
    kvecs_from_kmax,
)
from kups.potential.classical.pme import (
    make_pme_long_range_energy,
    pme_mesh_for_cell,
)
from kups.potential.common.graph import PointCloud

from .test_ewald import _make_particle_data, _make_systems

ORDER = 8


def _nacl(repeats: int = 5, eps: float = 5e-5, jitter: float = 0.3):
    """Build a NaCl rocksalt supercell + estimated single-system EwaldParameters.

    A small thermal ``jitter`` (Angstrom) is applied so the reciprocal-space
    energy is genuinely nonzero — a pristine lattice has a near-zero direct-Ewald
    reciprocal term (the k-vector set misses the Bragg peaks), which makes a
    reciprocal-only rtol comparison degenerate.
    """
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    charges = jnp.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3, dtype=float) * 2))
    cell, (positions, charges) = make_supercell(
        cell, repeats, (positions, charges), lens(lambda x: x[0])
    )
    if jitter:
        positions = positions + jitter * jax.random.normal(
            jax.random.key(7), positions.shape
        )
    est = estimate_ewald_parameters(charges, cell, epsilon_total=eps)
    params = EwaldParameters(
        alpha=Table((SystemId(0),), jnp.asarray([est.alpha])),
        cutoff=Table((SystemId(0),), jnp.asarray([est.real_cutoff])),
        reciprocal_lattice_shifts=Table(
            (SystemId(0),), kvecs_from_kmax(cell, est.k_max)[None]
        ),
    )
    return positions, charges, cell, params


def _lr_input(positions: Array, cell_vectors: Array, charges: Array, params, cutoff):
    """Rebuild an ``EwaldLongRangeInput`` from raw positions + cell matrix (differentiable)."""
    cell = PeriodicCell(TriclinicFrame.from_matrix(cell_vectors))
    particles = Table.arange(
        _make_particle_data(positions, charges, n_systems=1), label=ParticleId
    )
    systems = _make_systems(cell[None], cutoff)
    return EwaldLongRangeInput(PointCloud(particles, systems), params, None)


class TestPMEvsEwald:
    """PME reciprocal term must match the direct-Ewald oracle."""

    def test_recip_energy_matches_ewald(self):
        positions, charges, cell, params = _nacl()
        cellv = cell.vectors  # (3,3)
        mesh = pme_mesh_for_cell(np.asarray(cellv), spacing=0.5)
        inp = _lr_input(positions, cellv, charges, params, params.cutoff.data)
        e_ewald = float(ewald_long_range_energy(inp).data.data[0])
        e_pme = float(make_pme_long_range_energy(mesh, ORDER)(inp).data.data[0])
        npt.assert_allclose(e_pme, e_ewald, rtol=1e-5)

    def test_forces_match_ewald(self):
        positions, charges, cell, params = _nacl()
        cellv = cell.vectors
        mesh = pme_mesh_for_cell(np.asarray(cellv), spacing=0.5)

        def e_ewald(p):
            return ewald_long_range_energy(
                _lr_input(p, cellv, charges, params, params.cutoff.data)
            ).data.data.sum()

        def e_pme(p):
            return make_pme_long_range_energy(mesh, ORDER)(
                _lr_input(p, cellv, charges, params, params.cutoff.data)
            ).data.data.sum()

        f_ewald = jax.grad(e_ewald)(positions)
        f_pme = jax.grad(e_pme)(positions)
        npt.assert_allclose(f_pme, f_ewald, atol=3e-4)

    @pytest.mark.parametrize("shear", [0.0, 0.5])
    def test_energy_matches_ewald_under_shear(self, shear: float):
        """PME must use the same fractional-coordinate convention as the rest of the
        codebase. A transposed transform is invisible for orthorhombic cells (it
        cancels in k.r) but gives the reciprocal energy of the transposed lattice.

        Shears are kept below one lattice unit: an integer shear is a unimodular
        re-basis of the same lattice, and direct Ewald's truncated (and then
        sheared) k-set is not invariant under it, so it stops being a valid oracle.
        """
        positions, charges, cell, params = _nacl()
        cellv = cell.vectors.at[1, 0].add(shear * float(cell.vectors[0, 0]))
        mesh = pme_mesh_for_cell(np.asarray(cellv), spacing=0.4)
        inp = _lr_input(positions, cellv, charges, params, params.cutoff.data)
        e_ewald = float(ewald_long_range_energy(inp).data.data[0])
        e_pme = float(make_pme_long_range_energy(mesh, ORDER)(inp).data.data[0])
        npt.assert_allclose(e_pme, e_ewald, rtol=1e-3)

    def test_full_cell_gradient_matches_ewald(self):
        """Every dE/dcell entry must match direct Ewald, not just the diagonal: the
        off-diagonals are the shear virial an anisotropic barostat integrates, and a
        transposed fractional transform corrupts exactly those."""
        positions, charges, cell, params = _nacl()
        cellv = cell.vectors
        mesh = pme_mesh_for_cell(np.asarray(cellv), spacing=0.4)

        def grad_cell(energy_fn):
            return np.asarray(
                jax.grad(
                    lambda v: energy_fn(
                        _lr_input(positions, v, charges, params, params.cutoff.data)
                    ).data.data.sum()
                )(cellv)
            )

        g_ewald = grad_cell(ewald_long_range_energy)
        g_pme = grad_cell(make_pme_long_range_energy(mesh, ORDER))
        scale = float(np.abs(g_ewald).max())
        npt.assert_allclose(g_pme, g_ewald, rtol=1e-3, atol=1e-3 * scale)

    def test_two_system_batch(self):
        """A 2-system batch returns correct per-SystemId energies (flat batch_idx)."""
        positions, charges, cell, params = _nacl()
        cellv = cell.vectors
        mesh = pme_mesh_for_cell(np.asarray(cellv), spacing=0.5)

        # tile two identical replicas onto one SystemId axis
        pos2 = jnp.concatenate([positions, positions])
        q2 = jnp.concatenate([charges, charges])
        n = positions.shape[0]
        sys_ids = jnp.concatenate([jnp.zeros(n, int), jnp.ones(n, int)])
        cell2 = cell[None]
        cell2 = jax.tree.map(lambda a: jnp.concatenate([a, a]), cell2)
        params2 = EwaldParameters(
            alpha=Table(
                (SystemId(0), SystemId(1)),
                jnp.concatenate([params.alpha.data, params.alpha.data]),
            ),
            cutoff=Table(
                (SystemId(0), SystemId(1)),
                jnp.concatenate([params.cutoff.data, params.cutoff.data]),
            ),
            reciprocal_lattice_shifts=Table(
                (SystemId(0), SystemId(1)),
                jnp.concatenate(
                    [
                        params.reciprocal_lattice_shifts.data,
                        params.reciprocal_lattice_shifts.data,
                    ]
                ),
            ),
        )
        particles = Table.arange(
            _make_particle_data(pos2, q2, n_systems=2, system_ids=sys_ids),
            label=ParticleId,
        )
        systems = _make_systems(cell2, params2.cutoff.data)
        inp = EwaldLongRangeInput(PointCloud(particles, systems), params2, None)
        e_pme = np.asarray(make_pme_long_range_energy(mesh, ORDER)(inp).data.data)
        e_ewald = np.asarray(ewald_long_range_energy(inp).data.data)
        assert e_pme.shape == (2,)
        npt.assert_allclose(e_pme[0], e_pme[1], rtol=1e-10)  # identical replicas
        npt.assert_allclose(e_pme, e_ewald, rtol=1e-5)

    def test_net_charged_system_matches_ewald(self):
        """A net-charged cell needs the same neutralizing-background term as direct
        Ewald; without it the mesh sum (which also omits k=0) is off by E_net."""
        positions, charges, cell, params = _nacl()
        charges = charges.at[0].set(charges[0] + 1.0)  # break neutrality
        assert abs(float(jnp.sum(charges))) > 0.5
        cellv = cell.vectors
        mesh = pme_mesh_for_cell(np.asarray(cellv), spacing=0.5)
        inp = _lr_input(positions, cellv, charges, params, params.cutoff.data)
        e_ewald = float(ewald_long_range_energy(inp).data.data[0])
        e_pme = float(make_pme_long_range_energy(mesh, ORDER)(inp).data.data[0])
        npt.assert_allclose(e_pme, e_ewald, rtol=1e-4)
