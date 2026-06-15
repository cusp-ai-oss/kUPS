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

from ..clear_cache import clear_cache  # noqa: F401
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
        npt.assert_allclose(e_pme, e_ewald, rtol=1e-4)

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
        npt.assert_allclose(f_pme, f_ewald, rtol=1e-2, atol=1e-3)

    def test_stress_finite_difference_self_consistent(self):
        """PME autodiff dE/dcell (NPT virial) must equal a finite-difference of E.

        This is the gate from PME_SCOPING §6. We do NOT tightly compare to the
        direct-Ewald stress: the reciprocal *virial* converges much more slowly in
        ``k`` than the energy, and ``estimate_ewald_parameters`` targets energy
        accuracy, so the truncated-k Ewald shear stress is the *less* converged
        reference. The PME full-grid stress is correct iff it matches a
        finite-difference of the PME energy itself.
        """
        positions, charges, cell, params = _nacl()
        cellv = cell.vectors
        mesh = pme_mesh_for_cell(np.asarray(cellv), spacing=0.4)

        def e_pme(v):
            return make_pme_long_range_energy(mesh, ORDER)(
                _lr_input(positions, v, charges, params, params.cutoff.data)
            ).data.data.sum()

        g_pme = np.asarray(jax.grad(e_pme)(cellv))
        h = 1e-5
        for a, b in [(0, 0), (1, 1), (2, 2), (1, 0), (2, 0)]:
            fd = float(
                (e_pme(cellv.at[a, b].add(h)) - e_pme(cellv.at[a, b].add(-h))) / (2 * h)
            )
            npt.assert_allclose(g_pme[a, b], fd, rtol=1e-4, atol=1e-6)

    def test_stress_diagonal_matches_ewald(self):
        """The (well-converged) diagonal/normal stress should match direct Ewald."""
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
        diag_e = np.diag(g_ewald)
        diag_p = np.diag(g_pme)
        npt.assert_allclose(diag_p, diag_e, rtol=5e-3, atol=5e-3)

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
        npt.assert_allclose(e_pme, e_ewald, rtol=2e-3)
