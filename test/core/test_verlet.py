# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Verlet-skin neighbor list (kups.core.neighborlist.verlet).

A stored "skin" list built at ``cutoff + skin`` is reused across steps and re-masked to
the true cutoff via ``RefineCutoffNeighborList``; the expensive dense build runs only when
the PBC-aware trigger fires. Correctness == the reused list reproduces a fresh dense build,
and Verlet MD dynamics are bit-identical to the every-step-dense path.
"""

from __future__ import annotations

import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.application.md.data import MdParameters, md_state_from_ase
from kups.application.md.simulation import (
    make_md_propagator,
    make_verlet_md_propagators,
)
from kups.application.simulations.md_lj import LjMdState
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    UniversalNeighborlistParameters,
    build_skin_edges,
    dense_skin_nl,
    estimate_skin_params,
    should_rebuild,
    skin_mic_ratio,
)
from kups.core.propagator import propagate_and_fix
from kups.core.result import as_result_function
from kups.core.utils.jax import jit, key_chain
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    make_lennard_jones_from_state,
)

CUTOFF, SKIN = 6.0, 1.5


def _make_state(verlet_skin: float, integrator: str = "csvr_npt") -> LjMdState:
    # Ar fcc 4x4x4 → 256 atoms, ~21 Å box; (cutoff+skin)/box = 7.5/21 ≈ 0.36 < 0.5 (single image).
    atoms = ase.build.bulk("Ar", "fcc", a=5.26, cubic=True) * (4, 4, 4)
    md = MdParameters(
        temperature=50.0,
        time_step=2.0,
        friction_coefficient=0.01,
        thermostat_time_constant=100.0,
        target_pressure=1e5,
        pressure_coupling_time=1000.0,
        compressibility=5e-10,
        minimum_scale_factor=0.9,
        integrator=integrator,
        initialize_momenta=True,
        verlet_skin=verlet_skin,
    )
    particles, systems = md_state_from_ase(atoms, md, key=jax.random.key(0))
    ljp = LennardJonesParameters.from_dict(
        cutoff=CUTOFF, parameters={"Ar": (3.4, 0.0103)}, mixing_rule="lorentz_berthelot"
    )
    nlp = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, ljp.cutoff
    )
    if verlet_skin > 0:
        sp = estimate_skin_params(
            particles.data.system.counts, systems, ljp.cutoff, verlet_skin
        )
        edges = build_skin_edges(
            particles, systems, ljp.cutoff, verlet_skin, dense_skin_nl(sp)
        )
        return LjMdState(
            particles,
            systems,
            nlp,
            jnp.array([0]),
            ljp,
            stored_skin_edges=edges,
            reference_positions=particles.data.positions + 0.0,
            reference_cell=jax.tree.map(lambda x: x + 0.0, systems.data.cell),
            should_rebuild=jnp.array(True),
        )
    return LjMdState(particles, systems, nlp, jnp.array([0]), ljp)


def _edge_set(edges) -> set:
    idx = np.asarray(edges.indices.indices)
    sh = np.asarray(edges.shifts).reshape(idx.shape[0], -1)
    return {
        (int(idx[k, 0]), int(idx[k, 1]), tuple(np.round(sh[k]).astype(int)))
        for k in range(idx.shape[0])
        if idx[k, 0] != idx[k, 1]
    }


def test_skin_mic_ratio_safe():
    sv = _make_state(SKIN)
    assert float(skin_mic_ratio(sv.systems, CUTOFF, SKIN)) < 0.5


def test_reuse_reproduces_dense_neighbor_set():
    """The refined skin list at the true cutoff == a fresh dense build (no missed pairs)."""
    sv, sd = _make_state(SKIN), _make_state(0.0)
    reuse = sv.neighborlist(sv.particles, None, sv.systems, sv.lj_parameters.cutoff)
    fresh = DenseNearestNeighborList.from_state(sd)(
        sd.particles, None, sd.systems, sd.lj_parameters.cutoff
    )
    assert _edge_set(reuse) == _edge_set(fresh)


def test_trigger_fires_only_on_motion():
    sv = _make_state(SKIN)
    pos = sv.particles.data.positions
    perp = sv.systems.data.cell.perpendicular_lengths
    sidx = sv.particles.data.system.indices
    assert not bool(should_rebuild(pos, pos, sidx, perp, perp, CUTOFF, SKIN))
    moved = pos.at[0].add(
        jnp.array([SKIN, 0.0, 0.0])
    )  # one atom by a full skin > skin/2
    assert bool(should_rebuild(moved, pos, sidx, perp, perp, CUTOFF, SKIN))


# NVE (verlet), NVT (csvr / baoab_langevin), and NPT (csvr_npt) — the skin list is
# ensemble-agnostic and the trigger's cell-strain term vanishes for fixed-cell ensembles,
# so Verlet dynamics should be bit-identical to every-step-dense in EVERY ensemble.
@pytest.mark.parametrize("integrator", ["verlet", "csvr", "baoab_langevin", "csvr_npt"])
def test_verlet_dynamics_match_dense(integrator: str):
    """Verlet MD (rebuild every few steps) is bit-identical to every-step-dense in fp64,
    across all ensembles (NVE / NVT / NPT)."""
    sl = identity_lens(LjMdState)
    pot = make_lennard_jones_from_state(sl, compute_position_and_cell_gradients=True)
    base = _make_state(SKIN, integrator)
    sp = estimate_skin_params(
        base.particles.data.system.counts, base.systems, base.lj_parameters.cutoff, SKIN
    )
    reuse_prop, rebuild_prop = make_verlet_md_propagators(
        sl, integrator, pot, CUTOFF, SKIN, dense_skin_nl(sp)
    )
    dense_prop = make_md_propagator(sl, integrator, pot)

    reuse = jit(as_result_function(reuse_prop), donate_argnums=(1,))
    rebuild = jit(as_result_function(rebuild_prop), donate_argnums=(1,))
    dense = jit(as_result_function(dense_prop), donate_argnums=(1,))

    sv, sd = _make_state(SKIN, integrator), _make_state(0.0, integrator)
    cv, cd = (
        key_chain(jax.random.key(1)),
        key_chain(jax.random.key(1)),
    )  # identical keys
    maxdiff = 0.0
    for i in range(20):
        sd = propagate_and_fix(dense, next(cd), sd)
        do_rb = i % 7 == 0  # exercise both rebuild and reuse paths
        sv = propagate_and_fix(rebuild if do_rb else reuse, next(cv), sv)
        maxdiff = max(
            maxdiff,
            float(
                jnp.max(
                    jnp.abs(sd.particles.data.positions - sv.particles.data.positions)
                )
            ),
        )
    assert maxdiff < 1e-9, (
        f"Verlet diverged from dense ({integrator}): max|Δpos|={maxdiff:.2e}"
    )
