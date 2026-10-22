# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Float32 pair terms of ``RowPairEnergy`` against float64.

The Pallas kernel runs under the interpreter on CPU and on Triton on GPUs that have it;
the float32 XLA path runs elsewhere.
"""

import contextlib
import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.lens import bind
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import astype_floating
from kups.potential.classical.lennard_jones import lennard_jones_row_energy
from kups.potential.common import rows as rows_module
from kups.potential.common.graph import PointCloud
from kups.potential.common.rows import RowPairEnergy, RowPotentialInput

from ._rows import (
    Particles,
    Systems,
    lj_systems,
    row_energies,
    rows_skin_list,
    value_and_grad,
)

FLOAT32 = bind(lennard_jones_row_energy, lambda e: e.dtype).set(jnp.float32)


def _assert_float32_close(particles, systems, params, traced):
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    float64 = value_and_grad(
        row_energies(lennard_jones_row_energy, params, skin_list), particles, systems
    )
    float32 = value_and_grad(
        row_energies(FLOAT32, params, skin_list), particles, systems
    )
    for a, b in zip(jax.tree.leaves(float32), jax.tree.leaves(float64), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        finite = np.isfinite(b)
        np.testing.assert_allclose(
            a[finite], b[finite], atol=2e-6 * np.abs(b[finite]).max()
        )
    assert not np.array_equal(float32[0], float64[0])
    assert traced is None or traced


@pytest.mark.parametrize("periodic", [(True, True, True), (True, True, False)])
def test_float32_matches_float64(row_kernel, periodic):
    _assert_float32_close(*lj_systems(periodic), row_kernel)


def test_far_from_the_origin(row_kernel):
    particles, systems, params = lj_systems()
    particles = bind(particles, lambda p: p.data.positions).apply(lambda x: x + 300.0)
    _assert_float32_close(particles, systems, params, row_kernel)


def test_inactive_particles(row_kernel):
    particles, systems, params = lj_systems()
    raw = np.asarray(particles.data.system.raw_indices).copy()
    raw[:3] = systems.size  # out of bounds: inactive
    inactive = Index(particles.data.system.keys, jnp.asarray(raw))
    particles = bind(particles, lambda p: p.data.system).set(inactive)
    particles = bind(particles, lambda p: p.data.positions).apply(
        lambda x: x.at[:3].set(jnp.nan)
    )
    _assert_float32_close(particles, systems, params, row_kernel)


def test_float32_positions_and_float64_cells():
    particles, systems, params = lj_systems()
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    particles = bind(particles, lambda p: p.data.positions).apply(
        lambda x: x.astype(jnp.float32)
    )
    energies, positions, _ = value_and_grad(
        row_energies(FLOAT32, params, skin_list), particles, systems
    )
    assert energies.dtype == np.float64 and np.all(np.isfinite(positions))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_float32_inputs_give_float32_outputs(dtype):
    """Float64 tables do not promote the energies and gradients of float32
    positions and cells, at either precision of the pair terms."""
    particles, systems, params = lj_systems()
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    evaluator = bind(lennard_jones_row_energy, lambda e: e.dtype).set(dtype)
    energy = row_energies(evaluator, params, skin_list)
    single = astype_floating((particles, systems), jnp.float32)
    float32 = value_and_grad(energy, *single)
    float64 = value_and_grad(energy, *astype_floating(single, jnp.float64))
    for a, b in zip(jax.tree.leaves(float32), jax.tree.leaves(float64), strict=True):
        assert a.dtype == jnp.float32
        np.testing.assert_allclose(a, b, atol=2e-6 * np.abs(b).max())


def _count(r2: Array, coefficients: tuple[Array, ...]) -> Array:
    return jnp.ones_like(r2) + 0 * coefficients[0]


@pytest.mark.parametrize("rc", [8.5125, 12.0])  # rc² inexact in float32
@pytest.mark.parametrize("side", [1, -1])
def test_cutoff_agrees_with_float64(row_kernel, rc, side):
    # pairs 2e-7 rc inside or outside the cutoff, wrapped into a large skewed cell
    rng = np.random.default_rng(0)
    # 42 Å apart, also through the faces, which the outermost ones nearly touch
    anchors = np.array(list(itertools.product(range(-2, 3), repeat=3))) * 42.0 + 20.0
    direction = rng.normal(size=anchors.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    partners = anchors + direction * rc * (1 + side * 2e-7)
    lattice = np.array([[210.3, 0.0, 0.0], [20.17, 209.9, 0.0], [-15.31, 10.07, 210.6]])
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.asarray(lattice[None])))
    positions = cell[jnp.zeros(1, int)].wrap(jnp.concatenate([anchors, partners]))
    n = len(positions)
    zeros = np.zeros(n, int)
    particles = Table(
        tuple(ParticleId(i) for i in range(n)),
        Particles(
            positions,
            Index.new(["A"] * n),
            Index.integer(zeros, label=SystemId),
            Index.integer(zeros, label=SystemId),
            Index.integer(np.arange(n)),
        ),
    )
    systems = Table.arange(Systems(cell), label=SystemId)
    cutoffs = Table(systems.keys, jnp.array([rc]))
    skin_list = rows_skin_list(particles, systems, cutoffs)
    assert np.any(np.asarray(skin_list.rows.shifts) != 0)  # pairs across faces
    count = RowPairEnergy[Table[SystemId, Array], Particles](
        function=_count,
        tables=lambda _: (jnp.zeros((1, 1)),),
        species=lambda _, p: jnp.zeros(len(p.positions), int),
        cutoffs=lambda c: c,
    )
    expected = len(anchors) if side < 0 else 0
    for evaluator in (count, bind(count, lambda e: e.dtype).set(jnp.float32)):
        energies = row_energies(evaluator, cutoffs, skin_list)(particles, systems)
        assert float(energies[0]) == expected
    assert row_kernel is None or row_kernel


def _cuda_target():
    """An abstract L4 to lower for CUDA without a GPU.

    Newer JAX takes the Triton target from the abstract device; older JAX has no
    platform on it and needs none.
    """
    fields = {f.name for f in dataclasses.fields(jax.sharding.AbstractDevice)}
    if "platform" not in fields:
        return contextlib.nullcontext()
    l4 = jax.sharding.AbstractDevice(
        device_kind="NVIDIA L4", num_cores=None, platform="gpu"
    )
    mesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=l4)
    return jax.sharding.use_abstract_mesh(mesh)


def test_kernel_lowers_to_triton(monkeypatch):
    monkeypatch.setattr(rows_module, "triton_available", lambda: True)
    jax.clear_caches()
    particles, systems, params = lj_systems()
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    inp = RowPotentialInput(params, PointCloud(particles, systems), skin_list)
    with _cuda_target():
        traced = jax.jit(lambda x: FLOAT32(x).data.data).trace(inp)
        text = traced.lower(lowering_platforms=("cuda",)).as_text()
    jax.clear_caches()
    assert "triton" in text
