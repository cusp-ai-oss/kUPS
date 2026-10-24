# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""PME reciprocal energy against direct Ewald on random triclinic batches."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random

from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.core.capacity import FixedCapacity
from kups.core.cell import (
    Cell,
    DeformedFrame,
    MatrixLogFrame,
    PeriodicCell,
    TriclinicFrame,
)
from kups.core.data import Index, Table, cast_indices
from kups.core.lens import bind, const_lens, view
from kups.core.neighborlist import AllDenseNearestNeighborList, Edges
from kups.core.potential import EMPTY_LENS
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import astype_floating, dataclass
from kups.potential.classical.ewald import (
    EwaldLongRangeInput,
    EwaldParameters,
    PMEParameters,
    estimate_pme_mesh,
    ewald_long_range_energy,
    make_ewald_potential,
    make_pme_potential,
    pme_long_range_energy,
)
from kups.potential.classical.ewald.pme import _bspline_weights, _euler_factors
from kups.potential.common.graph import (
    POINTCLOUD_GEOMETRY,
    GraphPotentialInput,
    HyperGraph,
    PointCloud,
)

from .test_ewald import _make_particle_data, _make_systems

ALPHA, MESH = 0.35, (32, 48, 95)


def _input(batch: int = 4) -> tuple[Table, Table]:
    """19 particles; system 3 (when present) is empty, the last two rows inactive."""
    n = 19
    positions = random.uniform(random.key(81), (n, 3)) * 7
    charges = random.normal(random.key(82), (n,))
    ids = jnp.array([2, 0, 1, 0, 2, 1, 2, 1, 0, 2, 0, 1, 0, 2, 1, 0, 2, 4, 4])
    if batch == 1:
        ids = jnp.where(ids < 4, 0, batch)
    positions = positions.at[-2:].set(jnp.nan)
    charges = charges.at[-2:].set(jnp.nan)
    particles = Table.arange(
        _make_particle_data(positions, charges, batch, system_ids=ids),
        label=ParticleId,
    )
    matrix = jnp.array([[5.0, 0.0, 0.0], [1.0, 10.0, 0.0], [-0.5, 1.5, 20.0]])
    scale = jnp.linspace(1.0, 1.3, batch)[:, None, None]
    cell = PeriodicCell(TriclinicFrame.from_matrix(matrix[None] * scale))
    return particles, _make_systems(cell, jnp.full(batch, 3.0))


def _pme(order: int = 6, mesh=MESH, dtype=jnp.float64):
    def energy(particles: Table, systems: Table) -> Array:
        # strongly typed float64, as estimated screening is
        alpha = systems.set_data(jnp.full(systems.size, ALPHA, jnp.float64))
        params = PMEParameters(
            alpha, systems.map_data(lambda s: s.cutoff), mesh, order, dtype
        )
        edges = Edges(
            Index(particles.keys, jnp.zeros((0, 0), int), _cls=ParticleId),
            jnp.zeros((0, 0, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        return pme_long_range_energy(GraphPotentialInput(params, graph)).data.data

    return energy


def _ewald(systems: Table):
    alpha = systems.set_data(jnp.full(systems.size, ALPHA))
    k_max = systems.set_data(jnp.full(systems.size, 3.5))
    params = EwaldParameters.from_cutoffs(
        systems, alpha, systems.map_data(lambda s: s.cutoff), k_max
    )

    def energy(particles: Table, systems: Table) -> Array:
        inp = EwaldLongRangeInput(PointCloud(particles, systems), params)
        return ewald_long_range_energy(inp).data.data

    return energy


def _value_and_grad(energy, particles: Table, systems: Table):
    def total(positions: Array, cell: Cell) -> tuple[Array, Array]:
        p = bind(particles, lambda q: q.data.positions).set(positions)
        s = bind(systems, lambda q: q.data.cell).set(cell)
        e = energy(p, s)
        return e.sum(), e

    grad = jax.jit(jax.grad(total, argnums=(0, 1), has_aux=True))
    (positions, cell), energies = grad(particles.data.positions, systems.data.cell)
    return np.asarray(energies), np.asarray(positions), cell.frame.vectors


@pytest.mark.parametrize("order", [4, 6, 8])
def test_spline_weights(order):
    frac = random.uniform(random.key(0), (100,))
    w = _bspline_weights(frac, order)
    np.testing.assert_allclose(w.sum(-1), 1.0, rtol=1e-12)
    assert np.all(np.asarray(w) >= 0)
    for size in (7, 8, 9, 16):  # odd sizes and the Nyquist mode of even ones
        factors = np.asarray(_euler_factors(size, order))
        assert np.all(np.isfinite(factors)) and np.all(factors > 0)
    if order == 4:
        f = frac[:, None]
        cubic = jnp.concatenate(
            [
                f**3,
                1 + 3 * f + 3 * f**2 - 3 * f**3,
                4 - 6 * f**2 + 3 * f**3,
                (1 - f) ** 3,
            ],
            axis=1,
        )
        np.testing.assert_allclose(w, cubic / 6, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    "dtype,energy_rtol,force_atol",
    [(jnp.float64, 1e-8, 1e-6), (jnp.float32, 1e-5, 1e-4)],
)
def test_matches_direct_ewald(dtype, energy_rtol, force_atol):
    particles, systems = _input()
    valid = np.asarray(particles.data.system.raw_indices) < systems.size
    e_pme, g_pme, c_pme = _value_and_grad(_pme(dtype=dtype), particles, systems)
    e_ewald, g_ewald, c_ewald = _value_and_grad(_ewald(systems), particles, systems)
    np.testing.assert_allclose(e_pme, e_ewald, rtol=energy_rtol, atol=1e-12)
    rms = np.sqrt(np.mean(g_ewald[valid] ** 2))
    np.testing.assert_allclose(g_pme[valid], g_ewald[valid], atol=force_atol * rms)
    np.testing.assert_allclose(
        c_pme, c_ewald, atol=force_atol * np.abs(np.asarray(c_ewald)).max()
    )
    np.testing.assert_array_equal(g_pme[~valid], 0.0)
    assert e_pme[3] == 0.0  # the empty system


@pytest.mark.parametrize("mesh_dtype", [jnp.float64, jnp.float32])
def test_float32_inputs_give_float32_outputs(mesh_dtype):
    """Float32 positions, cells and charges with float64 screening give float32
    energies and gradients, with a mesh of either dtype."""
    single = astype_floating(_input(), jnp.float32)
    energy = _pme(dtype=mesh_dtype)
    float32 = _value_and_grad(energy, *single)
    float64 = _value_and_grad(energy, *astype_floating(single, jnp.float64))
    for a, b in zip(float32, float64, strict=True):
        assert a.dtype == jnp.float32
        np.testing.assert_allclose(a, b, atol=3e-5 * np.abs(b).max())


@pytest.mark.parametrize("order", [4, 6])
def test_convergence_order(order):
    particles, systems = _input()
    valid = np.asarray(particles.data.system.raw_indices) < systems.size
    _, reference, _ = _value_and_grad(_ewald(systems), particles, systems)
    errors = []
    for scale in (1, 2, 4):
        mesh = (8 * scale, 12 * scale, 24 * scale)
        _, g, _ = _value_and_grad(_pme(order, mesh), particles, systems)
        errors.append(np.abs(g[valid] - reference[valid]).max())
    assert errors[0] / errors[1] > 2 ** (order - 2)
    assert errors[1] / errors[2] > 2 ** (order - 2)


def test_particle_vocabulary_subset():
    particles, systems = _input()
    raw = np.asarray(particles.data.system.raw_indices)
    keep = np.nonzero((raw == 0) | (raw == 2))[0]
    subset = particles.subset(Index.integer(keep, n=particles.size, label=ParticleId))
    # the particles' system index only knows systems 0 and 2
    system = Index(
        (SystemId(0), SystemId(2)), jnp.asarray((raw[keep] == 2).astype(int))
    )
    relabelled = bind(subset, lambda p: p.data.system).set(system)
    expected = np.asarray(_pme()(subset, systems))
    actual = np.asarray(_pme()(relabelled, systems))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert actual[1] == actual[3] == 0.0 and np.all(actual[[0, 2]] != 0.0)


def test_one_system_and_no_particles():
    particles, systems = _input(batch=1)
    energies, gradients, _ = _value_and_grad(_pme(), particles, systems)
    assert np.all(np.isfinite(energies)) and np.all(np.isfinite(gradients))
    np.testing.assert_array_equal(gradients[-2:], 0.0)
    empty = particles.subset(
        Index.integer(np.zeros(0, int), n=particles.size, label=ParticleId)
    )
    np.testing.assert_array_equal(_pme()(empty, systems), 0.0)


def test_int32_indices():
    particles, systems = _input()
    expected = _pme()(particles, systems)
    actual = _pme()(*cast_indices((particles, systems), jnp.int32))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("order,mesh", [(5, (16, 16, 16)), (6, (16, 4, 16))])
def test_invalid_order_raises(order, mesh):
    alpha = Table((SystemId(0),), jnp.array([ALPHA]))
    with pytest.raises(ValueError, match="even spline order"):
        PMEParameters(alpha, alpha, mesh, order)


def _random_batch(seed: int, counts: tuple[int, int]) -> tuple[Table, Table]:
    """Two neutral random systems in skewed cells."""
    key = random.key(seed)
    ids = np.repeat(np.arange(2), counts)
    matrices = np.array(
        [
            [[14.0, 0.0, 0.0], [2.0, 13.0, 0.0], [-1.5, 1.0, 15.0]],
            [[11.0, 0.0, 0.0], [-1.0, 12.5, 0.0], [0.5, -2.0, 10.0]],
        ]
    )
    frac = np.asarray(random.uniform(key, (ids.size, 3)))
    positions = np.einsum("ni,nij->nj", frac, matrices[ids])
    charges = np.array(random.normal(random.fold_in(key, 1), (ids.size,)))
    for s in range(2):
        charges[ids == s] -= charges[ids == s].mean()
    particles = Table.arange(
        _make_particle_data(
            jnp.asarray(positions), jnp.asarray(charges), 2, system_ids=jnp.asarray(ids)
        ),
        label=ParticleId,
    )
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.asarray(matrices)))
    return particles, _make_systems(cell, jnp.full(2, 6.0))


def _reciprocal_forces(energy, particles: Table, systems: Table) -> np.ndarray:
    def total(positions: Array) -> Array:
        return energy(
            bind(particles, lambda p: p.data.positions).set(positions), systems
        ).sum()

    return np.asarray(jax.jit(jax.grad(total))(particles.data.positions))


@pytest.mark.parametrize("counts", [(120, 80), (30, 20)], ids=["dense", "dilute"])
@pytest.mark.parametrize("precision", [1e-4, 1e-6])
def test_make_is_as_accurate_as_direct_ewald(counts, precision):
    particles, systems = _random_batch(0, counts)
    ewald = EwaldParameters.make(particles, systems, precision, real_cutoff=6.0)
    pme = PMEParameters.make(particles, systems, precision, real_cutoff=6.0)
    np.testing.assert_allclose(pme.alpha.data, ewald.alpha.data)
    tight = EwaldParameters.from_cutoffs(
        systems, ewald.alpha, ewald.cutoff, ewald.alpha.map_data(lambda a: 12 * a)
    )

    def direct(params: EwaldParameters):
        def energy(particles: Table, systems: Table) -> Array:
            inp = EwaldLongRangeInput(PointCloud(particles, systems), params)
            return ewald_long_range_energy(inp).data.data

        return energy

    reference = _reciprocal_forces(direct(tight), particles, systems)
    error_ewald = _reciprocal_forces(direct(ewald), particles, systems) - reference

    def mesh(particles: Table, systems: Table) -> Array:
        edges = Edges(
            Index(particles.keys, jnp.zeros((0, 0), int), _cls=ParticleId),
            jnp.zeros((0, 0, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        return pme_long_range_energy(GraphPotentialInput(pme, graph)).data.data

    error_pme = _reciprocal_forces(mesh, particles, systems) - reference
    assert np.sqrt(np.mean(error_pme**2)) <= 2 * np.sqrt(np.mean(error_ewald**2))


def test_mesh_sizes():
    particles, systems = _random_batch(1, (120, 80))
    charges = np.asarray(particles.data.charges)
    ids = np.asarray(particles.data.system.raw_indices)
    ewald = EwaldParameters.make(particles, systems, 1e-6, real_cutoff=6.0)
    pme = PMEParameters.make(particles, systems, 1e-6, real_cutoff=6.0)
    meshes = [
        estimate_pme_mesh(charges[ids == i], systems.data.cell[i], a, k, pme.order)
        for i, (a, k) in enumerate(zip(ewald.alpha.data, ewald.k_max.data))
    ]
    assert pme.mesh == tuple(np.max(meshes, axis=0).tolist())
    for size in np.ravel(meshes):
        rest = int(size)
        for prime in (2, 3, 5):
            while rest % prime == 0:
                rest //= prime
        assert rest == 1 and size >= pme.order


def test_mesh_shrinks_with_looser_precision():
    particles, systems = _random_batch(2, (120, 80))
    meshes = [
        np.prod(PMEParameters.make(particles, systems, p, real_cutoff=6.0).mesh)
        for p in (1e-8, 1e-6, 1e-4)
    ]
    assert meshes[0] > meshes[1] > meshes[2]


def test_make_reads_deformed_cells():
    """Relaxation cells, whose frame parameters differ in shape, give the
    parameters of their triclinic cells."""
    particles, systems = _random_batch(3, (40, 30))
    deformed = bind(systems, lambda s: s.data.cell.frame).apply(
        lambda f: DeformedFrame.from_frame(
            f, cell_factor=jnp.array([40.0, 30.0]), deformation=MatrixLogFrame
        )
    )
    expected, actual = (
        PMEParameters.make(particles, s, 1e-6, real_cutoff=6.0)
        for s in (systems, deformed)
    )
    assert (actual.mesh, actual.order) == (expected.mesh, expected.order)
    np.testing.assert_allclose(actual.alpha.data, expected.alpha.data, rtol=1e-12)
    np.testing.assert_allclose(actual.cutoff.data, expected.cutoff.data, rtol=1e-12)


def test_make_ignores_inactive_rows():
    particles, systems = _input()
    loud = bind(particles, lambda p: p.data.charges).apply(
        lambda q: jnp.where(jnp.isnan(q), 100.0, q)
    )
    a = PMEParameters.make(particles, systems, real_cutoff=3.0)
    b = PMEParameters.make(loud, systems, real_cutoff=3.0)
    assert (a.mesh, a.order) == (b.mesh, b.order)


@dataclass
class _State:
    particles: Table
    systems: Table


def _molecules(exclusions: bool) -> _State:
    """The random batch; with ``exclusions`` in molecules of three, with a net charge."""
    particles, systems = _random_batch(3, (60, 45))
    if exclusions:
        n = particles.size
        ids = np.asarray(particles.data.system.raw_indices)
        particles = Table.arange(
            _make_particle_data(
                particles.data.positions,
                particles.data.charges + 0.1,  # net charge
                2,
                system_ids=jnp.asarray(ids),
                exclusion_ids=jnp.asarray(np.arange(n) // 3),
                exclusion_max_count=3,
            ),
            label=ParticleId,
        )
    return _State(particles, systems)


def _ewald_and_pme(state: _State, exclusions: bool) -> tuple:
    """Ewald and PME with every term, from the same float64 screening."""
    particles, systems = state.particles, state.systems
    alpha = systems.set_data(jnp.full(2, 0.45, jnp.float64))
    cutoff = systems.map_data(lambda s: s.cutoff)
    k_max = alpha.map_data(lambda a: 12 * a)
    n = particles.size
    neighbors = AllDenseNearestNeighborList(FixedCapacity(n), FixedCapacity(n), cutoff)
    views = (
        view(lambda s: s.particles),
        view(lambda s: s.systems),
        lambda s: neighbors,
    )
    derivatives = (POINTCLOUD_GEOMETRY.nest(POSITIONS_AND_CELL), EMPTY_LENS, EMPTY_LENS)
    ewald = make_ewald_potential(
        *views,
        const_lens(EwaldParameters.from_cutoffs(systems, alpha, cutoff, k_max)),
        None,
        None,
        *derivatives,
        include_exclusion_mask=exclusions,
    )
    pme = make_pme_potential(
        *views,
        const_lens(PMEParameters(alpha, cutoff, (48, 48, 48))),
        *derivatives,
        include_exclusion_mask=exclusions,
    )
    return ewald, pme


@pytest.mark.parametrize("exclusions", [False, True])
def test_potential_matches_make_ewald_potential(exclusions):
    state = _molecules(exclusions)
    ewald, pme = _ewald_and_pme(state, exclusions)
    a, b = jax.jit(lambda s: (pme(s).data, ewald(s).data))(state)
    np.testing.assert_allclose(a.total_energies.data, b.total_energies.data, rtol=1e-7)
    rms = np.sqrt(np.mean(np.asarray(b.gradients.positions.data) ** 2))
    np.testing.assert_allclose(
        a.gradients.positions.data, b.gradients.positions.data, atol=1e-6 * rms
    )
    np.testing.assert_allclose(
        a.gradients.cell.data.vectors, b.gradients.cell.data.vectors, atol=1e-5
    )
    for name in ("short_range", "self_interaction", "net_charge"):
        x, y = (getattr(p, name)(state).data.total_energies.data for p in (pme, ewald))
        np.testing.assert_allclose(x, y, rtol=1e-12)
    assert len(pme.potentials) == len(ewald.potentials) == 4 + exclusions


@pytest.mark.parametrize("exclusions", [False, True])
def test_float64_parameters_keep_float32_outputs(exclusions):
    """Float64 screening and reciprocal sums do not promote any term of float32
    positions, cells and charges."""
    state = astype_floating(_molecules(exclusions), jnp.float32)
    for potential in _ewald_and_pme(state, exclusions):
        out = jax.jit(lambda s, p=potential: p(s).data)(state)
        assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(out))
