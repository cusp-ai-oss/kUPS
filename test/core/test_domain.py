# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Domain decomposition gates: the stock energies reproduce single-device results under shard_map.

The contract (see ``kups.potential.common.graph.Decomposition``): every device
builds its owned-incident edge shard and tags the graph ``Sharded``; the
UNCHANGED energy functions then reduce owned-only, per-device energies are
per-system partials whose ``psum`` is the global value, and gradients of the
replicated inputs are already mesh-summed by ``shard_map``'s transpose. Pinned
here: the ``Sharded`` mask/combine algebra, single-step LJ energy+forces
equality, the Ewald reciprocal term (with net charge), ``ShardedPotential``'s
energy-only ``psum``, and that owned-buffer overflow raises instead of
corrupting results.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import CapacityError, FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.domain import (
    MortonPartitioner,
    Sharded,
    make_repartitioner_from_state,
    owned_subset,
    sharded_local_edges,
)
from kups.core.lens import bind, lens
from kups.core.neighborlist import CellListNeighborList
from kups.core.patch import IdPatch, WithPatch
from kups.core.potential import EMPTY, PotentialOut, ShardedPotential
from kups.core.result import as_result_function
from kups.core.sharding import device_put_replicated, shard_axis
from kups.core.typing import (
    ExclusionId,
    InclusionId,
    Label,
    OriginDeviceId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, shard_map
from kups.potential.classical.ewald import (
    EwaldLongRangeInput,
    EwaldParameters,
    estimate_ewald_parameters,
    ewald_long_range_energy,
    kvecs_from_kmax,
)
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    lennard_jones_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph, PointCloud

_AXIS = shard_axis(OriginDeviceId)
_REPL = jax.sharding.PartitionSpec()


@dataclass
class _Particle:
    positions: jax.Array
    labels: Index[Label]
    system: Index[SystemId]
    origin: Index[OriginDeviceId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class _ChargedParticle:
    positions: jax.Array
    charges: jax.Array
    system: Index[SystemId]
    origin: Index[OriginDeviceId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class _System:
    cell: PeriodicCell


@dataclass
class _DDState:
    particles: Table[ParticleId, _Particle]
    systems: Table[SystemId, _System]
    neighborlist: CellListNeighborList


def _mesh() -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.array(jax.devices()), axis_names=(_AXIS,))


def _round_robin(n: int, n_devices: int) -> Table[ParticleId, _Particle]:
    return Table.arange(
        _Particle(
            positions=jnp.zeros((n, 3)),
            labels=Index.new([Label("Ar")] * n),
            system=Index.zeros(n, label=SystemId),
            origin=Index.integer(
                np.arange(n) % n_devices, n=n_devices, label=OriginDeviceId
            ),
            inclusion=Index.zeros(n, label=InclusionId),
            exclusion=Index.integer(np.arange(n), n=n, label=ExclusionId),
        ),
        label=ParticleId,
    )


def _make_state(n_per_axis: int, box: float, cutoff: float, n_devices: int) -> _DDState:
    n = n_per_axis**3
    rng = np.random.default_rng(0)
    grid = np.stack(
        np.meshgrid(*[np.arange(n_per_axis)] * 3, indexing="ij"), -1
    ).reshape(-1, 3) * (box / n_per_axis)
    pos = jnp.asarray((grid + rng.uniform(-0.3, 0.3, size=grid.shape)) % box)
    parts = _round_robin(n, n_devices)
    parts = bind(parts).focus(lambda t: t.data.positions).set(pos)
    state = _DDState(
        particles=parts,
        systems=Table(
            (SystemId(0),),
            _System(PeriodicCell(TriclinicFrame.from_matrix(box * jnp.eye(3)[None]))),
        ),
        neighborlist=CellListNeighborList(
            avg_candidates=FixedCapacity(256),
            avg_edges=FixedCapacity(128),
            cells=FixedCapacity(1024),
            avg_image_candidates=FixedCapacity(256),
            cutoffs=Table((SystemId(0),), jnp.array([cutoff])),
        ),
    )
    sl = lens(lambda s: s, cls=_DDState)
    return make_repartitioner_from_state(sl, MortonPartitioner(), n_devices)(state)


# --------------------------------------------------------------------------- #
# Sharded mask/combine algebra.
# --------------------------------------------------------------------------- #
def test_sharded_partial_then_combine_reconstructs_global() -> None:
    """Each node is owned by exactly one device, so psum(owned_only(x)) == x.
    A no-op owned_only would yield n_devices * x; a wrong mask would
    drop/duplicate rows."""
    if len(jax.devices()) < 2:
        pytest.skip("needs a multi-device mesh for the sharded branch")
    n_dev = len(jax.devices())
    n = 6
    parts = device_put_replicated(_round_robin(n, n_dev), _mesh())

    def per_device(p: Table[ParticleId, _Particle]) -> jax.Array:
        d = Sharded[_Particle]()
        x = jnp.arange(n, dtype=float)
        return d.combine_across_shards(d.owned_only(p, x))

    out = shard_map(per_device, in_specs=(_REPL,), out_specs=_REPL, mesh=_mesh())(parts)
    assert jnp.allclose(out, jnp.arange(n, dtype=float))


def test_sharded_owned_only_zeros_non_owned_rows() -> None:
    """Directly observe the mask: on each device, only owned rows survive."""
    if len(jax.devices()) < 2:
        pytest.skip("needs a multi-device mesh for the sharded branch")
    n_dev = len(jax.devices())
    n = 6
    origin_ids = np.arange(n) % n_dev
    parts = device_put_replicated(_round_robin(n, n_dev), _mesh())

    def per_device(p: Table[ParticleId, _Particle]) -> jax.Array:
        masked = Sharded[_Particle]().owned_only(p, jnp.ones(n))
        return jnp.sum(masked)[None]  # this device's owned count

    out = shard_map(
        per_device,
        in_specs=(_REPL,),
        out_specs=jax.sharding.PartitionSpec(_AXIS),
        mesh=_mesh(),
    )(parts)
    expected = np.bincount(origin_ids, minlength=n_dev).astype(float)
    assert jnp.allclose(jnp.asarray(out), expected)


def test_sharded_owned_only_broadcasts_over_trailing_axes() -> None:
    # Shape-preserving for non-scalar per-node payloads (the Ewald rho path).
    if len(jax.devices()) < 2:
        pytest.skip("needs a multi-device mesh for the sharded branch")
    n_dev = len(jax.devices())
    n = 4
    parts = device_put_replicated(_round_robin(n, n_dev), _mesh())

    def per_device(p: Table[ParticleId, _Particle]) -> jax.Array:
        d = Sharded[_Particle]()
        return d.combine_across_shards(d.owned_only(p, jnp.ones((n, 3, 2))))

    out = shard_map(per_device, in_specs=(_REPL,), out_specs=_REPL, mesh=_mesh())(parts)
    assert jnp.allclose(out, jnp.ones((n, 3, 2)))


# --------------------------------------------------------------------------- #
# ShardedPotential: psum the energy, pass gradients through UNTOUCHED.
# --------------------------------------------------------------------------- #
@dataclass
class _PartialEnergyPotential:
    """Fake DD potential: per-device partial energy, transpose-summed gradients."""

    def __call__(self, state: jax.Array, patch: None = None):
        device = jax.lax.axis_index(_AXIS)
        energies = Table.arange(jnp.array([1.0 + device]) * state[0], label=SystemId)
        gradients = jnp.full((3,), 7.0) * state[0]
        return WithPatch(PotentialOut(energies, gradients, EMPTY), IdPatch[jax.Array]())


def test_sharded_potential_psums_energy_and_passes_gradients_through() -> None:
    """Psumming the gradients too would multiply forces by the device count —
    the exact bug class the wrapper exists to prevent."""
    if len(jax.devices()) < 2:
        pytest.skip("needs a multi-device mesh for the sharded branch")
    n_dev = len(jax.devices())
    wrapped = ShardedPotential(_PartialEnergyPotential())

    def per_device(state: jax.Array) -> tuple[jax.Array, jax.Array]:
        out = wrapped(state).data
        return out.total_energies.data, out.gradients

    energy, gradients = shard_map(
        per_device, in_specs=(_REPL,), out_specs=(_REPL, _REPL), mesh=_mesh()
    )(jnp.ones(1))
    # Partials 1 + d for d in 0..n_dev-1.
    assert jnp.allclose(energy, n_dev + n_dev * (n_dev - 1) / 2)
    assert jnp.allclose(gradients, jnp.full((3,), 7.0))


# --------------------------------------------------------------------------- #
# Owned-buffer overflow must RAISE under the assertion interpreter.
# --------------------------------------------------------------------------- #
def test_owned_buffer_overflow_raises() -> None:
    """An undersized ``cap_owned`` must fail the assertion interpreter, not
    silently drop owned atoms. The recorded requirement (max owned count over
    all devices) is computed from the replicated origin labels, so the
    assertion leaves the ``shard_map`` through the default replicated
    assertion context."""
    if len(jax.devices()) < 2:
        pytest.skip("needs a multi-device mesh for the sharded branch")
    n_dev = len(jax.devices())
    n = 4 * n_dev
    parts = device_put_replicated(_round_robin(n, n_dev), _mesh())
    too_small = FixedCapacity(1)  # every device owns 4 atoms

    def per_device(p: Table[ParticleId, _Particle]) -> jax.Array:
        owned = owned_subset(p, jax.lax.axis_index(_AXIS), too_small)
        return jnp.sum(owned.indices)[None]

    fn = shard_map(
        per_device,
        in_specs=(_REPL,),
        out_specs=jax.sharding.PartitionSpec(_AXIS),
        mesh=_mesh(),
    )
    result = as_result_function(fn)(parts)
    with pytest.raises(CapacityError):
        result.raise_assertion()


# --------------------------------------------------------------------------- #
# Single-step LJ: DD energy and forces match single-device to tight tolerance.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cap_slack", [0, 3])
def test_sharded_lj_matches_single_device_energy_and_forces(cap_slack: int) -> None:
    # cap_slack > 0 forces padding rows in each device's owned buffer; the
    # result must be unchanged (padding rows carry the OOB sentinel id and are
    # dropped by the neighbor list's segment masks).
    if len(jax.devices()) < 2:
        pytest.skip("DD force gate needs a multi-device mesh")

    n_devices = len(jax.devices())
    box, cutoff = 12.0, 3.0
    state = _make_state(n_per_axis=6, box=box, cutoff=cutoff, n_devices=n_devices)
    lj = LennardJonesParameters.from_dict(
        cutoff=cutoff, parameters={"Ar": (1.0, 0.5)}, mixing_rule="lorentz_berthelot"
    )
    pos0 = state.particles.data.positions

    # Single-device reference: whole radius graph, stock LJ energy + forces.
    whole_edges = state.neighborlist(state.particles, state.systems)

    def energy_ref(positions: jax.Array) -> jax.Array:
        particles = (
            bind(state.particles).focus(lambda t: t.data.positions).set(positions)
        )
        g = HyperGraph(particles, state.systems, whole_edges)
        return jnp.sum(lennard_jones_energy(GraphPotentialInput(lj, g)).data.data)

    e_ref, grad_ref = jax.value_and_grad(energy_ref)(pos0)

    # Domain-decomposed: owned-incident shard + the SAME stock energy.
    cap = FixedCapacity(
        int(np.bincount(np.asarray(state.particles.data.origin.indices)).max())
        + cap_slack
    )
    state_r = device_put_replicated(state, _mesh())

    def per_device(s: _DDState) -> tuple[jax.Array, jax.Array]:
        # Edges are fixed; positions are the differentiation variable.
        edges = sharded_local_edges(
            s.particles, s.systems, s.neighborlist, jax.lax.axis_index(_AXIS), cap
        )

        def energy(positions: jax.Array) -> jax.Array:
            particles = (
                bind(s.particles).focus(lambda t: t.data.positions).set(positions)
            )
            g = HyperGraph(
                particles, s.systems, edges, decomposition=Sharded[_Particle]()
            )
            return jnp.sum(lennard_jones_energy(GraphPotentialInput(lj, g)).data.data)

        # `energy` is this device's owned partial: psum the ENERGY output;
        # forces (gradients w.r.t. the replicated positions) are mesh-summed by
        # the transpose — do not psum them.
        e, grad = jax.value_and_grad(energy)(s.particles.data.positions)
        return jax.lax.psum(e, _AXIS), grad

    e_dd, grad_dd = shard_map(
        per_device, in_specs=(_REPL,), out_specs=(_REPL, _REPL), mesh=_mesh()
    )(state_r)

    assert jnp.allclose(e_dd, e_ref, rtol=1e-10), (
        f"energy {float(e_dd)} != {float(e_ref)}"
    )
    assert jnp.allclose(grad_dd, grad_ref, rtol=1e-8, atol=1e-8), (
        f"max force diff {float(jnp.abs(grad_dd - grad_ref).max())}"
    )


# --------------------------------------------------------------------------- #
# Ewald reciprocal term with NET CHARGE: the owned partial structure factor
# must be combined across the mesh BEFORE squaring, and the per-atom
# net-charge correction must reduce to a per-device partial (a closed-form
# E_net would be counted once per device).
# --------------------------------------------------------------------------- #
def test_sharded_ewald_reciprocal_matches_single_device() -> None:
    if len(jax.devices()) < 2:
        pytest.skip("DD reciprocal gate needs a multi-device mesh")

    n_devices = len(jax.devices())
    n_per_axis, box = 4, 20.0
    n = n_per_axis**3
    rng = np.random.default_rng(1)
    grid = np.stack(
        np.meshgrid(*[np.arange(n_per_axis)] * 3, indexing="ij"), -1
    ).reshape(-1, 3) * (box / n_per_axis)
    pos = jnp.asarray((grid + rng.uniform(-0.2, 0.2, size=grid.shape)) % box)
    # Deliberately NON-neutral so ewald_net_charge_energy contributes.
    q = rng.uniform(-1.0, 1.0, size=n) + 0.2
    particles = Table.arange(
        _ChargedParticle(
            positions=pos,
            charges=jnp.asarray(q),
            system=Index.zeros(n, label=SystemId),
            origin=Index.integer(np.zeros(n, int), n=n_devices, label=OriginDeviceId),
            inclusion=Index.zeros(n, label=InclusionId),
            exclusion=Index.integer(np.arange(n), n=n, label=ExclusionId),
        ),
        label=ParticleId,
    )
    cell = PeriodicCell(TriclinicFrame.from_matrix(box * jnp.eye(3)))
    systems = Table(
        (SystemId(0),),
        _System(PeriodicCell(TriclinicFrame.from_matrix(box * jnp.eye(3)[None]))),
    )
    origin = MortonPartitioner()(particles, systems, n_devices)
    particles = bind(particles).focus(lambda t: t.data.origin).set(origin)
    est = estimate_ewald_parameters(jnp.asarray(q), cell, epsilon_total=1e-4)
    params = EwaldParameters(
        alpha=Table((SystemId(0),), jnp.array([est.alpha])),
        cutoff=Table((SystemId(0),), jnp.array([est.real_cutoff])),
        reciprocal_lattice_shifts=Table(
            (SystemId(0),), kvecs_from_kmax(cell, est.k_max)[None]
        ),
    )

    def energy_ref(positions: jax.Array) -> jax.Array:
        p = bind(particles).focus(lambda t: t.data.positions).set(positions)
        inp = EwaldLongRangeInput(PointCloud(p, systems), params)
        return jnp.sum(ewald_long_range_energy(inp).data.data)

    e_ref, grad_ref = jax.value_and_grad(energy_ref)(particles.data.positions)

    parts_r = device_put_replicated(particles, _mesh())

    def per_device(
        parts: Table[ParticleId, _ChargedParticle],
    ) -> tuple[jax.Array, jax.Array]:
        def energy(positions: jax.Array) -> jax.Array:
            p = bind(parts).focus(lambda t: t.data.positions).set(positions)
            pc = PointCloud(p, systems, decomposition=Sharded[_ChargedParticle]())
            inp = EwaldLongRangeInput(pc, params)
            return jnp.sum(ewald_long_range_energy(inp).data.data)

        e, grad = jax.value_and_grad(energy)(parts.data.positions)
        return jax.lax.psum(e, _AXIS), grad

    e_dd, grad_dd = shard_map(
        per_device, in_specs=(_REPL,), out_specs=(_REPL, _REPL), mesh=_mesh()
    )(parts_r)

    assert jnp.allclose(e_dd, e_ref, rtol=1e-9), (
        f"reciprocal energy {float(e_dd)} != {float(e_ref)}"
    )
    assert jnp.allclose(grad_dd, grad_ref, rtol=1e-7, atol=1e-7), (
        f"max reciprocal force diff {float(jnp.abs(grad_dd - grad_ref).max())}"
    )
