# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""System electronic metadata survives ingestion, batching, and storage."""

from dataclasses import replace

import ase
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array

import kups.application.md.data as md_data
import kups.application.relaxation.data as relax_data
from kups.application.md.data import (
    MdParameters,
    VerletParams,
    md_state_from_ase,
    md_state_from_particles_and_cell,
)
from kups.application.relaxation.data import relax_state_from_ase
from kups.application.utils.particles import particles_from_ase
from kups.core.cell import Cell
from kups.core.data import Index, Table
from kups.core.lens import view
from kups.core.storage import (
    HDF5StorageReader,
    HDF5StorageWriter,
    Once,
    WriterGroupConfig,
)
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, no_post_init


def _config():
    """Return a deterministic MD configuration without sampled momenta."""
    return MdParameters(
        temperature=300,
        time_step=1,
        friction_coefficient=0.01,
        thermostat_time_constant=100,
        target_pressure=1e5,
        pressure_coupling_time=1000,
        compressibility=4.5e-10,
        minimum_scale_factor=0.9,
        integrator="verlet",
    )


def _atoms(n=2):
    """Use unequal molecule sizes and a finite skewed cell."""
    return ase.Atoms(
        "H" * n,
        positions=np.arange(n * 3).reshape(n, 3) * 0.2,
        cell=[[8, 0, 0], [1, 9, 0], [0.5, 0.25, 10]],
        pbc=False,
    )


@pytest.fixture(params=["md", "relax", "particles"])
def build(request):
    """Exercise all three public ingestion boundaries."""
    if request.param == "md":
        return lambda atoms, **kw: md_state_from_ase(atoms, _config(), **kw)
    if request.param == "relax":
        return relax_state_from_ase

    def from_particles(atoms, **kw):
        particles, cell, _ = particles_from_ase(atoms)
        return md_state_from_particles_and_cell(particles, cell, _config(), **kw)

    return from_particles


def test_defaults_and_integer_scalars(build):
    """Omitted metadata defaults to 0/1 and integer scalars survive ingestion."""
    _, systems = build(_atoms())
    npt.assert_array_equal(systems.data.total_charge, [0])
    npt.assert_array_equal(systems.data.spin_multiplicity, [1])
    _, systems = build(
        _atoms(), total_charge=np.array(-2), spin_multiplicity=jnp.array(3)
    )
    npt.assert_array_equal(systems.data.total_charge, [-2])
    npt.assert_array_equal(systems.data.spin_multiplicity, [3])
    assert systems.data.total_charge.dtype == jnp.asarray(0).dtype
    _, systems = build(_atoms(), spin_multiplicity=0)
    npt.assert_array_equal(systems.data.spin_multiplicity, [0])


@pytest.mark.parametrize("builder", [md_state_from_ase, relax_state_from_ase])
def test_selected_source_precedence(builder):
    """Independent explicit overrides ignore invalid info without mutating it."""
    atoms = _atoms()
    atoms.info.update(charge="stale", spin=2)
    args = (_config(),) if builder is md_state_from_ase else ()
    _, systems = builder(atoms, *args, total_charge=1)
    npt.assert_array_equal(systems.data.total_charge, [1])
    npt.assert_array_equal(systems.data.spin_multiplicity, [2])
    assert atoms.info == {"charge": "stale", "spin": 2}
    atoms.info.update(charge=-1, spin=None)
    _, systems = builder(atoms, *args, spin_multiplicity=3)
    npt.assert_array_equal(systems.data.total_charge, [-1])
    with pytest.raises(TypeError, match="spin"):
        builder(atoms, *args, spin_multiplicity=None)
    atoms.info.update(charge=None, spin=1)
    with pytest.raises(TypeError, match="charge"):
        builder(atoms, *args)


@pytest.mark.parametrize("field", ["total_charge", "spin_multiplicity"])
@pytest.mark.parametrize(
    "value,error",
    [
        (True, TypeError),
        (np.bool_(False), TypeError),
        (1.0, TypeError),
        (np.array(2.0), TypeError),
        (1j, TypeError),
        ("1", TypeError),
        ([1], ValueError),
        (np.array([[1]]), ValueError),
        (2**100, ValueError),
        (np.uint64(2**64 - 1), ValueError),
    ],
)
def test_invalid_metadata_rejected_before_conversion(build, field, value, error):
    """Wrong categories, shapes, and integer overflow cannot silently coerce."""
    with pytest.raises(error, match=field) as exc:
        build(_atoms(), **{field: value})
    assert "unexpected keyword" not in str(exc.value)


def test_negative_spin_rejected(build):
    """Ingestion permits the zero convention but rejects negative multiplicity."""
    with pytest.raises(ValueError, match="spin_multiplicity"):
        build(_atoms(), spin_multiplicity=-1)


def test_active_integer_width_overflow(build):
    """Int64 values cannot wrap when the active JAX integer width is int32."""
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="total_charge"):
            build(_atoms(), total_charge=np.int64(2**31))


def test_file_geometry_and_metadata_share_cached_read(tmp_path, monkeypatch):
    """Both builders and the public three-item API share one cached file read."""
    atoms = _atoms()
    atoms.info.update(charge=1, spin=2)
    calls = []

    def read(path, **kwargs):
        calls.append((path, kwargs))
        return iter([atoms])

    monkeypatch.setattr(ase.io, "iread", read)
    path = tmp_path / "metadata.cif"
    first = particles_from_ase(path)
    assert len(first) == 3
    assert particles_from_ase(path) is first
    for build in (lambda p: md_state_from_ase(p, _config()), relax_state_from_ase):
        particles, systems = build(path)
        npt.assert_array_equal(particles.data.positions, first[0].data.positions)
        npt.assert_array_equal(systems.data.total_charge, [1])
        npt.assert_array_equal(systems.data.spin_multiplicity, [2])
    assert calls == [(path, {"index": -1, "store_tags": True})]
    assert atoms.info == {"charge": 1, "spin": 2}


@pytest.mark.parametrize("builder", [md_state_from_ase, relax_state_from_ase])
def test_batch_selection_jit_and_shape_only_defaults(builder):
    """Union and selection retain metadata through JIT and default construction."""
    args = (_config(),) if builder is md_state_from_ase else ()
    first = builder(_atoms(2), *args, total_charge=-1, spin_multiplicity=2)
    second = builder(_atoms(3), *args, total_charge=2, spin_multiplicity=3)
    particles, systems = Table.union([first[0], second[0]], [first[1], second[1]])
    npt.assert_array_equal(particles.data.system.counts.data, [2, 3])
    npt.assert_array_equal(systems.data.total_charge, [-1, 2])
    npt.assert_array_equal(systems.data.spin_multiplicity, [2, 3])
    keyed = Table((SystemId(12), SystemId(37)), systems.data)
    selected = keyed[Index(keyed.keys, jnp.array([1, 0]))]
    npt.assert_array_equal(selected.total_charge, [2, -1])
    result = jax.jit(lambda s: s)(keyed)
    assert result.keys == keyed.keys
    npt.assert_array_equal(result.data.spin_multiplicity, [2, 3])
    defaults = jax.jit(lambda s: replace(s, total_charge=None, spin_multiplicity=None))(
        systems.data
    )
    npt.assert_array_equal(defaults.total_charge, [0, 0])
    npt.assert_array_equal(defaults.spin_multiplicity, [1, 1])
    assert isinstance(defaults.total_charge, jax.Array)
    with no_post_init():
        names = jax.tree.map(lambda _: "dataset", systems)
    assert names.data.total_charge == "dataset"


def test_noncanonical_md_system_identity():
    """The source-neutral builder keeps the supplied system label and geometry."""
    particles, cell, _ = particles_from_ase(_atoms())
    particles = particles.set_data(
        replace(particles.data, system=Index((SystemId(91),), jnp.zeros(2, dtype=int)))
    )
    result, systems = md_state_from_particles_and_cell(
        particles, cell, _config(), total_charge=2, spin_multiplicity=3
    )
    assert result.keys == particles.keys
    assert systems.keys == (SystemId(91),)
    npt.assert_array_equal(result.data.positions, particles.data.positions)
    npt.assert_array_equal(systems.data.total_charge, [2])
    npt.assert_array_equal(systems.data.cell.vectors[0], cell.vectors)


@pytest.mark.parametrize("builder", [md_state_from_ase, relax_state_from_ase])
def test_initial_hdf5_full_and_selective_reads(builder, tmp_path):
    """New initial states retain metadata in full and dataset-selected reads."""
    args = (_config(),) if builder is md_state_from_ase else ()
    _, systems = builder(_atoms(), *args, total_charge=-1, spin_multiplicity=2)
    path = tmp_path / "initial.h5"
    with HDF5StorageWriter(
        path, WriterGroupConfig(lambda x: x, Once()), systems, total_steps=1
    ) as writer:
        writer.log(systems, 0)
    with HDF5StorageReader(path) as reader:
        group = reader.focus_group("group")
        full = group.read(...)
        npt.assert_array_equal(full.data.total_charge, [-1])
        npt.assert_array_equal(full.data.spin_multiplicity, [2])
        npt.assert_array_equal(
            group.read(..., select=view(lambda s: s.data.total_charge)), [-1]
        )
        selected = group.read(..., select=view(lambda s: s.data.cell))
        for actual, expected in zip(
            jax.tree.leaves(selected), jax.tree.leaves(systems.data.cell), strict=True
        ):
            npt.assert_array_equal(actual, expected)


# Pickle stores classes by import path. These stand-ins take the original names
# while writing, so the files carry the same tree definitions as files written
# before the metadata fields existed.
@dataclass
class _PrechangeMDSystems:
    __module__ = md_data.__name__
    __qualname__ = "MDSystems"

    cell: Cell
    integrator_params: VerletParams
    cell_gradients: Cell
    cell_momentum: Array
    potential_energy: Array


@dataclass
class _PrechangeRelaxSystems:
    __module__ = relax_data.__name__
    __qualname__ = "RelaxSystems"

    cell: Cell
    cell_gradients: Cell
    potential_energy: Array


def _write_prechange_initial_state(kind, key, energy, path, monkeypatch):
    """Write an initial-state file in the layout used before the metadata fields."""
    _, systems = relax_state_from_ase(_atoms())
    cell, gradients = systems.data.cell, systems.data.cell_gradients
    energies = jnp.array([energy])
    if kind == "md":
        module, cls = md_data, _PrechangeMDSystems
        data = cls(
            cell,
            VerletParams(jnp.array([0.1])),
            gradients,
            jnp.zeros((1, 3, 3)),
            energies,
        )
    else:
        module, cls = relax_data, _PrechangeRelaxSystems
        data = cls(cell, gradients, energies)
    state = Table((SystemId(key),), data)
    with monkeypatch.context() as patch:
        patch.setattr(module, cls.__qualname__, cls)
        with HDF5StorageWriter(
            path, WriterGroupConfig(lambda x: x, Once()), state, total_steps=1
        ) as writer:
            writer.log(state, 0)


@pytest.mark.parametrize("kind,key,energy", [("md", 12, 2.5), ("relax", 37, 3.5)])
def test_prechange_initial_state_reads(kind, key, energy, tmp_path, monkeypatch):
    """Legacy initial data remain readable without inventing stored metadata."""
    path = tmp_path / "legacy.h5"
    _write_prechange_initial_state(kind, key, energy, path, monkeypatch)
    with HDF5StorageReader(path) as reader:
        group = reader.focus_group("group")
        full = group.read(...)
        assert full.keys == (SystemId(key),)
        npt.assert_array_equal(full.data.potential_energy, [energy])
        npt.assert_array_equal(
            group.read(..., select=view(lambda s: s.data.potential_energy)), [energy]
        )
        assert full.data.total_charge is None
        assert full.data.spin_multiplicity is None
