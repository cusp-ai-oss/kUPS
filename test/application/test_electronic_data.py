# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""System electronic metadata survives ingestion, batching, and storage."""

import base64
import zlib
from dataclasses import replace

import ase
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.application.md.data import (
    MdParameters,
    md_state_from_ase,
    md_state_from_particles_and_cell,
)
from kups.application.relaxation.data import relax_state_from_ase
from kups.application.utils.particles import particles_from_ase
from kups.core.data import Index, Table
from kups.core.lens import view
from kups.core.storage import (
    HDF5StorageReader,
    HDF5StorageWriter,
    Once,
    WriterGroupConfig,
)
from kups.core.typing import SystemId
from kups.core.utils.jax import no_post_init


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


# Compressed HDF5 initial states written at bc3399d044314bde4159f7c0176a283cd914bb0a.
# Retain the old serialized tree definitions; rebuilding them would miss schema drift.
_LEGACY_INITIAL_STATES = {
    "md": (
        "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"
    ),
    "relax": (
        "eJztWV1oHFUUvncmmUySzbpJE5ukTVzTRLI2nVQJpq0WEzdtVrZNSo0KSlkmu5PN6GRnnZklWUVpsT9Gp5DgVBAR"
        "RCkoFESL4JsiIj70QRBfCj74UxD0SRH1Qeo9d+7sbtZss9o02DgHlrn33HPOPfc792c5ZzE2drCpobOBF0Xk0VVG"
        "W3e4/d2M33Tl8tuTsbGjHFoIhUi/22Uvs+EGDmGEhHZEPk1JPTOjphPZvGUoykUm0daCaormPQJbH2FXYpJJCiGw"
        "Ups29Fw2yBU8Qy+8dPsb1AWMMCa2YIgXEW4jX538aokPfIPYQrgYsRXpnvIltKlIurdbmYzG4oEoWS1qQa0ALsM9"
        "qcmmmcjIcwrqwBhBtODnoixS2JrJT2cQAf+LC88dKYlurM4depfNxqJ7JwjfJhuGnJdSsiVLSUXTpBmDzCRNy6Yi"
        "WYaqLbOIgZ13atx2FBV5v9e67TFvKUyG++7Sm4Xo9pI+hzHbGEhARVdL+3wrws3h+1mwR9BQIxq6yrOd0NgYRZjj"
        "wHV3O9zy7E7afsy05ExKNlLhpD6XNRTTVPXMvvCsZWX3DQ7Oz89LT5tWSsooFp1Q7AKDHI+xIOKukhg4Pawx8Q8C"
        "VxUZ4q27ClgMESz4IhZ82bcSf2Ow6abY1GDM8yJuLlnCex44+68Ti8Nf9/TCTheGYa0Dq+++lDKjG3OyRZyVND2d"
        "IE1DXXidYfHKZ3+YleCs9rvBcGK8Es5P1gvOZH7pKwrnHljbrrXhBHZiRk5auvETO6rv//DWN//9o1oK35c37Kju"
        "dLZ+C3jC1Xam0+WlvLUPAQ6RMowTaUNOqUrGMsuvzvsYWMIo6N2zlt7qm15vd42AQ0vsusoW3D1/tv7TQud4zTnP"
        "VbvlCXlBU6elBPlK7svt2PVH8lOkMabMOC87kRMOlZISppGUQCCRs1TNsUPEETmnWQlDSaumZeSJ8DGnP47iaGIi"
        "juPYcuTTzrQ0Hp08RLdFyJ3T2yJeoB41VEsxxuHZj9KXjPG9vwDs4UBT8rSmoOJNd4AZerxnjaepZyBcQWR1KKuT"
        "LzkgqyhUjnY1wlX7dU3Nyh5mdYvoqLKWUDKKkc73HEOFBxnVIw9X+leMjpXSnmDFU+GTTz755JNPPvnkk08++eTT"
        "/5hu8ozm9+uVgrv7wx0Xb/J05K/rhcVdXb8seemz37a5vLNsTHgA1jv8LzMdz3R4c8RebFou4N1K8MZ/x3tj8NxO"
        "8eTK0RRYrQ0lrhPNtWmvsvdnwJtrgV55NU64AxDZdo0E0Y8sxRkYhXMZcutOIFlN1QkSeF7VCfpPdVx+tVJBr49b"
        "WdDTw65en5cN3GQFvSsPm5MQmNXLpMIWQFKkSVey309ud5UCwxAFwY1CVrZmzYrYQ5LUwx4u1C6bP1l5IwT2g1TA"
        "NUyzfzSylc1DrtAzD/nDgAQGmogJ0iPnMUd2kGkpWROFRFQ4ZRwqnsALn497tQKfNiUdrznXyU7vDSg3rKg39MeF"
        "OI7YwSdzWVNK6oZCXwbHDk4ZalJTM2ryILwdRJPIc5ZTplxmiYvM1tvBwzT3fkhPF1S5OF8i0DRGHyIlVRjn4wJz"
        "ZXFx8YxzyiFSDY/IyVxuLgruUJG6VSbEkdnGKh0LrfBiS8mcs21kwvYKM/ARO0zBkbNZTU26r6ehaPKC24Qr3bED"
        "R4HzUJ6c3DkT3K2Lh5h1O1TE1spn1UzasUVX8sEUiPadck7AkiOnHbu1KEtfFQsqOI5dSys5rt1my1GgSNT/weCf"
        "4d3nac0qjMjfjN4RclGgEYyCH3N0MHaAXj8YPc8NUIY4WthiI666J/EaalxTBZUxAiN/AYH0SEI="
    ),
}


@pytest.mark.parametrize("kind,key,energy", [("md", 12, 2.5), ("relax", 37, 3.5)])
def test_prechange_initial_state_reads(kind, key, energy, tmp_path):
    """Legacy initial data remain readable without inventing stored metadata."""
    path = tmp_path / "legacy.h5"
    path.write_bytes(zlib.decompress(base64.b64decode(_LEGACY_INITIAL_STATES[kind])))
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
