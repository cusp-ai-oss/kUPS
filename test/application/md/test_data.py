# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for source-neutral molecular-dynamics data construction."""

from __future__ import annotations

import inspect
from pathlib import Path

import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import kups.application.md.data as md_data
from kups.application.md.data import (
    MdParameters,
    md_state_from_ase,
    md_state_from_particles,
)
from kups.application.utils.particles import Particles
from kups.core.cell import AnyPeriodicity, Cell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.typing import Label, ParticleId, SystemId


@pytest.fixture
def md_config() -> MdParameters:
    """Return a complete MD configuration for builder tests."""
    return MdParameters(
        temperature=300.0,
        time_step=1.0,
        friction_coefficient=0.01,
        thermostat_time_constant=100.0,
        target_pressure=1.0e5,
        pressure_coupling_time=1.0e3,
        compressibility=4.5e-10,
        minimum_scale_factor=0.9,
        integrator="verlet",
        initialize_momenta=False,
    )


def _cell() -> Cell[AnyPeriodicity]:
    return Cell.from_pbc(
        TriclinicFrame.from_matrix(
            jnp.array(
                [
                    [4.0, 0.0, 0.0],
                    [0.5, 5.0, 0.0],
                    [0.25, 0.75, 6.0],
                ]
            )
        ),
        (True, True, True),
    )


def _particles(
    system: Index[SystemId],
    *,
    keys: tuple[ParticleId, ...] | None = None,
) -> Table[ParticleId, Particles]:
    n_particles = len(system)
    if keys is None:
        keys = tuple(ParticleId(i) for i in range(n_particles))
    return Table(
        keys,
        Particles(
            positions=jnp.arange(n_particles * 3, dtype=float).reshape(n_particles, 3),
            masses=jnp.arange(1, n_particles + 1, dtype=float),
            atomic_numbers=jnp.full(n_particles, 18, dtype=int),
            charges=jnp.zeros(n_particles),
            labels=Index(
                (Label("Ar"),),
                jnp.zeros(n_particles, dtype=int),
                _cls=Label,
            ),
            system=system,
        ),
        _cls=ParticleId,
    )


def test_public_builder_signature_and_export() -> None:
    signature = inspect.signature(md_state_from_particles)

    assert tuple(signature.parameters) == (
        "particles",
        "cell",
        "config",
        "key",
    )
    assert signature.parameters["key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["key"].default is None
    assert "md_state_from_particles" in md_data.__all__


def test_builder_preserves_identity_and_geometry(md_config: MdParameters) -> None:
    system_key = SystemId(12)
    particle_keys = (ParticleId(4), ParticleId(9))
    particles = _particles(
        Index((system_key,), jnp.zeros(2, dtype=int)),
        keys=particle_keys,
    )
    cell = _cell()

    md_particles, systems = md_state_from_particles(particles, cell, md_config)

    assert md_particles.keys == particle_keys
    assert md_particles.data.system.keys == (system_key,)
    npt.assert_array_equal(
        md_particles.data.system.indices,
        jnp.zeros(2, dtype=int),
    )
    assert systems.keys == (system_key,)
    npt.assert_array_equal(
        md_particles.data.positions,
        particles.data.positions,
    )
    assert systems.data.cell.vectors.shape == (1, 3, 3)
    npt.assert_array_equal(systems.data.cell.vectors[0], cell.vectors)


def test_empty_particles_are_rejected(md_config: MdParameters) -> None:
    particles = _particles(
        Index((SystemId(12),), jnp.empty((0,), dtype=int)),
        keys=(),
    )

    with pytest.raises(ValueError, match="particles.*at least one"):
        md_state_from_particles(particles, _cell(), md_config)


def test_multiple_system_keys_are_rejected(md_config: MdParameters) -> None:
    particles = _particles(
        Index((SystemId(12), SystemId(13)), jnp.array([0, 1])),
    )

    with pytest.raises(ValueError, match="particles.*exactly one system"):
        md_state_from_particles(particles, _cell(), md_config)


def test_invalid_reference_into_single_system_key_is_rejected(
    md_config: MdParameters,
) -> None:
    particles = _particles(
        Index((SystemId(12),), jnp.array([0, 1])),
    )

    with pytest.raises(ValueError, match="particles.*sole SystemId"):
        md_state_from_particles(particles, _cell(), md_config)


def test_batched_cell_is_rejected(md_config: MdParameters) -> None:
    particles = _particles(
        Index((SystemId(12),), jnp.zeros(2, dtype=int)),
    )

    with pytest.raises(ValueError, match=r"cell\.vectors.*\(3, 3\)"):
        md_state_from_particles(particles, _cell()[None], md_config)


def test_ase_wrapper_delegates_to_source_neutral_builder(
    md_config: MdParameters, monkeypatch: pytest.MonkeyPatch
) -> None:
    particles = _particles(
        Index((SystemId(0),), jnp.zeros(2, dtype=int)),
    )
    cell = _cell()
    key = jax.random.key(7)
    expected = (object(), object())
    calls: list[tuple[object, ...]] = []

    def fake_particles_from_ase(
        atoms: ase.Atoms | str | Path,
    ) -> tuple[Table[ParticleId, Particles], Cell[AnyPeriodicity], object]:
        calls.append(("adapter", atoms))
        return particles, cell, object()

    def fake_md_state_from_particles(
        actual_particles: Table[ParticleId, Particles],
        actual_cell: Cell[AnyPeriodicity],
        actual_config: MdParameters,
        *,
        key: jax.Array | None = None,
    ) -> tuple[object, object]:
        calls.append(("builder", actual_particles, actual_cell, actual_config, key))
        return expected

    monkeypatch.setattr(md_data, "particles_from_ase", fake_particles_from_ase)
    monkeypatch.setattr(
        md_data, "md_state_from_particles", fake_md_state_from_particles
    )

    actual = md_state_from_ase("input.cif", md_config, key=key)

    assert actual is expected
    assert calls == [
        ("adapter", "input.cif"),
        ("builder", particles, cell, md_config, key),
    ]


@pytest.mark.parametrize("use_path", [False, True], ids=["atoms", "file-path"])
def test_ase_wrapper_retains_input_forms(
    md_config: MdParameters, tmp_path: Path, use_path: bool
) -> None:
    atoms = ase.Atoms(
        "Ar2",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    source: ase.Atoms | str = atoms
    if use_path:
        path = tmp_path / "atoms.cif"
        ase.io.write(path, atoms)
        source = str(path)

    particles, systems = md_state_from_ase(source, md_config)

    assert particles.keys == (ParticleId(0), ParticleId(1))
    assert systems.keys == (SystemId(0),)
