# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.md.data."""

from __future__ import annotations

import inspect
from pathlib import Path

import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import kups.application.md.data as md_data
from kups.application.md.data import (
    BAOABLangevinParams,
    BAOABNPTLangevinParams,
    CSVRNPTParams,
    CSVRParams,
    MdParameters,
    MDParticles,
    MDSystems,
    VerletParams,
    md_state_from_ase,
    md_state_from_particles_and_cell,
)
from kups.application.utils.particles import Particles, particles_from_arrays
from kups.core.cell import AnyPeriodicity, Cell, TriclinicFrame
from kups.core.constants import BOLTZMANN_CONSTANT, FEMTO_SECOND, PASCAL
from kups.core.data import Index, Table
from kups.core.typing import ExclusionId, Label, ParticleId, SystemId


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
    """Return an unbatched triclinic test cell."""
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
    """Build a one-system particle table for builder tests."""
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


def _assert_tree_allclose(actual: object, expected: object) -> None:
    """Compare two PyTrees numerically, including structure."""
    actual_structure = jax.tree_util.tree_structure(actual)
    expected_structure = jax.tree_util.tree_structure(expected)
    assert actual_structure == expected_structure
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        npt.assert_allclose(actual_leaf, expected_leaf, rtol=1e-12, atol=1e-12)


def test_public_builder_signature_and_export() -> None:
    """Public signature, keyword-only key, and module export."""
    signature = inspect.signature(md_state_from_particles_and_cell)

    assert tuple(signature.parameters) == (
        "particles",
        "cell",
        "config",
        "key",
    )
    assert signature.parameters["key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["key"].default is None
    assert "md_state_from_particles_and_cell" in md_data.__all__


def test_builder_preserves_identity_and_geometry(md_config: MdParameters) -> None:
    """Particle keys, the sole SystemId, and geometry are preserved."""
    system_key = SystemId(12)
    particle_keys = (ParticleId(4), ParticleId(9))
    particles = _particles(
        Index((system_key,), jnp.zeros(2, dtype=int)),
        keys=particle_keys,
    )
    cell = _cell()

    md_particles, systems = md_state_from_particles_and_cell(particles, cell, md_config)

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
    """Empty particle tables raise ValueError."""
    particles = _particles(
        Index((SystemId(12),), jnp.empty((0,), dtype=int)),
        keys=(),
    )

    with pytest.raises(ValueError, match="particles.*at least one"):
        md_state_from_particles_and_cell(particles, _cell(), md_config)


def test_multiple_system_keys_are_rejected(md_config: MdParameters) -> None:
    """Multiple SystemId keys raise ValueError."""
    particles = _particles(
        Index((SystemId(12), SystemId(13)), jnp.array([0, 1])),
    )

    with pytest.raises(ValueError, match="particles.*exactly one system"):
        md_state_from_particles_and_cell(particles, _cell(), md_config)


def test_invalid_reference_into_single_system_key_is_rejected(
    md_config: MdParameters,
) -> None:
    """Indices that do not select the sole SystemId raise ValueError."""
    particles = _particles(
        Index((SystemId(12),), jnp.array([0, 1])),
    )

    with pytest.raises(ValueError, match="particles.*sole SystemId"):
        md_state_from_particles_and_cell(particles, _cell(), md_config)


def test_column_vector_system_indices_are_rejected(md_config: MdParameters) -> None:
    """Column-vector system.indices raise ValueError."""
    particles = _particles(
        Index((SystemId(12),), jnp.zeros((2, 1), dtype=int)),
    )

    with pytest.raises(
        ValueError, match=r"particles.*one system reference per particle"
    ):
        md_state_from_particles_and_cell(particles, _cell(), md_config)


def test_batched_cell_is_rejected(md_config: MdParameters) -> None:
    """A cell whose vectors are not shape (3, 3) raises ValueError."""
    particles = _particles(
        Index((SystemId(12),), jnp.zeros(2, dtype=int)),
    )

    with pytest.raises(ValueError, match=r"cell\.vectors.*\(3, 3\)"):
        md_state_from_particles_and_cell(particles, _cell()[None], md_config)


def test_ase_wrapper_delegates_to_source_neutral_builder(
    md_config: MdParameters, monkeypatch: pytest.MonkeyPatch
) -> None:
    """md_state_from_ase forwards adapter output to the source-neutral builder."""
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

    def fake_md_state_from_particles_and_cell(
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
        md_data,
        "md_state_from_particles_and_cell",
        fake_md_state_from_particles_and_cell,
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
    """ASE Atoms and file-path inputs remain supported."""
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


def test_array_builder_matches_ase_for_skewed_cell(
    md_config: MdParameters,
) -> None:
    """Array and ASE paths agree for the same skewed-cell input."""
    positions = jnp.array(
        [
            [0.4, 0.8, 1.2],
            [1.5, 0.3, 2.1],
            [0.7, 1.1, 0.5],
        ]
    )
    cell_vectors = jnp.array(
        [
            [0.0, 2.0, 0.0],
            [1.5, 0.2, 0.0],
            [0.3, 0.4, 3.0],
        ]
    )
    masses = jnp.array([12.5, 16.25, 14.0])
    atomic_numbers = jnp.array([6, 8, 7])
    charges = jnp.array([0.2, -0.3, 0.1])
    labels = ["carbon-site", "oxygen-site", "nitrogen-site"]
    atoms = ase.Atoms(
        "CON",
        positions=np.asarray(positions),
        cell=np.asarray(cell_vectors),
        masses=np.asarray(masses),
        pbc=(True, False, True),
    )
    atoms.info["_atom_type_partial_charge"] = np.asarray(charges)
    atoms.info["_atom_site_label"] = labels
    key = jax.random.key(23)

    array_particles, array_cell, _ = particles_from_arrays(
        positions=positions,
        cell_vectors=cell_vectors,
        periodicity=(True, False, True),
        masses=masses,
        atomic_numbers=atomic_numbers,
        charges=charges,
        labels=labels,
    )
    actual_particles, actual_systems = md_state_from_particles_and_cell(
        array_particles,
        array_cell,
        md_config,
        key=key,
    )
    expected_particles, expected_systems = md_state_from_ase(
        atoms,
        md_config,
        key=key,
    )

    assert not jnp.allclose(cell_vectors, jnp.tril(cell_vectors))
    assert actual_particles.keys == expected_particles.keys
    assert actual_systems.keys == expected_systems.keys
    _assert_tree_allclose(actual_particles.data, expected_particles.data)
    _assert_tree_allclose(actual_systems.data, expected_systems.data)


def test_canonical_array_outputs_support_table_union(
    md_config: MdParameters,
) -> None:
    """Canonical source-adapter outputs remain compatible with Table.union()."""
    outputs = []
    for offset in (0.0, 0.5):
        particles, cell, _ = particles_from_arrays(
            positions=jnp.array(
                [
                    [0.2 + offset, 0.4, 0.6],
                    [1.1 + offset, 1.3, 1.5],
                ]
            ),
            cell_vectors=jnp.diag(jnp.array([4.0, 5.0, 6.0])),
            periodicity=(True, True, True),
            masses=jnp.array([22.99, 35.45]),
            atomic_numbers=jnp.array([11, 17]),
            charges=jnp.array([0.75, -0.75]),
            labels=["Na", "Cl"],
        )
        outputs.append(md_state_from_particles_and_cell(particles, cell, md_config))

    merged_particles, merged_systems = Table.union(
        [outputs[0][0], outputs[1][0]],
        [outputs[0][1], outputs[1][1]],
    )

    assert merged_particles.keys == tuple(ParticleId(i) for i in range(4))
    assert merged_systems.keys == (SystemId(0), SystemId(1))
    assert merged_particles.data.system.keys == (SystemId(0), SystemId(1))
    npt.assert_array_equal(merged_particles.data.system.indices, [0, 0, 1, 1])
    assert merged_particles.data.exclusion.keys == (
        ExclusionId(0),
        ExclusionId(1),
    )
    npt.assert_array_equal(merged_particles.data.exclusion.indices, [0, 1, 0, 1])
    assert merged_systems.data.cell.vectors.shape == (2, 3, 3)


def test_key_alone_controls_momentum_initialization(
    md_config: MdParameters,
) -> None:
    """key controls momenta; MdParameters.initialize_momenta is ignored."""
    particles = _particles(
        Index((SystemId(4),), jnp.zeros(3, dtype=int)),
    )
    initialize_config = md_config.model_copy(update={"initialize_momenta": True})
    zero_particles, _ = md_state_from_particles_and_cell(
        particles,
        _cell(),
        initialize_config,
        key=None,
    )

    npt.assert_array_equal(zero_particles.data.momenta, jnp.zeros((3, 3)))

    key = jax.random.key(41)
    no_initialize_config = md_config.model_copy(update={"initialize_momenta": False})
    sampled_particles, _ = md_state_from_particles_and_cell(
        particles,
        _cell(),
        no_initialize_config,
        key=key,
    )
    masses = particles.data.masses
    standard_deviation = jnp.sqrt(masses * md_config.temperature * BOLTZMANN_CONSTANT)
    raw_momenta = jax.random.normal(key, (3, 3)) * standard_deviation[:, None]
    expected_momenta = raw_momenta - masses[:, None] * (
        raw_momenta.sum(axis=0) / masses.sum()
    )

    npt.assert_allclose(
        sampled_particles.data.momenta,
        expected_momenta,
        rtol=1e-12,
        atol=1e-12,
    )
    npt.assert_allclose(
        sampled_particles.data.momenta.sum(axis=0),
        jnp.zeros(3),
        atol=1e-12,
    )


def test_builder_initial_zero_state_and_pytree_contract(
    md_config: MdParameters,
) -> None:
    """Zero-state fields, dtypes, shapes, and PyTree layout are preserved."""
    particles = _particles(
        Index((SystemId(4),), jnp.zeros(3, dtype=int)),
    )
    cell = _cell()

    md_particles, systems = md_state_from_particles_and_cell(
        particles,
        cell,
        md_config,
        key=None,
    )

    for field_name in (
        "positions",
        "masses",
        "atomic_numbers",
        "charges",
    ):
        actual = getattr(md_particles.data, field_name)
        expected = getattr(particles.data, field_name)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        npt.assert_array_equal(actual, expected)
    assert md_particles.data.labels.keys == particles.data.labels.keys
    assert md_particles.data.system.keys == particles.data.system.keys
    npt.assert_array_equal(
        md_particles.data.labels.indices,
        particles.data.labels.indices,
    )
    npt.assert_array_equal(
        md_particles.data.system.indices,
        particles.data.system.indices,
    )
    assert md_particles.data.position_gradients.shape == (3, 3)
    assert md_particles.data.position_gradients.dtype == particles.data.positions.dtype
    npt.assert_array_equal(
        md_particles.data.position_gradients,
        jnp.zeros_like(particles.data.positions),
    )
    assert md_particles.data.momenta.shape == (3, 3)
    assert md_particles.data.momenta.dtype == jnp.zeros((3, 3)).dtype
    npt.assert_array_equal(md_particles.data.momenta, jnp.zeros((3, 3)))
    assert md_particles.data.exclusion.keys == tuple(ExclusionId(i) for i in range(3))
    npt.assert_array_equal(md_particles.data.exclusion.indices, jnp.arange(3))

    assert systems.data.cell.vectors.shape == (1, 3, 3)
    assert systems.data.cell.vectors.dtype == cell.vectors.dtype
    npt.assert_array_equal(systems.data.cell.vectors[0], cell.vectors)
    assert jax.tree_util.tree_structure(
        systems.data.cell_gradients
    ) == jax.tree_util.tree_structure(systems.data.cell)
    for gradient_leaf, cell_leaf in zip(
        jax.tree_util.tree_leaves(systems.data.cell_gradients),
        jax.tree_util.tree_leaves(systems.data.cell),
        strict=True,
    ):
        assert gradient_leaf.shape == cell_leaf.shape
        assert gradient_leaf.dtype == cell_leaf.dtype
        npt.assert_array_equal(gradient_leaf, jnp.zeros_like(cell_leaf))
    assert systems.data.cell_momentum.shape == (1, 3, 3)
    assert systems.data.cell_momentum.dtype == jnp.zeros((1, 3, 3)).dtype
    npt.assert_array_equal(systems.data.cell_momentum, jnp.zeros((1, 3, 3)))
    assert systems.data.potential_energy.shape == (1,)
    assert systems.data.potential_energy.dtype == jnp.array([0.0]).dtype
    npt.assert_array_equal(systems.data.potential_energy, jnp.array([0.0]))

    particle_leaves, particle_structure = jax.tree_util.tree_flatten(md_particles.data)
    system_leaves, system_structure = jax.tree_util.tree_flatten(systems.data)
    assert len(particle_leaves) == 9
    assert len(system_leaves) == 5
    assert (
        type(jax.tree_util.tree_unflatten(particle_structure, particle_leaves))
        is MDParticles
    )
    assert (
        type(jax.tree_util.tree_unflatten(system_structure, system_leaves)) is MDSystems
    )


@pytest.mark.parametrize(
    ("integrator", "expected_type"),
    [
        ("verlet", VerletParams),
        ("baoab_langevin", BAOABLangevinParams),
        ("csvr", CSVRParams),
        ("csvr_npt", CSVRNPTParams),
        ("baoab_npt_langevin", BAOABNPTLangevinParams),
    ],
)
def test_integrator_parameters_and_unit_conversions(
    md_config: MdParameters,
    integrator: str,
    expected_type: type[object],
) -> None:
    """Integrator parameters and unit conversions match for every base integrator."""
    config = md_config.model_copy(update={"integrator": integrator})
    particles = _particles(
        Index((SystemId(4),), jnp.zeros(3, dtype=int)),
    )

    _, systems = md_state_from_particles_and_cell(particles, _cell(), config)
    params = systems.data.integrator_params

    assert type(params) is expected_type
    npt.assert_allclose(
        params.time_step,
        [config.time_step * FEMTO_SECOND],
        rtol=1e-12,
    )
    match params:
        case VerletParams():
            pass
        case BAOABLangevinParams():
            npt.assert_allclose(params.temperature, [config.temperature])
            npt.assert_allclose(
                params.friction_coefficient,
                [config.friction_coefficient / FEMTO_SECOND],
                rtol=1e-12,
            )
        case CSVRParams():
            npt.assert_allclose(params.temperature, [config.temperature])
            npt.assert_allclose(
                params.thermostat_time_constant,
                [config.thermostat_time_constant * FEMTO_SECOND],
                rtol=1e-12,
            )
        case CSVRNPTParams():
            npt.assert_allclose(params.temperature, [config.temperature])
            npt.assert_allclose(
                params.thermostat_time_constant,
                [config.thermostat_time_constant * FEMTO_SECOND],
                rtol=1e-12,
            )
            npt.assert_allclose(
                params.target_pressure,
                [config.target_pressure * PASCAL],
                rtol=1e-12,
            )
            npt.assert_allclose(
                params.pressure_coupling_time,
                [config.pressure_coupling_time * FEMTO_SECOND],
                rtol=1e-12,
            )
            npt.assert_allclose(
                params.compressibility,
                [config.compressibility / PASCAL],
                rtol=1e-12,
            )
            npt.assert_allclose(
                params.minimum_scale_factor,
                [config.minimum_scale_factor],
                rtol=1e-12,
            )
        case BAOABNPTLangevinParams():
            npt.assert_allclose(params.temperature, [config.temperature])
            gamma = config.friction_coefficient / FEMTO_SECOND
            tau_p = config.pressure_coupling_time * FEMTO_SECOND
            compressibility = config.compressibility / PASCAL
            npt.assert_allclose(
                params.friction_coefficient,
                [gamma],
                rtol=1e-12,
            )
            npt.assert_allclose(
                params.target_pressure,
                [config.target_pressure * PASCAL],
                rtol=1e-12,
            )
            npt.assert_allclose(
                params.pressure_coupling_time,
                [tau_p],
                rtol=1e-12,
            )
            npt.assert_allclose(
                params.compressibility,
                [compressibility],
                rtol=1e-12,
            )
            cell_vectors = systems.data.cell.vectors
            volume = jnp.abs(jnp.linalg.det(cell_vectors))[0]
            diagonal = jnp.diagonal(cell_vectors[0])
            per_column = (
                3.0
                * volume
                * (tau_p / (2.0 * jnp.pi)) ** 2
                / (compressibility * diagonal**2)
            )
            expected_mass = jnp.tril(
                jnp.broadcast_to(per_column[None, None, :], (1, 3, 3))
            )
            expected_friction = jnp.tril(jnp.full((1, 3, 3), gamma))
            npt.assert_allclose(
                params.barostat_mass,
                expected_mass,
                rtol=1e-12,
            )
            npt.assert_allclose(
                params.barostat_friction,
                expected_friction,
                rtol=1e-12,
            )
        case _:
            pytest.fail(f"Unexpected integrator parameters: {type(params).__name__}")
