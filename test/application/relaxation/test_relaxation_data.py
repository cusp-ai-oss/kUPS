# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.relaxation.data source-neutral builder."""

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

import kups.application.relaxation.data as relax_data
from kups.application.relaxation.data import (
    RelaxParticles,
    RelaxSystems,
    relax_state_from_ase,
    relax_state_from_particles_and_cell,
)
from kups.application.utils.particles import Particles, particles_from_arrays
from kups.core.cell import (
    AnyPeriodicity,
    Cell,
    DeformedFrame,
    LogTriclinicFrame,
    MatrixLogFrame,
    OrthogonalFrame,
    TriclinicFrame,
)
from kups.core.data import Index, Table
from kups.core.typing import ExclusionId, Label, ParticleId, SystemId


def _cell() -> Cell[AnyPeriodicity]:
    """Return an unbatched skewed triclinic test cell."""
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


def _deformation_factor(systems: Table[SystemId, RelaxSystems]) -> jax.Array:
    """Return the per-system deformation cell_factor as a flat array."""
    frame = systems.data.cell.frame
    assert isinstance(frame, DeformedFrame)
    deformation = frame.deformation
    assert isinstance(deformation, MatrixLogFrame)
    return deformation.cell_factor.reshape(-1)


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


def test_public_builder_signature_without_module_all() -> None:
    """Public signature is (particles, cell); the module adds no __all__."""
    signature = inspect.signature(relax_state_from_particles_and_cell)

    assert tuple(signature.parameters) == ("particles", "cell")
    assert not hasattr(relax_data, "__all__")


def test_builder_preserves_identity_and_geometry() -> None:
    """Particle keys, the referenced SystemId, and geometry are preserved."""
    system_key = SystemId(12)
    particle_keys = (ParticleId(4), ParticleId(9))
    particles = _particles(
        Index((system_key,), jnp.zeros(2, dtype=int)),
        keys=particle_keys,
    )
    cell = _cell()

    relax_particles, systems = relax_state_from_particles_and_cell(particles, cell)

    assert relax_particles.keys == particle_keys
    assert relax_particles.data.system.keys == (system_key,)
    npt.assert_array_equal(
        relax_particles.data.system.indices,
        jnp.zeros(2, dtype=int),
    )
    assert systems.keys == (system_key,)
    npt.assert_array_equal(relax_particles.data.positions, particles.data.positions)
    assert systems.data.cell.vectors.shape == (1, 3, 3)
    npt.assert_allclose(systems.data.cell.vectors[0], cell.vectors, atol=1e-12)


def test_empty_particles_are_rejected() -> None:
    """Empty particle tables raise ValueError."""
    particles = _particles(
        Index((SystemId(12),), jnp.empty((0,), dtype=int)),
        keys=(),
    )

    with pytest.raises(ValueError, match="particles.*at least one"):
        relax_state_from_particles_and_cell(particles, _cell())


def test_multiple_system_keys_are_rejected() -> None:
    """Multiple SystemId keys raise ValueError."""
    particles = _particles(
        Index((SystemId(12), SystemId(13)), jnp.array([0, 1])),
    )

    with pytest.raises(ValueError, match="particles.*exactly one system"):
        relax_state_from_particles_and_cell(particles, _cell())


def test_invalid_reference_into_single_system_key_is_rejected() -> None:
    """Indices that do not select the sole SystemId raise ValueError."""
    particles = _particles(
        Index((SystemId(12),), jnp.array([0, 1])),
    )

    with pytest.raises(ValueError, match="particles.*sole SystemId"):
        relax_state_from_particles_and_cell(particles, _cell())


def test_column_vector_system_indices_are_rejected() -> None:
    """Column-vector system.indices raise ValueError."""
    particles = _particles(
        Index((SystemId(12),), jnp.zeros((2, 1), dtype=int)),
    )

    with pytest.raises(
        ValueError, match=r"particles.*one system reference per particle"
    ):
        relax_state_from_particles_and_cell(particles, _cell())


def test_batched_cell_is_rejected() -> None:
    """A cell whose vectors are not shape (3, 3) raises ValueError."""
    particles = _particles(
        Index((SystemId(12),), jnp.zeros(2, dtype=int)),
    )

    with pytest.raises(ValueError, match=r"cell\.vectors.*\(3, 3\)"):
        relax_state_from_particles_and_cell(particles, _cell()[None])


def test_deformed_frame_is_rejected() -> None:
    """An unbatched cell with an outer DeformedFrame raises ValueError.

    Distinct from the already-batched cell rejected by the shape check.
    """
    particles = _particles(
        Index((SystemId(12),), jnp.zeros(2, dtype=int)),
    )
    base = _cell()
    deformed = Cell.from_pbc(
        DeformedFrame.from_frame(base.frame),
        (True, True, True),
    )
    assert deformed.vectors.shape == (3, 3)

    with pytest.raises(ValueError, match=r"DeformedFrame"):
        relax_state_from_particles_and_cell(particles, deformed)


def test_ase_wrapper_delegates_to_source_neutral_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """relax_state_from_ase forwards adapter output to the builder."""
    particles = _particles(
        Index((SystemId(0),), jnp.zeros(2, dtype=int)),
    )
    cell = _cell()
    expected = (object(), object())
    calls: list[tuple[object, ...]] = []

    def fake_particles_from_ase(
        atoms: ase.Atoms | str | Path,
    ) -> tuple[Table[ParticleId, Particles], Cell[AnyPeriodicity], object]:
        calls.append(("adapter", atoms))
        return particles, cell, object()

    def fake_builder(
        actual_particles: Table[ParticleId, Particles],
        actual_cell: Cell[AnyPeriodicity],
    ) -> tuple[object, object]:
        calls.append(("builder", actual_particles, actual_cell))
        return expected

    monkeypatch.setattr(relax_data, "particles_from_ase", fake_particles_from_ase)
    monkeypatch.setattr(relax_data, "relax_state_from_particles_and_cell", fake_builder)

    actual = relax_state_from_ase("input.cif")

    assert actual is expected
    assert calls == [
        ("adapter", "input.cif"),
        ("builder", particles, cell),
    ]


@pytest.mark.parametrize("use_path", [False, True], ids=["atoms", "file-path"])
def test_ase_wrapper_retains_input_forms(tmp_path: Path, use_path: bool) -> None:
    """ASE Atoms and file-path inputs remain supported and canonical."""
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

    particles, systems = relax_state_from_ase(source)

    assert particles.keys == (ParticleId(0), ParticleId(1))
    assert systems.keys == (SystemId(0),)
    npt.assert_array_equal(_deformation_factor(systems), [2.0])


def test_array_builder_matches_ase_for_skewed_cell() -> None:
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

    array_particles, array_cell, _ = particles_from_arrays(
        positions=positions,
        cell_vectors=cell_vectors,
        periodicity=(True, False, True),
        masses=masses,
        atomic_numbers=atomic_numbers,
        charges=charges,
        labels=labels,
    )
    actual_particles, actual_systems = relax_state_from_particles_and_cell(
        array_particles,
        array_cell,
    )
    expected_particles, expected_systems = relax_state_from_ase(atoms)

    assert not jnp.allclose(cell_vectors, jnp.tril(cell_vectors))
    assert actual_particles.keys == expected_particles.keys
    assert actual_systems.keys == expected_systems.keys
    _assert_tree_allclose(actual_particles.data, expected_particles.data)
    _assert_tree_allclose(actual_systems.data, expected_systems.data)


def test_canonical_array_outputs_support_table_union() -> None:
    """Union of canonical outputs preserves each differently-sized system.

    Two systems of unequal size (2 and 3 particles) with distinct diagonal
    cells are merged; particle-to-system associations, per-system cell vectors,
    and per-system deformation factors are checked against independently
    specified expectations.
    """
    system_lengths = (
        jnp.array([4.0, 5.0, 6.0]),
        jnp.array([7.0, 8.0, 9.0]),
    )
    specs = [
        {
            "positions": jnp.array([[0.2, 0.4, 0.6], [1.1, 1.3, 1.5]]),
            "cell_vectors": jnp.diag(system_lengths[0]),
            "masses": jnp.array([22.99, 35.45]),
            "atomic_numbers": jnp.array([11, 17]),
            "charges": jnp.array([0.75, -0.75]),
            "labels": ["Na", "Cl"],
        },
        {
            "positions": jnp.array([[0.3, 0.5, 0.7], [1.2, 1.4, 1.6], [2.1, 2.3, 2.5]]),
            "cell_vectors": jnp.diag(system_lengths[1]),
            "masses": jnp.array([15.999, 1.008, 1.008]),
            "atomic_numbers": jnp.array([8, 1, 1]),
            "charges": jnp.array([-0.8, 0.4, 0.4]),
            "labels": ["O", "H", "H"],
        },
    ]
    outputs = []
    for spec in specs:
        particles, cell, _ = particles_from_arrays(
            periodicity=(True, True, True), **spec
        )
        outputs.append(relax_state_from_particles_and_cell(particles, cell))

    merged_particles, merged_systems = Table.union(
        [outputs[0][0], outputs[1][0]],
        [outputs[0][1], outputs[1][1]],
    )

    # Particle-to-system associations: globally-unique keys, two systems, and
    # a per-particle reference of [system0 x2, system1 x3].
    assert merged_particles.keys == tuple(ParticleId(i) for i in range(5))
    assert merged_systems.keys == (SystemId(0), SystemId(1))
    assert merged_particles.data.system.keys == (SystemId(0), SystemId(1))
    npt.assert_array_equal(merged_particles.data.system.indices, [0, 0, 1, 1, 1])
    # The non-key exclusion leaf merges by dedup: keys (0, 1, 2), each system
    # keeping its own per-atom groups.
    assert merged_particles.data.exclusion.keys == (
        ExclusionId(0),
        ExclusionId(1),
        ExclusionId(2),
    )
    npt.assert_array_equal(merged_particles.data.exclusion.indices, [0, 1, 0, 1, 2])

    # Each system keeps its own cell vectors (diagonal cells stay diagonal) and
    # its own deformation factor (its particle count).
    assert merged_systems.data.cell.vectors.shape == (2, 3, 3)
    npt.assert_allclose(
        merged_systems.data.cell.vectors[0], jnp.diag(system_lengths[0]), atol=1e-12
    )
    npt.assert_allclose(
        merged_systems.data.cell.vectors[1], jnp.diag(system_lengths[1]), atol=1e-12
    )
    npt.assert_array_equal(_deformation_factor(merged_systems), [2.0, 3.0])


def test_builder_initial_zero_state_and_pytree_contract() -> None:
    """Zero gradients, zero energy, default exclusions, matching cell trees."""
    particles = _particles(
        Index((SystemId(4),), jnp.zeros(3, dtype=int)),
    )
    cell = _cell()

    relax_particles, systems = relax_state_from_particles_and_cell(particles, cell)

    for field_name in ("positions", "masses", "atomic_numbers", "charges"):
        actual = getattr(relax_particles.data, field_name)
        expected = getattr(particles.data, field_name)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        npt.assert_array_equal(actual, expected)

    assert relax_particles.data.position_gradients.shape == (3, 3)
    assert (
        relax_particles.data.position_gradients.dtype == particles.data.positions.dtype
    )
    npt.assert_array_equal(
        relax_particles.data.position_gradients,
        jnp.zeros_like(particles.data.positions),
    )
    npt.assert_array_equal(relax_particles.data.forces, jnp.zeros((3, 3)))
    assert relax_particles.data.exclusion.keys == tuple(
        ExclusionId(i) for i in range(3)
    )
    npt.assert_array_equal(relax_particles.data.exclusion.indices, jnp.arange(3))

    assert systems.data.cell.vectors.shape == (1, 3, 3)
    assert systems.data.potential_energy.shape == (1,)
    npt.assert_array_equal(systems.data.potential_energy, jnp.zeros(1))

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

    system_structure = jax.tree_util.tree_structure(systems.data)
    assert (
        type(
            jax.tree_util.tree_unflatten(
                system_structure, jax.tree_util.tree_leaves(systems.data)
            )
        )
        is RelaxSystems
    )
    particle_structure = jax.tree_util.tree_structure(relax_particles.data)
    assert (
        type(
            jax.tree_util.tree_unflatten(
                particle_structure, jax.tree_util.tree_leaves(relax_particles.data)
            )
        )
        is RelaxParticles
    )


@pytest.mark.parametrize("n_particles", [1, 5])
def test_cell_factor_equals_particle_count_and_ase_parity(n_particles: int) -> None:
    """Deformation cell_factor is the per-system particle count for both paths."""
    particles = _particles(
        Index((SystemId(0),), jnp.zeros(n_particles, dtype=int)),
    )
    cell = _cell()

    _, systems = relax_state_from_particles_and_cell(particles, cell)
    factor = _deformation_factor(systems)

    assert factor.shape == (1,)
    npt.assert_array_equal(factor, [float(n_particles)])
    assert factor.dtype == particles.data.positions.dtype

    atoms = ase.Atoms(
        f"Ar{n_particles}",
        positions=np.asarray(particles.data.positions),
        cell=np.asarray(cell.vectors),
        pbc=True,
    )
    _, ase_systems = relax_state_from_ase(atoms)
    npt.assert_array_equal(_deformation_factor(ase_systems), factor)


@pytest.mark.parametrize(
    "frame",
    [
        TriclinicFrame.from_matrix(
            jnp.array([[4.0, 0.0, 0.0], [0.5, 5.0, 0.0], [0.25, 0.75, 6.0]])
        ),
        OrthogonalFrame(lengths=jnp.array([4.0, 5.0, 6.0])),
    ],
    ids=["triclinic", "orthogonal"],
)
def test_undeformed_frames_are_accepted_without_transform(frame: object) -> None:
    """Representative undeformed frames build with unchanged physical vectors."""
    particles = _particles(
        Index((SystemId(0),), jnp.zeros(2, dtype=int)),
    )
    cell = Cell.from_pbc(frame, (True, True, True))

    _, systems = relax_state_from_particles_and_cell(particles, cell)

    assert isinstance(systems.data.cell.frame, DeformedFrame)
    npt.assert_allclose(systems.data.cell.vectors[0], cell.vectors, atol=1e-12)


@pytest.mark.parametrize(
    "periodicity",
    [(True, True, True), (False, False, False), (True, True, False)],
    ids=["periodic", "vacuum", "slab-xy"],
)
def test_periodicity_and_cell_variant_are_preserved(
    periodicity: AnyPeriodicity,
) -> None:
    """The prepared cell keeps the input periodicity and concrete Cell variant.

    Checked directly against the input cell, independent of the ASE wrapper, so
    fully periodic, non-periodic, and partially periodic inputs are all covered.
    """
    frame = TriclinicFrame.from_matrix(
        jnp.array([[4.0, 0.0, 0.0], [0.5, 5.0, 0.0], [0.25, 0.75, 6.0]])
    )
    cell = Cell.from_pbc(frame, periodicity)
    particles = _particles(Index((SystemId(0),), jnp.zeros(2, dtype=int)))

    _, systems = relax_state_from_particles_and_cell(particles, cell)

    prepared = systems.data.cell
    assert prepared.periodic == periodicity
    assert prepared.periodic == cell.periodic
    assert type(prepared) is type(cell)


@pytest.mark.parametrize(
    "make_frame",
    [
        lambda vecs: LogTriclinicFrame.from_matrix(vecs, cell_factor=7.0),
        lambda vecs: MatrixLogFrame.from_lower_triangular(vecs, cell_factor=7.0),
    ],
    ids=["log-triclinic", "matrix-log"],
)
def test_source_log_frame_factor_preserved_in_reference_base(
    make_frame: object,
) -> None:
    """A raw log-frame source keeps its cell_factor in the deformation base."""
    vecs = jnp.array([[4.0, 0.0, 0.0], [0.5, 5.0, 0.0], [0.25, 0.75, 6.0]])
    cell = Cell.from_pbc(make_frame(vecs), (True, True, True))  # type: ignore[operator]

    particles = _particles(
        Index((SystemId(0),), jnp.zeros(3, dtype=int)),
    )

    _, systems = relax_state_from_particles_and_cell(particles, cell)

    frame = systems.data.cell.frame
    assert isinstance(frame, DeformedFrame)
    npt.assert_allclose(systems.data.cell.vectors[0], cell.vectors, atol=1e-12)
    npt.assert_array_equal(np.unique(np.asarray(frame.base.cell_factor)), [7.0])
    npt.assert_array_equal(_deformation_factor(systems), [3.0])
