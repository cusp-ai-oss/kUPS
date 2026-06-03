# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for mcmc_state_from_ase, _make_molecule, and estimate_occupied_volume."""

import tempfile

import ase
import ase.data
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from kups.application.mcmc.data import (
    AdsorbateConfig,
    HostConfig,
    MCMCParticles,
    MCMCSystems,
    MotifParticles,
    _make_molecule,
    estimate_max_adsorbates,
    estimate_occupied_volume,
    mcmc_state_from_config,
)
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.typing import (
    GroupId,
    Label,
    MotifId,
    MotifParticleId,
    ParticleId,
    SystemId,
)

L = 10.0  # cubic box side (Ang)


def _cubic_atoms(n: int = 2) -> ase.Atoms:
    """Simple cubic cell with *n* dummy atoms."""
    positions = [[1.0 * i, 1.0 * i, 1.0 * i] for i in range(n)]
    return ase.Atoms("Ar" * n, positions=positions, cell=[L, L, L], pbc=True)


def _write_cif(atoms: ase.Atoms) -> str:
    """Write atoms to a temporary CIF file and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    ase.io.write(f.name, atoms)
    return f.name


def _co2_config() -> AdsorbateConfig:
    return AdsorbateConfig(
        critical_temperature=304.2,
        critical_pressure=7.38e6,
        acentric_factor=0.224,
        positions=((0.0, 0.0, 0.0), (-1.16, 0.0, 0.0), (1.16, 0.0, 0.0)),
        symbols=("C_co2", "O_co2", "O_co2"),
    )


def _ch4_config() -> AdsorbateConfig:
    return AdsorbateConfig(
        critical_temperature=190.6,
        critical_pressure=4.6e6,
        acentric_factor=0.011,
        positions=((0.0, 0.0, 0.0),),
        symbols=("CH4_sp3",),
    )


def _host(
    cif_file: str,
    init_adsorbates: tuple[int, ...] = (0,),
    n_species: int = 1,
    temperature: float = 300.0,
) -> HostConfig:
    comp = tuple(1.0 / n_species for _ in range(n_species))
    interaction = tuple(tuple(0.0 for _ in range(n_species)) for _ in range(n_species))
    return HostConfig(
        cif_file=cif_file,
        pressure=1e5,
        temperature=temperature,
        init_adsorbates=init_adsorbates,
        adsorbate_composition=comp,
        adsorbate_interaction=interaction,
    )


def _call(host, ads_configs, *, key):
    """Wrapper matching the current mcmc_state_from_ase signature."""
    return mcmc_state_from_config(key, host, tuple(ads_configs))


class TestAdsorbateConfig:
    """Tests for AdsorbateConfig validation and as_particles."""

    def test_auto_derived_masses(self):
        cfg = _co2_config()
        assert len(cfg.masses) == 3
        assert all(m > 0 for m in cfg.masses)

    def test_auto_derived_charges(self):
        cfg = _co2_config()
        assert len(cfg.charges) == 3
        assert all(c == 0.0 for c in cfg.charges)

    def test_as_particles_shape(self):
        cfg = _co2_config()
        motif = cfg.as_particles
        assert motif.data.positions.shape == (3, 3)
        assert motif.data.masses.shape == (3,)

    def test_as_particles_labels(self):
        cfg = _co2_config()
        motif = cfg.as_particles
        assert Label("C_co2") in motif.data.labels.keys
        assert Label("O_co2") in motif.data.labels.keys


class TestHostOnly:
    """init_adsorbates=(0,) -- no adsorbates placed."""

    @classmethod
    def setup_class(cls):
        cif = _write_cif(_cubic_atoms(3))
        configs = [_co2_config()]
        cls.particles3, cls.groups3, cls.systems3, _ = _call(
            _host(cif, (0,)), configs, key=jax.random.key(0)
        )
        cif2 = _write_cif(_cubic_atoms(2))
        cls.particles2, cls.groups2, cls.systems2, _ = _call(
            _host(cif2, (0,)), configs, key=jax.random.key(0)
        )

    def test_all(self):
        # particle count matches host
        assert len(self.particles3.keys) == 3
        # groups empty for host
        assert len(self.groups2.keys) == 0
        # host group sentinel motif
        npt.assert_array_equal(
            self.particles2.data.group.indices, jnp.zeros(2, dtype=int)
        )
        # system fields
        npt.assert_allclose(self.systems2.data.temperature, jnp.array([300.0]))
        assert self.systems2.data.log_fugacity.shape == (1, 1)
        assert jnp.isfinite(self.systems2.data.log_fugacity).all()
        assert float(self.systems2.data.log_fugacity[0, 0]) < 0
        npt.assert_allclose(self.systems2.data.potential_energy, jnp.array([0.0]))


class TestWithInitialAdsorbates:
    """init_adsorbates=(2,) -- two CO2 molecules placed."""

    @classmethod
    def setup_class(cls):
        cif = _write_cif(_cubic_atoms(2))
        configs = [_co2_config()]
        cls.particles, cls.groups, cls.systems, _ = _call(
            _host(cif, (2,)), configs, key=jax.random.key(1)
        )

    def test_all(self):
        # 2 host + 2 * 3 adsorbate atoms = 8
        assert len(self.particles.keys) == 8
        # 0 host groups + 2 adsorbate groups
        assert len(self.groups.keys) == 2
        # adsorbate motif assignment
        npt.assert_array_equal(self.groups.data.motif.indices, jnp.array([0, 0]))
        # adsorbate positions within box
        assert jnp.all(jnp.isfinite(self.particles.data.positions))
        ads_pos = self.particles.data.positions[2:]
        assert jnp.all(ads_pos > -L) and jnp.all(ads_pos < 2 * L)


class TestMultipleSpecies:
    """init_adsorbates=(1, 1) -- one CO2 + one CH4."""

    @classmethod
    def setup_class(cls):
        cif = _write_cif(_cubic_atoms(2))
        configs = [_co2_config(), _ch4_config()]
        cls.particles, cls.groups, cls.systems, _ = _call(
            _host(cif, (1, 1), n_species=2), configs, key=jax.random.key(2)
        )

    def test_all(self):
        # 2 host + 3 CO2 atoms + 1 CH4 atom = 6
        assert len(self.particles.keys) == 6
        # First group: CO2 (motif 0), second group: CH4 (motif 1)
        npt.assert_array_equal(self.groups.data.motif.indices, jnp.array([0, 1]))
        # log fugacity shape and values
        assert self.systems.data.log_fugacity.shape == (1, 2)
        assert jnp.isfinite(self.systems.data.log_fugacity).all()
        assert jnp.all(self.systems.data.log_fugacity < 0)


def _build_motifs(*configs: AdsorbateConfig):
    """Build motifs with unique MotifId per species via concatenate_indexed."""
    ads = [c.as_particles for c in configs]
    dummies = [Table((MotifId(0),), jnp.zeros(1)) for _ in ads]
    result, _ = Table.union(ads, dummies)
    return result


def _motif_index(idx: int) -> Index[MotifId]:
    """Create an Index[MotifId] for a single species index."""
    return Index.integer(np.array([idx]), label=MotifId)


class TestMakeMolecule:
    """Direct tests for _make_molecule."""

    def test_positions_offset_by_com(self):
        """Offsets from COM preserve bond lengths after random rotation."""
        motifs = _build_motifs(_co2_config())
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * L))
        particles, _ = _make_molecule(motifs, _motif_index(0), cell, jax.random.key(42))

        pos = particles.data.positions
        com = pos.mean(axis=0)
        offsets = pos - com[None, :]
        ref = jnp.array(_co2_config().positions)
        npt.assert_allclose(
            jnp.linalg.norm(offsets, axis=-1),
            jnp.linalg.norm(ref, axis=-1),
            atol=1e-5,
        )

    def test_labels_match_motif(self):
        motifs = _build_motifs(_co2_config())
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * L))
        particles, _ = _make_molecule(motifs, _motif_index(0), cell, jax.random.key(0))
        labels = particles.data.labels
        assert Label("C_co2") in labels.keys
        assert Label("O_co2") in labels.keys
        assert labels.indices.shape == (3,)

    def test_group_has_correct_motif(self):
        motifs = _build_motifs(_co2_config(), _ch4_config())
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * L))
        _, group = _make_molecule(motifs, _motif_index(1), cell, jax.random.key(0))
        npt.assert_array_equal(group.data.motif.indices, jnp.array([1]))

    def test_single_group_created(self):
        motifs = _build_motifs(_co2_config())
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3) * L))
        _, group = _make_molecule(motifs, _motif_index(0), cell, jax.random.key(0))

        assert len(group.keys) == 1


class TestSystemFields:
    """Verify cell, temperature, and log_activity on the system object."""

    @classmethod
    def setup_class(cls):
        cif = _write_cif(_cubic_atoms(2))
        configs_single = [_co2_config()]
        _, _, cls.systems_default, _ = _call(
            _host(cif), configs_single, key=jax.random.key(0)
        )
        _, _, cls.systems_450, _ = _call(
            _host(cif, temperature=450.0), configs_single, key=jax.random.key(0)
        )
        configs_multi = [_co2_config(), _ch4_config()]
        _, _, cls.systems_multi, _ = _call(
            _host(cif, n_species=2), configs_multi, key=jax.random.key(0)
        )

    def test_all(self):
        # cell matches input
        diag = jnp.diag(self.systems_default.data.cell.vectors[0])
        npt.assert_allclose(diag, jnp.full(3, L), atol=1e-5)
        # temperature
        npt.assert_allclose(self.systems_450.data.temperature, jnp.array([450.0]))
        # log activity computed
        assert self.systems_multi.data.log_fugacity.shape == (1, 2)
        assert jnp.all(self.systems_multi.data.log_fugacity < 0)
        log_act = self.systems_multi.data.log_activity
        assert log_act.shape == (1, 2)
        assert jnp.all(log_act > self.systems_multi.data.log_fugacity)


def _particles(
    atomic_numbers: list[int],
    system: list[int],
    group: list[int] | None = None,
) -> Table[ParticleId, MCMCParticles]:
    """Build a minimal particle table from atomic numbers, system, and group ids."""
    n = len(atomic_numbers)
    group = group if group is not None else [0] * n
    return Table.arange(
        MCMCParticles(
            positions=jnp.zeros((n, 3)),
            masses=jnp.ones(n),
            atomic_numbers=jnp.asarray(atomic_numbers),
            charges=jnp.zeros(n),
            labels=Index.new([Label("X")] * n),
            system=Index.integer(np.asarray(system), label=SystemId),
            group=Index.integer(np.asarray(group), label=GroupId, max_count=n),
            motif=Index.zeros(n, label=MotifParticleId),
        ),
        label=ParticleId,
    )


def _sphere_volume(z: int) -> float:
    return float(4.0 / 3.0 * np.pi * ase.data.vdw_radii[z] ** 3)


class TestEstimateOccupiedVolume:
    """Tests for estimate_occupied_volume."""

    def test_per_system_sum(self):
        # System 0: H + O, system 1: C + H.
        vol = estimate_occupied_volume(
            _particles([1, 8, 6, 1], [0, 0, 1, 1]), lambda p: p.system
        )
        assert vol.keys == (SystemId(0), SystemId(1))
        npt.assert_allclose(
            vol.data,
            jnp.array(
                [
                    _sphere_volume(1) + _sphere_volume(8),
                    _sphere_volume(6) + _sphere_volume(1),
                ]
            ),
            rtol=1e-5,
        )

    def test_group_index_selects_aggregation(self):
        # Same particles, grouped by group index instead of system.
        particles = _particles([1, 8, 6], [0, 0, 0], group=[0, 1, 1])
        vol = estimate_occupied_volume(particles, lambda p: p.group)
        assert vol.keys == (GroupId(0), GroupId(1))
        npt.assert_allclose(
            vol.data,
            jnp.array([_sphere_volume(1), _sphere_volume(8) + _sphere_volume(6)]),
            rtol=1e-5,
        )

    def test_undefined_radius_contributes_zero(self):
        # Sc (Z=21) has no tabulated vdW radius -> treated as zero volume.
        vol = estimate_occupied_volume(_particles([21, 8], [0, 0]), lambda p: p.system)
        npt.assert_allclose(vol.data, jnp.array([_sphere_volume(8)]), rtol=1e-5)

    def test_jit(self):
        particles = _particles([1, 8, 6, 1], [0, 0, 1, 1])
        fn = lambda p: estimate_occupied_volume(p, lambda x: x.system)  # noqa: E731
        npt.assert_allclose(jax.jit(fn)(particles).data, fn(particles).data)


def _motifs(species: list[list[int]]) -> Table[MotifParticleId, MotifParticles]:
    """Build motif templates, one motif per inner list of atomic numbers."""
    atomic_numbers = [z for s in species for z in s]
    motif_ids = [i for i, s in enumerate(species) for _ in s]
    n = len(atomic_numbers)
    return Table.arange(
        MotifParticles(
            positions=jnp.zeros((n, 3)),
            masses=jnp.ones(n),
            atomic_numbers=jnp.asarray(atomic_numbers),
            charges=jnp.zeros(n),
            labels=Index.new([Label("X")] * n),
            motif=Index.integer(np.asarray(motif_ids), label=MotifId),
        ),
        label=MotifParticleId,
    )


def _systems(sides: list[float]) -> Table[SystemId, MCMCSystems]:
    """Build a per-system table of cubic cells with the given side lengths."""
    n = len(sides)
    frames = TriclinicFrame.from_matrix(jnp.stack([jnp.eye(3) * s for s in sides]))
    return Table.arange(
        MCMCSystems(
            cell=PeriodicCell(frames),
            temperature=jnp.full(n, 300.0),
            potential_energy=jnp.zeros(n),
            log_fugacity=jnp.zeros((n, 1)),
        ),
        label=SystemId,
    )


class TestEstimateMaxAdsorbates:
    """Tests for estimate_max_adsorbates."""

    def test_counts_use_smallest_motif(self):
        # Cubic box (vol L^3) holding two O atoms; motifs are [H] and [C, O].
        particles = _particles([8, 8], [0, 0])
        motifs = _motifs([[1], [6, 8]])
        free = L**3 - 2 * _sphere_volume(8)
        expected = np.floor(free / _sphere_volume(1))  # smallest motif is the H atom
        result = estimate_max_adsorbates(particles, motifs, _systems([L]))
        assert result.keys == (SystemId(0),)
        npt.assert_array_equal(result.data, jnp.array([expected], dtype=int))

    def test_per_system(self):
        # System 0 empty (side L); system 1 holds one O atom (side L / 2).
        particles = _particles([8], [1])
        result = estimate_max_adsorbates(
            particles, _motifs([[1], [6, 8]]), _systems([L, L / 2])
        )
        expected = jnp.array(
            [
                np.floor(L**3 / _sphere_volume(1)),
                np.floor(((L / 2) ** 3 - _sphere_volume(8)) / _sphere_volume(1)),
            ],
            dtype=int,
        )
        npt.assert_array_equal(result.data, expected)

    def test_clamped_at_zero(self):
        # Tiny box fully occupied -> no room for adsorbates.
        particles = _particles([8, 8, 8], [0, 0, 0])
        result = estimate_max_adsorbates(particles, _motifs([[6, 8]]), _systems([1.0]))
        npt.assert_array_equal(result.data, jnp.array([0]))
