# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for mcmc_state_from_ase and _make_molecule."""

import tempfile

import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from kups.application.mcmc.data import (
    AdsorbateConfig,
    HostConfig,
    _make_molecule,
    mcmc_state_from_config,
)
from kups.core.data import Index, Table
from kups.core.typing import Label, MotifId
from kups.core.unitcell import TriclinicUnitCell

from ..clear_cache import clear_cache  # noqa: F401

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
        masses=(16.043,),
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
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3) * L)
        particles, _ = _make_molecule(motifs, _motif_index(0), uc, jax.random.key(42))

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
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3) * L)
        particles, _ = _make_molecule(motifs, _motif_index(0), uc, jax.random.key(0))
        labels = particles.data.labels
        assert Label("C_co2") in labels.keys
        assert Label("O_co2") in labels.keys
        assert labels.indices.shape == (3,)

    def test_group_has_correct_motif(self):
        motifs = _build_motifs(_co2_config(), _ch4_config())
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3) * L)
        _, group = _make_molecule(motifs, _motif_index(1), uc, jax.random.key(0))
        npt.assert_array_equal(group.data.motif.indices, jnp.array([1]))

    def test_single_group_created(self):
        motifs = _build_motifs(_co2_config())
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3) * L)
        _, group = _make_molecule(motifs, _motif_index(0), uc, jax.random.key(0))

        assert len(group.keys) == 1


class TestSystemFields:
    """Verify unitcell, temperature, and log_activity on the system object."""

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
        # unitcell matches input
        diag = jnp.diag(self.systems_default.data.unitcell.lattice_vectors[0])
        npt.assert_allclose(diag, jnp.full(3, L), atol=1e-5)
        # temperature
        npt.assert_allclose(self.systems_450.data.temperature, jnp.array([450.0]))
        # log activity computed
        assert self.systems_multi.data.log_fugacity.shape == (1, 2)
        assert jnp.all(self.systems_multi.data.log_fugacity < 0)
        log_act = self.systems_multi.data.log_activity
        assert log_act.shape == (1, 2)
        assert jnp.all(log_act > self.systems_multi.data.log_fugacity)
