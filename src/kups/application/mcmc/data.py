# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Reusable data types and configuration for rigid-body MCMC simulations."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import ase
import ase.data
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from pydantic import BaseModel, model_validator

from kups.application.utils.particles import Particles, particles_from_ase
from kups.core.constants import BOLTZMANN_CONSTANT, KELVIN, PASCAL
from kups.core.data import Index, Table
from kups.core.data.buffered import Buffered
from kups.core.lens import bind, lens
from kups.core.typing import (
    ExclusionId,
    GroupId,
    InclusionId,
    Label,
    MotifId,
    MotifParticleId,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import UnitCell, make_supercell
from kups.core.utils.jax import dataclass, field, key_chain, tree_zeros_like
from kups.core.utils.quaternion import Quaternion
from kups.mcmc.fugacity import peng_robinson_log_fugacity


class AdsorbateConfig(BaseModel):
    """Configuration for a single adsorbate species.

    Thermodynamic parameters feed Peng-Robinson EOS for fugacity calculation.
    Molecular geometry is defined by ``positions`` and ``symbols``; charges,
    masses, and atomic numbers are auto-derived from ASE if omitted.
    """

    critical_temperature: float
    """Critical temperature (K) for Peng-Robinson EOS."""
    critical_pressure: float
    """Critical pressure (Pa) for Peng-Robinson EOS."""
    acentric_factor: float
    """Acentric factor (dimensionless)."""
    positions: tuple[tuple[float, float, float], ...]
    """Atom positions within the molecule, relative to its centre of mass (Ang)."""
    symbols: tuple[str, ...]
    """Element symbols for each atom in the molecule."""
    charges: tuple[float, ...] = ()
    """Partial charges on each atom (e). Defaults to zeros."""
    masses: tuple[float, ...] = ()
    """Atomic masses (amu). Derived from symbols via ASE if omitted."""
    atomic_numbers: tuple[int, ...] = ()
    """Atomic numbers. Derived from symbols via ASE if omitted."""

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults_from_ase(cls, data: dict) -> dict:  # type: ignore[override]
        """Derive charges, masses, and atomic numbers from symbols when absent."""
        symbols = data.get("symbols", ())
        if not data.get("charges"):
            data["charges"] = tuple(0.0 for _ in symbols)
        if not data.get("masses") or not data.get("atomic_numbers"):
            zs = [ase.data.atomic_numbers.get(s.split("_")[0], 0) for s in symbols]
            if not data.get("masses"):
                unknown = [s for s, z in zip(symbols, zs, strict=True) if z == 0]
                if unknown:
                    raise ValueError(
                        "Cannot derive masses from unrecognised symbols "
                        f"{unknown!r} (ASE doesn't know these — typical for "
                        "united-atom pseudo-symbols like CH4, CH3). Pass "
                        "`masses=(...)` explicitly in AdsorbateConfig."
                    )
                data["masses"] = tuple(float(ase.data.atomic_masses[z]) for z in zs)
            if not data.get("atomic_numbers"):
                data["atomic_numbers"] = tuple(zs)
        return data

    @property
    def as_particles(self) -> Table[MotifParticleId, MotifParticles]:
        """Convert this adsorbate config into indexed particles with motif data."""
        n = len(self.symbols)
        assert self.masses is not None
        assert self.atomic_numbers is not None
        assert self.charges is not None
        return Table.arange(
            MotifParticles(
                positions=jnp.asarray(self.positions),
                masses=jnp.asarray(self.masses),
                atomic_numbers=jnp.asarray(self.atomic_numbers),
                charges=jnp.asarray(self.charges),
                labels=Index.new(list(map(Label, self.symbols))),
                motif=Index.integer(
                    jnp.zeros(n, dtype=int),
                    label=MotifId,
                    max_count=len(self.positions),
                ),
            ),
            label=MotifParticleId,
        )


class HostConfig(BaseModel):
    """Configuration for a single host framework (e.g. zeolite, MOF).

    Each host gets its own system with unit cell, thermodynamic conditions,
    and optional initial adsorbate placement.
    """

    cif_file: str
    """Path to the host structure CIF file."""
    pressure: float
    """System pressure (Pa)."""
    temperature: float
    """System temperature (K)."""
    init_adsorbates: tuple[int, ...] = (0,)
    """Number of initial adsorbates per species to randomly insert."""
    adsorbate_composition: tuple[float, ...] = (1.0,)
    """Mole fractions for each adsorbate component (must sum to 1)."""
    adsorbate_interaction: tuple[tuple[float, ...], ...] = ((0.0,),)
    """Binary interaction parameters k_ij (n_ads x n_ads)."""
    unitcell_replication: int | tuple[int, int, int] | None = None
    """(nx, ny, nz) replication of the unit cell. Auto-computed if None."""


class LJConfig(BaseModel):
    """Lennard-Jones force-field parameters."""

    parameters: dict[str, tuple[float, float]]
    """Mapping symbol -> (epsilon/kB [K], sigma [Ang]) for each element type."""
    cutoff: float = 12.0
    """Real-space cutoff (Ang)."""
    tail_correction: bool = True
    """Apply long-range tail correction."""


class EwaldCfg(BaseModel):
    """Ewald summation settings for electrostatic interactions."""

    enabled: bool = True
    cutoff: float = 12.0
    """Real-space cutoff (Ang)."""
    precision: float = 1e-6
    """Target error tolerance."""


class RunConfig(BaseModel):
    """Run configuration for MCMC simulation."""

    out_file: str | Path
    """Path to the output HDF5 file."""
    num_cycles: int
    """Number of production cycles."""
    num_warmup_cycles: int
    """Number of warmup cycles (not written to output)."""
    min_cycle_length: int
    translation_prob: float = 1 / 6
    rotation_prob: float = 1 / 6
    reinsertion_prob: float = 1 / 6
    exchange_prob: float = 1 / 2
    """Ordered move sequence for the palindrome propagator."""
    seed: int | None = None
    """Random seed. ``None`` uses a time-based seed."""


class StressResult(NamedTuple):
    """Decomposed stress tensor with potential, tail correction, and ideal gas components."""

    potential: Array
    tail_correction: Array
    ideal_gas: Array


@dataclass
class MotifParticles:
    """Template particle data for an adsorbate motif (molecule template).

    Stores the reference geometry, atomic properties, and species assignment
    for each atom in an adsorbate molecule. Used as a lookup table during
    exchange moves to populate new particle data.

    Attributes:
        positions: Reference atom positions relative to COM (Ang), shape ``(n_atoms, 3)``.
        masses: Atomic masses (amu), shape ``(n_atoms,)``.
        atomic_numbers: Atomic numbers, shape ``(n_atoms,)``.
        charges: Partial charges (e), shape ``(n_atoms,)``.
        labels: Per-atom string labels.
        motif: Index mapping atoms to their species (MotifId).
    """

    positions: Array
    masses: Array
    atomic_numbers: Array
    charges: Array
    labels: Index[Label]
    motif: Index[MotifId]


@dataclass
class MCMCParticles(Particles):
    """Particle state for rigid-body MCMC with group membership.

    Attributes:
        group: Index mapping each particle to its molecular group.
    """

    group: Index[GroupId]
    motif: Index[MotifParticleId]
    position_gradients: Array = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.position_gradients is None:
            object.__setattr__(
                self, "position_gradients", jnp.zeros_like(self.positions)
            )

    @property
    def inclusion(self) -> Index[InclusionId]:
        """Inclusion index derived from the system index."""
        return Index(
            tuple(InclusionId(lab) for lab in self.system.keys),
            self.system.indices,
            self.system.max_count,
        )

    @property
    def exclusion(self) -> Index[ExclusionId]:
        """Exclusion index derived from the group index."""
        return Index(
            tuple(ExclusionId(lab) for lab in self.group.keys),
            self.group.indices,
            self.group.max_count,
        )

    def guest_only(self) -> MCMCParticles:
        # Buffered sanitizes the the data to only include particles with valid group membership.
        return Buffered.arange(self, view=lambda x: x.group).data


@dataclass
class MCMCGroup:
    """Molecular group state for rigid-body MCMC.

    Attributes:
        system: Index mapping each group to its parent system.
        motif: Index mapping each group to its adsorbate species.
    """

    system: Index[SystemId]
    motif: Index[MotifId]

    @property
    def labels(self) -> Index[Label]:
        """Group labels derived from motif index."""
        return Index(tuple(Label(str(m)) for m in self.motif.keys), self.motif.indices)


@dataclass
class MCMCSystems:
    """Per-system thermodynamic state for MCMC simulations.

    Attributes:
        unitcell: Unit cell geometry, batched shape ``(n_systems,)``.
        temperature: Temperature (K), shape ``(n_systems,)``.
        potential_energy: Total potential energy per system (eV), shape ``(n_systems,)``.
        log_fugacity: Log fugacity per species (dimensionless), shape ``(n_systems, n_species)``.
    """

    unitcell: UnitCell
    temperature: Array
    potential_energy: Array
    log_fugacity: Array
    unitcell_gradients: UnitCell = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.unitcell_gradients is None:
            object.__setattr__(
                self, "unitcell_gradients", tree_zeros_like(self.unitcell)
            )

    @property
    def log_activity(self) -> Array:
        """Log activity: log(fugacity / kT), shape ``(n_systems, n_species)``."""
        return (
            self.log_fugacity - jnp.log(self.temperature * BOLTZMANN_CONSTANT)[:, None]
        )


def _make_molecule(
    motifs: Table[MotifParticleId, MotifParticles],
    species_idx: Index[MotifId],
    unitcell: UnitCell,
    key: Array,
) -> tuple[Table[ParticleId, MCMCParticles], Table[GroupId, MCMCGroup]]:
    """Place one adsorbate molecule at a random position within the unit cell.

    Samples a uniform centre-of-mass position in fractional coordinates,
    then offsets the motif template positions by the Cartesian COM.

    Args:
        motifs: Concatenated motif particle templates for all species.
        species_idx: Index selecting which adsorbate species to place.
        unitcell: Unit cell defining the simulation box.
        key: JAX PRNG key for random placement.

    Returns:
        Tuple of ``(particles, group)`` for the new molecule.
    """
    chain = key_chain(key)
    motif_index = Index(motifs.keys, motifs.data.motif.where_flat(species_idx))
    com = (
        jax.random.uniform(next(chain), (3,), minval=-0.5, maxval=0.5)
        @ unitcell.lattice_vectors
    )
    rot = Quaternion.random(next(chain))
    tpl = motifs[motif_index]
    n_a = len(tpl.positions)
    particles = Table.arange(
        MCMCParticles(
            positions=tpl.positions @ rot + com[None, :],
            masses=tpl.masses,
            atomic_numbers=tpl.atomic_numbers,
            charges=tpl.charges,
            labels=tpl.labels,
            system=Index.zeros(n_a, label=SystemId),
            group=Index.zeros(n_a, label=GroupId, max_count=n_a),
            motif=motif_index,
        ),
        label=ParticleId,
    )
    group = Table.arange(
        MCMCGroup(Index.zeros(1, label=SystemId), species_idx), label=GroupId
    )
    return particles, group


@dataclass
class PreparedHost:
    """Host + motif data prepared once per host, reused for every adsorbate count.

    Holds everything that does not depend on the number of adsorbates placed
    in a particular macrostate: the parsed host framework, its supercell, the
    combined adsorbate motif templates, and the reservoir log-fugacity. Callers
    that need to build multiple single-system states for the same host (e.g.
    the flat-histogram ``mcmc_nvtw`` entry point that fans out over
    macrostates $N = 0, \\ldots, N_{\\max}$) pay the CIF parse and
    Peng--Robinson evaluation exactly once.

    Attributes:
        motifs: Concatenated motif templates for all adsorbate species.
        unitcell: Simulation cell after optional supercell replication.
        host_particles: Per-atom host data with sentinel system/group indices
            (``system=Index.zeros(n_host)``). Must be re-emitted when the
            number of systems is known.
        host_groups: Empty group table (host atoms have no groups).
        log_fugacity: Per-species log-fugacity at the host's $(T, P)$,
            shape ``(n_adsorbates,)``.
        temperature: Host temperature [K].
        n_adsorbate_species: Number of adsorbate species (for sizing indices).
    """

    motifs: Table[MotifParticleId, MotifParticles]
    unitcell: UnitCell
    host_particles: Table[ParticleId, MCMCParticles]
    host_groups: Table[GroupId, MCMCGroup]
    log_fugacity: Array
    temperature: float = field(static=True)
    n_adsorbate_species: int = field(static=True)


def prepare_host(
    host: HostConfig,
    adsorbates: tuple[AdsorbateConfig, ...],
) -> PreparedHost:
    """Perform the expensive per-host work once: CIF parse, supercell, motifs, fugacity.

    Args:
        host: Host configuration (CIF path, $T$, $P$, supercell, ...).
        adsorbates: Adsorbate species configurations.

    Returns:
        A :class:`PreparedHost` bundle ready to be fed into
        :func:`place_adsorbates` any number of times.
    """
    hp, unitcell, _ = particles_from_ase(host.cif_file)
    p = hp.data
    n_host = len(hp.keys)
    if host.unitcell_replication is not None:
        pos_lens = lens(lambda x: x.positions, cls=Particles)
        p = pos_lens.apply(p, unitcell.wrap)
        unitcell, p = make_supercell(unitcell, host.unitcell_replication, p, pos_lens)
        n_host = len(p.positions)
    n_ads = len(adsorbates)

    # Build motifs: concatenate with a MotifId group to assign unique species indices
    ads_particles = [c.as_particles for c in adsorbates]
    motif_ids = [Table((MotifId(0),), jnp.zeros(1)) for _ in ads_particles]
    motifs, _ = Table.union(ads_particles, motif_ids)

    # Compute log activity via Peng-Robinson EOS
    crit_temps = jnp.asarray([a.critical_temperature for a in adsorbates])
    crit_press = jnp.asarray([a.critical_pressure for a in adsorbates])
    acentric = jnp.asarray([a.acentric_factor for a in adsorbates])
    result = peng_robinson_log_fugacity(
        jnp.asarray(float(host.pressure)) * PASCAL,
        jnp.asarray(float(host.temperature)) * KELVIN,
        crit_press * PASCAL,
        crit_temps * KELVIN,
        acentric,
        jnp.asarray(host.adsorbate_composition),
        jnp.asarray(host.adsorbate_interaction),
    )

    host_particles = Table.arange(
        MCMCParticles(
            positions=p.positions,
            masses=p.masses,
            atomic_numbers=p.atomic_numbers,
            charges=p.charges,
            labels=p.labels,
            system=Index.zeros(n_host, label=SystemId),
            group=Index.integer(jnp.zeros(n_host, dtype=int), n=0, label=GroupId),
            motif=Index.integer(
                jnp.zeros(n_host, dtype=int), n=0, label=MotifParticleId
            ),
        ),
        label=ParticleId,
    )
    host_groups = Table.arange(
        MCMCGroup(
            system=Index.zeros(0, label=SystemId),
            motif=Index.integer(jnp.full(0, n_ads, dtype=int), n=n_ads, label=MotifId),
        ),
        label=GroupId,
    )

    return PreparedHost(
        motifs=motifs,
        unitcell=unitcell,
        host_particles=host_particles,
        host_groups=host_groups,
        log_fugacity=result.log_fugacity,
        temperature=host.temperature,
        n_adsorbate_species=n_ads,
    )


def place_adsorbates(
    key: Array,
    prepared: PreparedHost,
    init_adsorbates: tuple[int, ...],
) -> tuple[
    Table[ParticleId, MCMCParticles],
    Table[GroupId, MCMCGroup],
    Table[SystemId, MCMCSystems],
]:
    """Place ``init_adsorbates`` per species in a prepared host, yielding a single-system state.

    Args:
        key: JAX PRNG key for random adsorbate placement.
        prepared: Output of :func:`prepare_host`.
        init_adsorbates: Number of adsorbates to place per species.

    Returns:
        ``(particles, groups, system)`` for a single-system build. The caller
        is responsible for unioning multiple such triples into a batched state.
    """
    system = Table.arange(
        MCMCSystems(
            unitcell=prepared.unitcell[None],
            temperature=jnp.array([prepared.temperature]),
            potential_energy=jnp.zeros(1),
            log_fugacity=prepared.log_fugacity[None],
        ),
        label=SystemId,
    )

    ads_parts: list[Table[ParticleId, MCMCParticles]] = []
    ads_groups: list[Table[GroupId, MCMCGroup]] = []
    for motif_idx, n_init in enumerate(init_adsorbates):
        idx = Index.integer(np.array([motif_idx]), label=MotifId)
        for _ in range(n_init):
            key, subkey = jax.random.split(key)
            mol_p, mol_g = _make_molecule(
                prepared.motifs, idx, prepared.unitcell, subkey
            )
            ads_parts.append(mol_p)
            ads_groups.append(mol_g)

    particles, groups = Table.union(
        [prepared.host_particles, *ads_parts],
        [prepared.host_groups, *ads_groups],
    )
    particles = bind(
        particles, lambda x: (x.data.motif.keys, x.data.group.max_count)
    ).set((prepared.motifs.keys, prepared.motifs.data.motif.max_count))
    return particles, groups, system


def mcmc_state_from_config(
    key: Array,
    host: HostConfig,
    adsorbates: tuple[AdsorbateConfig, ...],
) -> tuple[
    Table[ParticleId, MCMCParticles],
    Table[GroupId, MCMCGroup],
    Table[SystemId, MCMCSystems],
    Table[MotifParticleId, MotifParticles],
]:
    """Build per-host MCMC state from ASE Atoms and adsorbate configs.

    Thin composition of :func:`prepare_host` and :func:`place_adsorbates`.
    Kept as the stable, single-call entry point for rigid-body GCMC
    (:mod:`kups.application.simulations.mcmc_rigid`).

    Args:
        key: JAX PRNG key for random adsorbate placement.
        host: Host configuration.
        adsorbates: Adsorbate species configurations.

    Returns:
        ``(particles, groups, system, motifs)``.
    """
    prepared = prepare_host(host, adsorbates)
    particles, groups, system = place_adsorbates(key, prepared, host.init_adsorbates)
    return particles, groups, system, prepared.motifs
