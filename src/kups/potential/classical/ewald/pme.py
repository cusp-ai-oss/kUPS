# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Smooth particle-mesh Ewald (PME): the reciprocal energy on a charge mesh.

Charges are spread onto a mesh with cardinal B-splines, and the energy is a sum
over the mesh's Fourier modes (Essmann et al., J. Chem. Phys. 103, 8577, 1995).
Forces and cell gradients come from autodiff. PME evaluates whole systems and has
no incremental update; Monte Carlo keeps the cached structure factors of direct
Ewald. Positions, cells and screening enter as quantities; the mesh, its FFT and
the spline weights run on fractional coordinates and magnitudes in e.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core import units
from kups.core.cell import Periodic3D
from kups.core.data import Table
from kups.core.lens import Lens, View, lens
from kups.core.neighborlist import EmptyNeighborList, NeighborList
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import EMPTY_LENS, Potential, PotentialOut
from kups.core.typing import HasCell, ParticleId, SystemId
from kups.core.utils.functools import pipe
from kups.core.utils.segment import segment_sum
from kups.potential.common.energy import (
    EnergyFunction,
    FullSumComposer,
    PotentialFromEnergy,
)
from kups.potential.common.graph import (
    GraphConstructor,
    GraphInputConstructor,
    GraphPotentialInput,
    PointCloud,
)

from .parameters import IsEwaldPointData, PMEParameters
from .potential import (
    EwaldPotential,
    _ewald_particle_table,
    _influence,
    make_ewald_exclusion_correction_potential,
    make_ewald_net_charge_potential,
    make_ewald_self_interaction_potential,
    make_ewald_short_range_potential,
)

type PMEInput = GraphPotentialInput[
    PMEParameters, IsEwaldPointData, HasCell[Periodic3D], Literal[0]
]


def _bspline_weights(frac: Array, order: int) -> Array:
    """Cardinal B-spline weights ``M_order(frac + j)`` of mesh points ``floor(u) - j``."""
    x = frac[..., None] + jnp.arange(order, dtype=frac.dtype)
    zeros = jnp.zeros((*frac.shape, order - 2), frac.dtype)
    w = jnp.concatenate([frac[..., None], 1 - frac[..., None], zeros], axis=-1)  # M_2
    for n in range(3, order + 1):
        # Cox-de Boor: M_n(x) = (x M_{n-1}(x) + (n - x) M_{n-1}(x - 1)) / (n - 1)
        shifted = jnp.concatenate([jnp.zeros_like(w[..., :1]), w[..., :-1]], axis=-1)
        w = (x * w + (n - x) * shifted) / (n - 1)
    return w


def _euler_factors(size: int, order: int) -> Array:
    """``|b(m)|²`` of the Euler exponential splines along a mesh axis of ``size`` points."""
    with jax.ensure_compile_time_eval():
        spline = _bspline_weights(jnp.zeros(()), order)[1:]  # M(1), ..., M(order - 1)
        m, k = jnp.arange(size), jnp.arange(order - 1)
        phase = jnp.exp(2j * jnp.pi * m[:, None] * k[None, :] / size)
        return 1 / jnp.abs(phase @ spline) ** 2


def _rfft_weights(size: int) -> np.ndarray:
    """How often each half-spectrum mode of a real FFT along ``size`` points counts."""
    weights = np.full(size // 2 + 1, 2.0)
    weights[0] = 1.0
    if size % 2 == 0:
        weights[-1] = 1.0
    return weights


def _mesh_power(
    fractional: Array,
    charges: Array,
    system: Array,
    n_systems: int,
    /,
    params: PMEParameters,
) -> Array:
    """Squared mesh structure factors ``|S(m)|²`` over the real FFT's half spectrum.

    Args:
        fractional: Coordinates folded into ``[0, 1)``, shaped ``(n_particles, 3)``.
        charges: Charge magnitudes, shaped ``(n_particles,)``; zero in inactive rows.
        system: System position of each particle, shaped ``(n_particles,)``.
        n_systems: Number of systems; particles of later systems are dropped.
        params: Mesh, spline order and mesh precision.

    Returns:
        Powers in ``reciprocal_dtype``, shaped ``(n_systems, mx, my, mz // 2 + 1)``.
    """
    n_points = int(np.prod(params.mesh))
    if (n_systems + 1) * n_points > jnp.iinfo(system.dtype).max:
        raise ValueError(f"{n_systems} meshes {params.mesh} exceed {system.dtype}.")
    scaled = fractional * jnp.asarray(params.mesh, fractional.dtype)
    base = jnp.floor(scaled)
    w = _bspline_weights(scaled - base, params.order)  # (n_particles, 3, order)
    # Weight j belongs to mesh point floor(u) - j; ids run over all systems' meshes.
    sizes = jnp.asarray(params.mesh, system.dtype)[:, None]
    j = jnp.arange(params.order, dtype=system.dtype)
    x, y, z = jnp.moveaxis((base.astype(system.dtype)[..., None] - j) % sizes, 1, 0)
    ids = (
        (system[:, None, None, None] * sizes[0] + x[:, :, None, None]) * sizes[1]
        + y[:, None, :, None]
    ) * sizes[2] + z[:, None, None, :]
    spread = (
        charges[:, None, None, None]
        * w[:, 0, :, None, None]
        * w[:, 1, None, :, None]
        * w[:, 2, None, None, :]
    )
    grid = segment_sum(
        spread.astype(params.reciprocal_dtype), ids, n_systems * n_points
    )
    modes = jnp.fft.rfftn(grid.reshape(n_systems, *params.mesh), axes=(1, 2, 3))
    return modes.real**2 + modes.imag**2


def _mode_weights(mesh: tuple[int, int, int], order: int) -> Array:
    """Euler factors of every half-spectrum mode times its multiplicity."""
    mx, my, mz = mesh
    half = mz // 2 + 1
    return (
        _euler_factors(mx, order)[:, None, None]
        * _euler_factors(my, order)[None, :, None]
        * (_euler_factors(mz, order)[:half] * _rfft_weights(mz))
    )


def pme_long_range_energy(
    inp: PMEInput,
) -> WithPatch[Table[SystemId, units.Energy], IdPatch[Any]]:
    """Reciprocal energy ``k_e sum_m P(k_m) |b(m)|² |S(m)|²`` from the charge mesh.

    Args:
        inp: Particles, periodic cells and PME parameters.

    Returns:
        Per-system energies without the neutralizing-background correction, in
        the dtype of the positions, and an identity patch.
    """
    params, systems = inp.parameters, inp.graph.systems
    particles = inp.graph.particles.data
    dtype = particles.positions.dtype
    system = particles.system.indices_in(systems.keys)
    # Inactive rows can contain NaNs. Mask before spreading and differentiation.
    valid = system < systems.size
    positions = units.where(
        valid[:, None], particles.positions, units.zeros_like(particles.positions)
    )
    cells = systems.map_data(lambda s: s.cell.materialize())
    cell = cells[particles.system]
    fractional = cell.fold(cell.frame.to_fractional(positions))[0]
    power = _mesh_power(
        jnp.where(valid[:, None], fractional, 0.0),
        jnp.where(valid, particles.charges.value, 0.0),
        system,
        systems.size,
        params,
    )
    # k = 2π h⁻¹ m for integer modes m, so k² = mᵀ G m.
    inverse = cells.data.inverse_vectors
    metric = (2 * jnp.pi) ** 2 * units.einsum("sia,sib->sab", inverse, inverse)
    mx, my, mz = params.mesh
    m = jnp.stack(
        jnp.meshgrid(
            jnp.fft.fftfreq(mx, 1 / mx, dtype=dtype),
            jnp.fft.fftfreq(my, 1 / my, dtype=dtype),
            jnp.arange(mz // 2 + 1, dtype=dtype),
            indexing="ij",
        ),
        axis=-1,
    )
    k_squared = units.einsum("sab,xyza,xyzb->sxyz", metric, m, m)
    sys_idx = systems.index
    influence = _influence(
        k_squared,
        params.alpha[sys_idx][:, None, None, None],
        cells.data.volume[:, None, None, None],
    )
    weights = (_mode_weights(params.mesh, params.order) * power).astype(dtype)
    squared_charges = units.from_dim(
        weights, units.dim_of(particles.charges**2), units.base_of(particles.charges)
    )
    reciprocal = units.sum(influence * squared_charges, axis=(1, 2, 3))
    energy = (units.constants.k_e * reciprocal).astype(dtype)
    return WithPatch(Table.arange(energy, label=SystemId), IdPatch[Any]())


def make_pme_long_range_potential[State, Ptch: Patch[Any], Gradients, Hessians](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasCell[Periodic3D]]],
    parameter_view: View[State, PMEParameters],
    gradient_lens: Lens[
        PointCloud[IsEwaldPointData, HasCell[Periodic3D]], Gradients
    ] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[State, Hessians] = EMPTY_LENS,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create the PME reciprocal-space (long-range) potential.

    Args:
        particles_view: View of the particle table.
        systems_view: View of the system table and periodic cells.
        parameter_view: View of PME parameters.
        gradient_lens: Lens selecting point-cloud variables to differentiate.
        hessian_lens: Lens selecting gradient entries to differentiate again.
        hessian_idx_view: View supplying Hessian row and column indices.
        patch_idx_view: Optional view supplying indices for output-cache updates.

    Returns:
        Reciprocal potential that evaluates whole systems, without probe or cache.
    """
    return PotentialFromEnergy(
        energy_fn=pme_long_range_energy,
        composer=FullSumComposer(
            GraphInputConstructor(
                graph_constructor=GraphConstructor(
                    particles=particles_view,
                    systems=systems_view,
                    neighborlist=lambda _: EmptyNeighborList[Literal[0]](),
                    probe=None,
                ),
                parameter_view=parameter_view,
            )
        ),
        gradient_lens=lens(lambda x: x.graph, cls=GraphPotentialInput).nest(
            gradient_lens
        ),
        hessian_lens=hessian_lens,
        cache_lens=None,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )


def make_pme_potential[State, Ptch: Patch[Any], Gradients, Hessians](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasCell[Periodic3D]]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    parameter_view: View[State, PMEParameters],
    gradient_lens: Lens[PointCloud[IsEwaldPointData, HasCell[Periodic3D]], Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    include_exclusion_mask: bool = False,
) -> EwaldPotential[State, Gradients, Hessians, Ptch]:
    """Combine the Ewald real-space, self and background terms with the PME term.

    Args:
        particles_view: View of particles with charges and inclusion/exclusion groups.
        systems_view: View of the system table and periodic cells.
        neighborlist_view: View of the real-space pair neighbor list.
        parameter_view: View of PME parameters.
        gradient_lens: Lens selecting point-cloud variables to differentiate.
        hessian_lens: Lens selecting gradient entries to differentiate again.
        hessian_idx_view: View supplying Hessian row and column indices.
        patch_idx_view: Optional view supplying indices for output-cache updates.
        include_exclusion_mask: Whether to subtract molecular-exclusion pair energies.

    Returns:
        Potential with the components of ``make_ewald_potential``: four, or five
        with exclusions.
    """
    atomic_view = pipe(particles_view, _ewald_particle_table)
    sr_potential = make_ewald_short_range_potential(
        particles_view=atomic_view,
        systems_view=systems_view,
        neighborlist_view=neighborlist_view,
        parameter_view=parameter_view,
        probe=None,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
    lr_potential = make_pme_long_range_potential(
        particles_view=atomic_view,
        systems_view=systems_view,
        parameter_view=parameter_view,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
    self_potential = make_ewald_self_interaction_potential(
        particles_view=atomic_view,
        systems_view=systems_view,
        parameter_view=parameter_view,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
    net_charge_potential = make_ewald_net_charge_potential(
        particles_view=atomic_view,
        systems_view=systems_view,
        parameter_view=parameter_view,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
    components = (sr_potential, lr_potential, self_potential, net_charge_potential)
    if not include_exclusion_mask:
        return EwaldPotential(components)

    exclusion_correction = make_ewald_exclusion_correction_potential(
        particles_view=particles_view,
        systems_view=systems_view,
        probe=None,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
    return EwaldPotential((*components, exclusion_correction))


if TYPE_CHECKING:
    _lr: EnergyFunction[Any, PMEInput] = pme_long_range_energy
