# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Smooth particle-mesh Ewald (PME) reciprocal-space energy.

Charges are spread onto a mesh with cardinal B-splines, and the energy is a sum over
the mesh's Fourier modes (Essmann et al., J. Chem. Phys. 103, 8577, 1995). Forces and
cell gradients come from autodiff. PME evaluates whole systems and has no incremental
update; MC keeps the cached structure factors of direct Ewald.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jax import Array

from kups.core.cell import Cell, Periodic3D
from kups.core.data import Table
from kups.core.lens import Lens, View, lens
from kups.core.neighborlist import EmptyNeighborList, NeighborList
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import EMPTY_LENS, Energy, Potential, PotentialOut
from kups.core.typing import DTypeLike, HasCell, ParticleId, SystemId
from kups.core.utils.functools import pipe
from kups.core.utils.jax import dataclass, field, jit, no_jax_tracing
from kups.core.utils.segment import segment_sum
from kups.potential.common.energy import FullSumComposer, PotentialFromEnergy
from kups.potential.common.graph import (
    GraphConstructor,
    GraphInputConstructor,
    GraphPotentialInput,
    PointCloud,
)

from .parameters import IsEwaldPointData, estimate_ewald_parameters
from .potential import (
    TO_STANDARD_UNITS,
    EwaldPotential,
    _ewald_particle_table,
    _influence,
    make_ewald_exclusion_correction_potential,
    make_ewald_net_charge_potential,
    make_ewald_self_interaction_potential,
    make_ewald_short_range_potential,
)

# a_m of the P3M force error estimate per spline order, Deserno and Holm, J. Chem.
# Phys. 109, 7694 (1998), for ik differentiation and the optimal influence function
_P3M_ERROR_COEFFICIENTS = {
    4: (1 / 4320, 3 / 1936, 7601 / 2271360, 143 / 28800),
    6: (
        691 / 68140800,
        13 / 57600,
        47021 / 35512320,
        9694607 / 2095994880,
        733191589 / 59609088000,
        326190917 / 11700633600,
    ),
}
# PME's RMS force error per order: the P3M estimate with (h alpha)^(p-1) instead of
# (h alpha)^p, times a pair factor, and each charge's self force times a self factor,
# added in quadrature; the factors bound direct-Ewald errors of random systems
_PME_ERROR_FACTORS = {4: (1.8, 1.3), 6: (1.6, 1.0)}


@dataclass
class PMEParameters:
    """Per-system screening and one mesh shared by all systems.

    Attributes:
        alpha: Screening parameter (1/Å) per system.
        cutoff: Real-space cutoff (Å) per system, read by the real-space term.
        mesh: Mesh points along each cell vector; static, as FFT shapes must be.
        order: Even cardinal B-spline order, at most the smallest mesh size.
        reciprocal_dtype: Precision of the charge mesh and its FFT.
    """

    alpha: Table[SystemId, Array]
    cutoff: Table[SystemId, Array]
    mesh: tuple[int, int, int] = field(static=True)
    order: int = field(static=True, default=6)
    reciprocal_dtype: DTypeLike = field(static=True, default=jnp.float64)

    def __post_init__(self) -> None:
        if self.order < 2 or self.order % 2 or min(self.mesh) < self.order:
            raise ValueError(
                "PME needs an even spline order of at most the mesh size; got order "
                f"{self.order} and mesh {self.mesh}."
            )

    @classmethod
    @no_jax_tracing
    def make(
        cls,
        charges: Table[ParticleId, IsEwaldPointData],
        cell: Table[SystemId, HasCell[Periodic3D]],
        epsilon_total: float = 1e-8,
        real_cutoff: float | None = None,
        *,
        reciprocal_dtype: DTypeLike = jnp.float64,
    ) -> PMEParameters:
        """Estimate screening, cutoffs, mesh and spline order for each system.

        Screening and cutoffs follow ``estimate_ewald_parameters``. The mesh is about
        as accurate as the direct sum at the estimated ``k_max``; the order is the
        cheaper of 4 and 6, and the mesh is the per-axis maximum over systems.

        Args:
            charges: Particle table with charges and system assignments.
            cell: System table with periodic cells.
            epsilon_total: Target combined real-space and reciprocal-space error.
            real_cutoff: Optional fixed real-space cutoff in Å.
            reciprocal_dtype: Precision of the charge mesh and its FFT.

        Returns:
            Parameters whose mesh meets the target in every system.
        """
        system = np.asarray(charges.data.system.indices_in(cell.keys))
        values = np.asarray(charges.data.charges)
        cells = [cell.data.cell[i] for i in range(cell.size)]
        estimates = [
            estimate_ewald_parameters(
                values[system == i],
                c,
                real_cutoff=real_cutoff,
                epsilon_total=epsilon_total,
            )
            for i, c in enumerate(cells)
        ]

        def mesh(order: int) -> tuple[int, int, int]:
            sizes = [
                estimate_pme_mesh(values[system == i], c, e.alpha, e.k_max, order)
                for i, (c, e) in enumerate(zip(cells, estimates, strict=True))
            ]
            mx, my, mz = np.max(sizes, axis=0).tolist()
            return mx, my, mz

        def cost(order: int) -> float:
            points = np.prod(mesh(order))
            n_particles = np.sum(system < cell.size)
            return n_particles * order**3 + cell.size * points * np.log2(points)

        order = min(_PME_ERROR_FACTORS, key=cost)
        return cls(
            alpha=cell.set_data(jnp.asarray([e.alpha for e in estimates])),
            cutoff=cell.set_data(jnp.asarray([e.real_cutoff for e in estimates])),
            mesh=mesh(order),
            order=order,
            reciprocal_dtype=reciprocal_dtype,
        )


@no_jax_tracing
def estimate_pme_mesh(
    charges: Array | np.ndarray,
    cell: Cell[Periodic3D],
    /,
    alpha: float,
    k_max: float,
    order: int,
) -> tuple[int, int, int]:
    """Smallest 2-3-5 mesh about as accurate as the direct sum cut at ``k_max``.

    Both are RMS force errors: the direct sum's from Kolafa and Perram (1992), the
    mesh's from the P3M estimate of Deserno and Holm (1998), fitted to PME.

    Args:
        charges: Charges in e of one system, ``(n_particles,)``.
        cell: Its periodic cell.
        alpha: Screening parameter in 1/Å.
        k_max: Reciprocal cutoff of the direct sum in 1/Å.
        order: Spline order, 4 or 6.

    Returns:
        Mesh points along each cell vector, each at least ``order``.
    """
    q = np.asarray(charges, np.float64)
    widths = np.asarray(cell.perpendicular_lengths).reshape(3)
    if not np.any(q):
        return order, order, order
    volume = float(np.asarray(cell.volume).squeeze())
    length = volume ** (1 / 3)
    pair = q @ q / np.sqrt(q.size)
    decay = np.exp(-(k_max**2) / (4 * alpha**2))
    target = pair * alpha * np.sqrt(8 / (volume * k_max)) * decay
    pair_factor, self_factor = _PME_ERROR_FACTORS[order]
    scale = np.hypot(
        pair_factor * pair * np.sqrt(alpha * length * np.sqrt(2 * np.pi)) / length**2,
        self_factor * np.sqrt(np.mean(q**4)) * alpha**2,
    )

    def error(spacing: float) -> float:
        x = spacing * alpha
        series = sum(
            a * x ** (2 * m) for m, a in enumerate(_P3M_ERROR_COEFFICIENTS[order])
        )
        return scale * x ** (order - 1) * np.sqrt(series)

    # the error grows with the spacing; the coarsest mesh has ``order`` points
    spacing = widths.min() / order
    if error(spacing) > target:
        spacing = scipy.optimize.brentq(
            lambda h: np.log(error(h) / target), 1e-9 * spacing, spacing
        )
    mx, my, mz = (_next_2_3_5(max(order, int(np.ceil(w / spacing)))) for w in widths)
    return mx, my, mz


def _next_2_3_5(n: int) -> int:
    """Smallest integer of at least ``n`` without prime factors above 5."""
    while True:
        rest = n
        for prime in (2, 3, 5):
            while rest % prime == 0:
                rest //= prime
        if rest == 1:
            return n
        n += 1


type PMEInput = GraphPotentialInput[
    PMEParameters, IsEwaldPointData, HasCell[Periodic3D], Literal[0]
]


@jit
def pme_long_range_energy(
    inp: PMEInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Reciprocal-space energy from the B-spline charge mesh.

    Args:
        inp: Particles, periodic cells and PME parameters.

    Returns:
        Energy (eV) of each system without the neutralizing background, in the
        dtype of the positions, and an identity patch.
    """
    params, systems = inp.parameters, inp.graph.systems
    particles = inp.graph.particles.data
    dtype = particles.positions.dtype
    system = particles.system.indices_in(systems.keys)
    valid = system < systems.size
    n_points = int(np.prod(params.mesh))
    if (systems.size + 1) * n_points > jnp.iinfo(system.dtype).max:
        raise ValueError(f"{systems.size} meshes {params.mesh} exceed {system.dtype}.")
    cells = systems.map_data(lambda s: s.cell.materialize())
    cell = cells[particles.system]
    positions = jnp.where(valid[:, None], particles.positions, 0.0)
    fractional = cell.fold(cell.frame.to_fractional(positions))[0]
    u = jnp.where(valid[:, None], fractional, 0.0) * jnp.asarray(params.mesh, dtype)
    base = jnp.floor(u)
    w = _bspline_weights(u - base, params.order)  # (n, 3, order)
    # weight j belongs to mesh point floor(u) - j; ids run over all systems' meshes
    sizes = jnp.asarray(params.mesh, system.dtype)[:, None]
    j = jnp.arange(params.order, dtype=system.dtype)
    x, y, z = jnp.moveaxis((base.astype(system.dtype)[..., None] - j) % sizes, 1, 0)
    ids = (
        (system[:, None, None, None] * sizes[0] + x[:, :, None, None]) * sizes[1]
        + y[:, None, :, None]
    ) * sizes[2] + z[:, None, None, :]
    charges = jnp.where(valid, particles.charges, 0.0)
    spread = (
        charges[:, None, None, None]
        * w[:, 0, :, None, None]
        * w[:, 1, None, :, None]
        * w[:, 2, None, None, :]
    )
    grid = segment_sum(
        spread.astype(params.reciprocal_dtype), ids, systems.size * n_points
    )
    modes = jnp.fft.rfftn(grid.reshape(systems.size, *params.mesh), axes=(1, 2, 3))
    power = (modes.real**2 + modes.imag**2).astype(dtype)
    # k = 2π h⁻¹ m for integer modes m, so k² = mᵀ G m
    inverse = cells.data.inverse_vectors
    metric = (2 * jnp.pi) ** 2 * jnp.swapaxes(inverse, -1, -2) @ inverse
    mx, my, mz = params.mesh
    half = mz // 2 + 1
    m = jnp.stack(
        jnp.meshgrid(
            jnp.fft.fftfreq(mx, 1 / mx, dtype=dtype),
            jnp.fft.fftfreq(my, 1 / my, dtype=dtype),
            jnp.arange(half, dtype=dtype),
            indexing="ij",
        ),
        axis=-1,
    )
    k_squared = jnp.einsum("xyza,sab,xyzb->sxyz", m, metric, m)
    alpha = Table.broadcast_to(params.alpha, systems).data[:, None, None, None]
    volume = cells.data.volume[:, None, None, None]
    weights = (
        _euler_factors(mx, params.order)[:, None, None]
        * _euler_factors(my, params.order)[None, :, None]
        * (_euler_factors(mz, params.order)[:half] * _rfft_weights(mz))
    ).astype(dtype)
    influence = _influence(k_squared, alpha, volume)
    energy = jnp.sum(influence * weights * power, axis=(1, 2, 3)) * TO_STANDARD_UNITS
    return WithPatch(systems.set_data(energy.astype(dtype)), IdPatch[Any]())


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
    """Create the PME reciprocal-space potential; it evaluates whole systems.

    Args:
        particles_view: View of the particle table.
        systems_view: View of the system table and periodic cells.
        parameter_view: View of PME parameters.
        gradient_lens: Lens selecting point-cloud variables to differentiate.
        hessian_lens: Lens selecting gradient entries to differentiate again.
        hessian_idx_view: View supplying Hessian row and column indices.
        patch_idx_view: Optional view supplying indices for output-cache updates.

    Returns:
        Reciprocal potential without probe or cache.
    """
    graph = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda _: EmptyNeighborList[Literal[0]](),
        probe=None,
    )
    return PotentialFromEnergy(
        energy_fn=pme_long_range_energy,
        composer=FullSumComposer(GraphInputConstructor(graph, parameter_view)),
        gradient_lens=lens(lambda x: x.graph).nest(gradient_lens),
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=None,
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
    derivatives = (gradient_lens, hessian_lens, hessian_idx_view, patch_idx_view)
    components = (
        make_ewald_short_range_potential(
            atomic_view,
            systems_view,
            neighborlist_view,
            parameter_view,
            None,
            *derivatives,
        ),
        make_pme_long_range_potential(
            atomic_view, systems_view, parameter_view, *derivatives
        ),
        make_ewald_self_interaction_potential(
            atomic_view, systems_view, parameter_view, *derivatives
        ),
        make_ewald_net_charge_potential(
            atomic_view, systems_view, parameter_view, *derivatives
        ),
    )
    if include_exclusion_mask:
        exclusions = make_ewald_exclusion_correction_potential(
            particles_view, systems_view, None, *derivatives
        )
        components = (*components, exclusions)
    return EwaldPotential(components)


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
