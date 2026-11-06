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
from jax import Array

from kups.core.cell import Periodic3D
from kups.core.data import Table
from kups.core.patch import IdPatch, WithPatch
from kups.core.potential import Energy
from kups.core.typing import DTypeLike, HasCell, SystemId
from kups.core.utils.jax import dataclass, field, jit
from kups.core.utils.segment import segment_sum
from kups.potential.common.graph import GraphPotentialInput

from .parameters import IsEwaldPointData
from .potential import TO_STANDARD_UNITS, _influence


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
