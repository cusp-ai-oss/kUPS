# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Grimme's D3 dispersion correction with Becke-Johnson damping, natively in JAX.

D3 adds the long-range correlation energy that semi-local DFT — and MLIPs trained
on it — systematically miss. It is a *correction*: compose it with any other
[Potential][kups.core.potential.Potential] via
[sum_potentials][kups.core.potential.sum_potentials].

The pairwise energy is

$$
E_\text{disp} = -\frac{1}{2}\sum_{i}\sum_{j,\mathbf{T}}\left[
    \frac{s_6\,C^{ij}_6}{r^6 + (R^{ij}_0)^6}
  + \frac{s_8\,C^{ij}_8}{r^8 + (R^{ij}_0)^8}\right],
$$

summed over neighbours $j$ and lattice translations $\mathbf{T}$, including
$j = i$ with $\mathbf{T} \neq 0$. The dispersion coefficients depend on the
chemical environment through a coordination number

$$
\mathrm{CN}_i = \sum_{j,\mathbf{T}}
    \frac{1}{1 + \exp\!\left[-k_1\left(\tfrac{R^\text{cov}_i + R^\text{cov}_j}{r} - 1\right)\right]},
$$

which selects a Gaussian-weighted interpolation between tabulated reference
systems,

$$
C^{ij}_6 = \sum_{a,b} \hat g^a_i\, C^{ij,ab}_{6,\text{ref}}\, \hat g^b_j,
\qquad
\hat g^a_i \propto \exp\!\left[-k_3 (\mathrm{CN}_i - \mathrm{CN}^a_i)^2\right].
$$

Because the joint pair weight factorizes, the normalization is done once per atom
rather than once per pair.

**Scope.** This module implements the pairwise (two-body) term with Becke-Johnson
damping. The Axilrod-Teller-Muto three-body term and the zero-damping variants are
not implemented.

**Cutoffs.** D3's real-space sum converges absolutely, so no Ewald machinery is
needed, but it is genuinely long-ranged: ``simple-dftd3`` defaults to 60 a0
(31.75 Å). kUPS defaults to a much cheaper
[D3_DEFAULT_CUTOFF][kups.potential.dispersion.d3.D3_DEFAULT_CUTOFF]; see that
constant for the accuracy trade-off.

**Gradients.** Only the energy is written here. Forces, cell gradients and hence
stress come from [PotentialFromEnergy][kups.potential.common.energy.PotentialFromEnergy]
by automatic differentiation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.cell import AnyPeriodicity
from kups.core.constants import BOHR
from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import NeighborList
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import Energy, Potential, PotentialOut
from kups.core.typing import (
    HasAtomicNumbers,
    HasCell,
    HasExclusionIndex,
    HasInclusionIndex,
    HasPositions,
    HasSystemIndex,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, jit, no_jax_tracing
from kups.potential.common.energy import EnergyFunction, PotentialFromEnergy
from kups.potential.common.graph import (
    FullGraphSumComposer,
    GraphConstructor,
    GraphPotentialInput,
)
from kups.potential.dispersion.damping import damping_for_functional
from kups.potential.dispersion.data import (
    MAX_ATOMIC_NUMBER,
    D3Reference,
    load_d3_reference,
)

__all__ = [
    "D3_DEFAULT_CN_CUTOFF",
    "D3_DEFAULT_CUTOFF",
    "D3_REFERENCE_CN_CUTOFF",
    "D3_REFERENCE_CUTOFF",
    "K_COORDINATION",
    "K_WEIGHT",
    "D3Parameters",
    "IsD3Particles",
    "d3_c6_coefficients",
    "d3_coordination_numbers",
    "d3_energy",
    "make_d3_potential",
]

K_COORDINATION = 16.0
"""Steepness ``k1`` of the coordination-number counting function."""

K_WEIGHT = 4.0
"""Exponent ``k3`` of the Gaussian reference weighting."""

D3_REFERENCE_CUTOFF = 60.0 * BOHR
"""``simple-dftd3``'s own dispersion cutoff (60 a0 ≈ 31.75 Å).

Use this to reproduce a stock ``dftd3`` run exactly. It is expensive: at a typical
condensed-phase density it implies of order 10⁴ neighbours per atom.
"""

D3_REFERENCE_CN_CUTOFF = 40.0 * BOHR
"""``simple-dftd3``'s own coordination-number cutoff (40 a0 ≈ 21.17 Å).

Setting both this and
[D3_REFERENCE_CUTOFF][kups.potential.dispersion.d3.D3_REFERENCE_CUTOFF] reproduces
a stock ``dftd3`` run, with one measure-zero exception: kUPS excludes a pair at
*exactly* ``r == cutoff`` (strict ``<``, matching
[DistanceCutoffMask][kups.core.neighborlist.masks.DistanceCutoffMask] and every
other kUPS pair potential), whereas ``simple-dftd3`` includes it. The two agree
for every separation either side of the boundary.
"""

D3_DEFAULT_CUTOFF = 15.0
"""Default dispersion cutoff [Å].

**This deviates deliberately from ``simple-dftd3``**, which uses
[D3_REFERENCE_CUTOFF][kups.potential.dispersion.d3.D3_REFERENCE_CUTOFF]. The
neglected tail is roughly ``4πρ⟨C6⟩ / (3 R³)`` per atom — of order 2 meV/atom at
15 Å for a dense organic solid — and is slowly varying, so forces converge much
faster with cutoff than energies do. Because it is volume dependent it biases
stress more than forces. Raise it (and ``cn_cutoff``) when absolute energies or
pressures matter, or when reproducing published D3 numbers.

**The truncation is abrupt.** No shift and no switching function are applied, at
this or any other cutoff -- matching ``simple-dftd3``, which is what makes the
reference agreement exact. So the energy steps as a pair crosses: about 2.6 µeV
for a carbon pair at 15 Å. Spread over a shell that thin at a condensed-phase
density (``2 pi R^2 rho dr`` pair crossings per atom, halved because a pair
energy is shared) that is of order 2 µeV/atom per 0.01 Å of motion, with a
correspondingly discontinuous force. That is
negligible against the correction itself but is not zero, so it puts a floor on
NVE energy conservation and on how tight a force convergence criterion is
meaningful. Composing D3 with a smoothly cut-off MLIP does not remove it. If a
smooth cutoff matters more than reproducing ``dftd3``, wrap the pair term the way
[PairTailCorrectedLennardJonesParameters][kups.potential.classical.lennard_jones.PairTailCorrectedLennardJonesParameters]
does rather than raising the cutoff.
"""

D3_DEFAULT_CN_CUTOFF = D3_REFERENCE_CN_CUTOFF
"""Default coordination-number cutoff [Å], capped at the dispersion cutoff.

``D3Parameters`` takes ``min(cutoff, cn_cutoff)`` so that a single neighbor list
built at ``cutoff`` always suffices. The counting function decays to a floor of
``exp(-k1) ≈ 1e-7`` per pair rather than to zero, so the coordination number picks
up a small, cutoff-dependent tail (of order 1e-3 for a dense solid between 15 Å
and 21 Å). Match this value to ``simple-dftd3`` when comparing against it.
"""

_UNUSED_REFERENCE_WEIGHT = 0.0


@no_jax_tracing
def _resolve_cutoffs(
    cutoff: float | Array, cn_cutoff: float | Array | None
) -> tuple[Array, Array]:
    """Broadcast and validate the two cutoffs against each other.

    Both accept a scalar or a per-system array, in either combination — they are
    broadcast symmetrically, so a scalar pair cutoff with a per-system CN cutoff
    is fine.

    ``cn_cutoff`` is clamped to ``cutoff`` because a single neighbor list built at
    ``cutoff`` has to serve both. Clamping the default is silent; clamping a value
    the caller asked for explicitly warns, since that changes their result.

    Raises:
        ValueError: If a cutoff is empty, non-finite, or not strictly positive.
    """
    explicit = cn_cutoff is not None
    cutoff_arr, cn_arr = jnp.broadcast_arrays(
        jnp.atleast_1d(jnp.asarray(cutoff, dtype=float)),
        jnp.atleast_1d(
            jnp.asarray(
                D3_DEFAULT_CN_CUTOFF if cn_cutoff is None else cn_cutoff, dtype=float
            )
        ),
    )
    for name, values in (("cutoff", cutoff_arr), ("cn_cutoff", cn_arr)):
        array = np.asarray(values)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite; got {array.tolist()}.")
        if (array <= 0).any():
            raise ValueError(f"{name} must be strictly positive; got {array.tolist()}.")
    if explicit and bool((cn_arr > cutoff_arr).any()):
        warnings.warn(
            "cn_cutoff exceeds cutoff and was reduced to it: a single neighbor "
            "list built at cutoff serves both. Raise cutoff to keep the "
            "requested coordination-number range.",
            stacklevel=3,
        )
    return cutoff_arr, jnp.minimum(cn_arr, cutoff_arr)


@runtime_checkable
class IsD3Particles(HasPositions, HasAtomicNumbers, HasSystemIndex, Protocol):
    """Particle data D3 reads: positions, atomic numbers and system assignment."""


@runtime_checkable
class IsD3GraphParticles(IsD3Particles, HasInclusionIndex, HasExclusionIndex, Protocol):
    """[IsD3Particles][kups.potential.dispersion.d3.IsD3Particles] plus the traits every neighbor list needs."""


@dataclass
class D3Parameters:
    """D3(BJ) damping parameters and reference tables, in kUPS units.

    Build with [from_functional][kups.potential.dispersion.d3.D3Parameters.from_functional]
    or [from_damping][kups.potential.dispersion.d3.D3Parameters.from_damping] rather
    than directly: they load the reference tables and convert ``a2`` from the Bohr
    used in the literature.

    Attributes:
        s6: Scaling of the ``C6`` term [dimensionless].
        s8: Scaling of the ``C8`` term [dimensionless].
        a1: Becke-Johnson damping parameter [dimensionless].
        a2: Becke-Johnson damping parameter [Å].
        cutoff: Dispersion pair cutoff [Å], shape ``(n_systems,)``.
        cn_cutoff: Coordination-number cutoff [Å], shape ``(n_systems,)``.
            Never exceeds ``cutoff``, so one neighbor list serves both.
        reference: Element tables, see
            [D3Reference][kups.potential.dispersion.data.D3Reference].
    """

    s6: Array  # () float, dimensionless
    s8: Array  # () float, dimensionless
    a1: Array  # () float, dimensionless
    a2: Array  # () float [Å]
    cutoff: Table[SystemId, Array]  # (n_systems,) float [Å]
    cn_cutoff: Table[SystemId, Array]  # (n_systems,) float [Å]
    reference: D3Reference  # element tables, see kups.potential.dispersion.data

    @classmethod
    def from_damping(
        cls,
        *,
        s8: float,
        a1: float,
        a2: float,
        s6: float = 1.0,
        cutoff: float | Array = D3_DEFAULT_CUTOFF,
        cn_cutoff: float | Array | None = None,
    ) -> D3Parameters:
        """Build parameters from explicit D3(BJ) damping values.

        Args:
            s8: Scaling of the ``C8`` term.
            a1: Becke-Johnson damping parameter.
            a2: Becke-Johnson damping parameter **in Bohr**, as published.
            s6: Scaling of the ``C6`` term; ``1.0`` except for double hybrids.
            cutoff: Dispersion pair cutoff [Å], scalar or per system.
            cn_cutoff: Coordination-number cutoff [Å], scalar or per system.
                Defaults to ``min(cutoff, D3_DEFAULT_CN_CUTOFF)``; an explicit
                value larger than ``cutoff`` is clamped with a warning.

        Returns:
            Parameters ready to pass to a D3 potential factory.

        Raises:
            ValueError: If a cutoff is empty, non-finite, or not strictly positive.
        """
        cutoff_arr, cn_arr = _resolve_cutoffs(cutoff, cn_cutoff)
        return cls(
            s6=jnp.asarray(s6, dtype=float),
            s8=jnp.asarray(s8, dtype=float),
            a1=jnp.asarray(a1, dtype=float),
            a2=jnp.asarray(a2 * BOHR, dtype=float),
            cutoff=Table.arange(cutoff_arr, label=SystemId),
            cn_cutoff=Table.arange(cn_arr, label=SystemId),
            reference=load_d3_reference(),
        )

    @classmethod
    def from_functional(
        cls,
        functional: str,
        *,
        cutoff: float | Array = D3_DEFAULT_CUTOFF,
        cn_cutoff: float | Array | None = None,
    ) -> D3Parameters:
        """Build parameters from a density functional's published D3(BJ) fit.

        Args:
            functional: Functional name, matched case- and punctuation-insensitively
                against
                [BECKE_JOHNSON_PARAMETERS][kups.potential.dispersion.damping.BECKE_JOHNSON_PARAMETERS].
            cutoff: Dispersion pair cutoff [Å], scalar or per system.
            cn_cutoff: Coordination-number cutoff [Å]; see
                [from_damping][kups.potential.dispersion.d3.D3Parameters.from_damping].

        Returns:
            Parameters ready to pass to a D3 potential factory.

        Raises:
            KeyError: If the functional has no tabulated parameters.
            ValueError: If a cutoff is empty, non-finite, or not strictly positive.
        """
        damping = damping_for_functional(functional)
        return cls.from_damping(
            s6=damping.s6,
            s8=damping.s8,
            a1=damping.a1,
            a2=damping.a2,
            cutoff=cutoff,
            cn_cutoff=cn_cutoff,
        )


type D3Input = GraphPotentialInput[
    D3Parameters, IsD3Particles, HasCell[AnyPeriodicity], Literal[2]
]

type D3GraphInput = GraphPotentialInput[
    D3Parameters, IsD3GraphParticles, HasCell[AnyPeriodicity], Literal[2]
]
"""What the factory actually builds: the same input, but over particles that also
carry the inclusion/exclusion traits the neighbor list needs.

Separate from [D3Input][kups.potential.dispersion.d3.D3Input] because
``SumComposer`` is invariant in its input type, so the narrower particle protocol
the kernels declare is not substitutable for the wider one the graph produces.
Mirrors the ``LennardJonesInput`` / ``LJRadiusInp`` pair.
"""


def _occupied_atomic_numbers(
    particles: Table[ParticleId, IsD3Particles],
) -> tuple[Array, Array]:
    """Atomic numbers paired with the mask marking which slots are real.

    A particle buffer -- what a grand-canonical state keeps its particles in --
    marks a slot unoccupied by sending its index out of bounds, and zeroes every
    other leaf of that row. The atomic number therefore reads back as ``Z = 0``,
    which is not an element. That is exactly the criterion
    [InBoundsMask][kups.core.neighborlist.masks.InBoundsMask] already uses to
    drop those rows from the neighbor list, so it is reused here rather than
    treating ``Z = 0`` as a sentinel in its own right: a genuine ``Z = 0`` in an
    *occupied* slot is still a bug and must still be caught.
    """
    return particles.data.atomic_numbers, particles.data.system.valid_mask


def _validate_atomic_numbers_if_concrete(
    atomic_numbers: Array, occupied: Array | None = None
) -> None:
    """Run the eager check unless a JAX transformation owns the values."""
    if any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree.leaves((atomic_numbers, occupied))
    ):
        return
    try:
        if occupied is not None:
            atomic_numbers = atomic_numbers[occupied]
        validate_atomic_numbers(atomic_numbers)
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerArrayConversionError,
    ):
        # Third-party JAX interpreters may use tracers outside jax.core.Tracer.
        pass


def _assert_supported_atomic_numbers(
    atomic_numbers: Array, occupied: Array | None = None
) -> None:
    """Fail loudly on atomic numbers the reference tables do not cover.

    Without this, an out-of-range ``Z`` is silently clamped by JAX's default
    gather behavior — ``Z = 104`` would quietly borrow lawrencium's
    coefficients and return a plausible but wrong energy. Uses
    [runtime_assert][kups.core.assertion.runtime_assert] so the check survives
    ``jit`` and surfaces through the standard evaluation path.

    Args:
        atomic_numbers: Atomic numbers of every slot, occupied or not.
        occupied: Marks the slots that hold a real particle; ``None`` means all
            of them do. Unoccupied slots are exempt, since a buffer zeroes them
            (see ``_occupied_atomic_numbers``). They are inert either way, so
            rejecting them would only block grand-canonical states.
    """
    _validate_atomic_numbers_if_concrete(atomic_numbers, occupied)
    supported = (atomic_numbers >= 1) & (atomic_numbers <= MAX_ATOMIC_NUMBER)
    if occupied is not None:
        supported = supported | ~occupied
    if atomic_numbers.size:
        # report the range over occupied slots only, or the message would name
        # a padding zero the caller never wrote
        real = (
            jnp.ones_like(atomic_numbers, dtype=bool) if occupied is None else occupied
        )
        lo = jnp.min(jnp.where(real, atomic_numbers, MAX_ATOMIC_NUMBER))
        hi = jnp.max(jnp.where(real, atomic_numbers, 1))
    else:
        lo = jnp.asarray(1)
        hi = jnp.asarray(MAX_ATOMIC_NUMBER)
    runtime_assert(
        supported.all(),
        f"D3 covers atomic numbers 1..{MAX_ATOMIC_NUMBER}; got {{lo}}..{{hi}}.",
        fmt_args={"lo": lo, "hi": hi},
        exception_type=ValueError,
    )


def _edge_geometry(inp: D3Input) -> tuple[Array, Array, Array]:
    """Squared edge lengths, guarded against padded rows.

    Padded edges carry an out-of-bounds index and a zero shift, so both endpoints
    gather to the same position and ``r`` would be exactly zero — fatal for the
    ``r^-6`` and ``r_c / r`` expressions here. ``segment_sum`` drops those rows from
    the energy, but a zero cotangent times an infinite partial derivative still
    produces a NaN gradient, so the distance is replaced *before* any division.

    Returns:
        ``(r2, valid, edge_z)`` where ``r2`` is safe to divide by, ``valid`` marks
        real edges and ``edge_z`` is the ``(n_edges, 2)`` atomic-number pair.
    """
    graph = inp.graph
    assert graph.edges.indices.shape[-1] == 2, "D3 consumes pair edges"
    valid = graph.edges.indices.valid_mask.all(axis=-1)
    delta = graph.edge_shifts[:, 0]
    r2 = jnp.sum(delta * delta, axis=-1)
    r2 = jnp.where(valid, r2, 1.0)
    edge_z = graph.particles[graph.edges.indices].atomic_numbers
    edge_z = jnp.where(valid[:, None], edge_z, 0)
    return r2, valid, edge_z


def _edge_cutoff_squared(cutoff: Table[SystemId, Array], inp: D3Input) -> Array:
    """Squared per-edge cutoff, resolved from the per-system table.

    Goes through ``Table.broadcast_to`` rather than gathering the raw array with
    the edge's system id. Both accept the two shapes callers actually pass — one
    entry for every system, or a single entry standing for all of them — but a
    raw gather answers a *third*, wrong case silently: JAX clamps an
    out-of-bounds index, so a table with fewer entries than the batch has systems
    would hand the surplus systems the last system's cutoff and return a
    plausible energy. ``broadcast_to`` rejects that, and it is what
    [DistanceCutoffMask][kups.core.neighborlist.masks.DistanceCutoffMask] already
    uses to size the neighbor list from the same tables.
    """
    per_system = Table.broadcast_to(cutoff, inp.graph.systems)
    return per_system[inp.graph.edge_batch_mask] ** 2


@jit
def d3_coordination_numbers(inp: D3Input) -> Array:
    """Fractional D3 coordination number per atom.

    Uses the exponential counting function with ``k1 = 16`` and covalent radii
    scaled by ``4/3``, truncated hard at ``cn_cutoff`` exactly as ``simple-dftd3``
    does. Self-image pairs contribute; the ``i == j``, ``T == 0`` pair does not,
    because the neighbor list never emits it.

    Args:
        inp: Graph input carrying particles, edges and D3 parameters.

    Returns:
        Coordination numbers, shape ``(n_particles,)``.
    """
    graph = inp.graph
    r2, valid, edge_z = _edge_geometry(inp)
    radii = inp.parameters.reference.covalent_radii[edge_z]  # (n_edges, 2)
    r_cov = radii[:, 0] + radii[:, 1]
    r = jnp.sqrt(r2)
    count = jax.nn.sigmoid(K_COORDINATION * (r_cov / r - 1.0))
    within = r2 < _edge_cutoff_squared(inp.parameters.cn_cutoff, inp)
    count = jnp.where(valid & within, count, 0.0)
    return jax.ops.segment_sum(
        count,
        graph.edges.indices.indices[:, 0],
        num_segments=len(graph.particles),
        mode="drop",
    )


def _reference_weights(inp: D3Input, coordination_numbers: Array) -> Array:
    """Per-atom normalized Gaussian weights over the reference systems.

    The joint pair weight factorizes into a product of per-atom weights, so the
    joint normalization equals the product of these — an ``O(n_atoms * n_ref)``
    normalization instead of ``O(n_edges * n_ref^2)``.

    Computed with the largest exponent subtracted. That is algebraically exact for
    a softmax and removes any chance of the normalization underflowing, which is
    what ``simple-dftd3`` guards against with an explicit fallback.
    """
    reference = inp.parameters.reference
    z = inp.graph.particles.data.atomic_numbers
    reference_cn = reference.reference_cn[z]  # (n_particles, n_ref)
    mask = reference.reference_mask[z]

    exponent = -K_WEIGHT * (coordination_numbers[:, None] - reference_cn) ** 2
    # a finite stand-in for masked slots keeps the max (and its gradient) clean
    largest = jnp.max(jnp.where(mask, exponent, -jnp.inf), axis=-1, keepdims=True)
    largest = jnp.where(jnp.isfinite(largest), largest, 0.0)
    # zero the exponent on masked slots *before* exp so no branch can overflow
    shifted = jnp.where(mask, exponent - largest, 0.0)
    weights = jnp.where(mask, jnp.exp(shifted), _UNUSED_REFERENCE_WEIGHT)
    total = jnp.sum(weights, axis=-1, keepdims=True)
    return weights / jnp.where(total > 0, total, 1.0)


@jit
def d3_c6_coefficients(inp: D3Input, coordination_numbers: Array) -> Array:
    """Environment-dependent ``C6`` coefficient per edge [eV·Å⁶].

    Args:
        inp: Graph input carrying particles, edges and D3 parameters.
        coordination_numbers: Per-atom coordination numbers, shape ``(n_particles,)``.

    Returns:
        Pair coefficients, shape ``(n_edges,)``.
    """
    graph = inp.graph
    weights = _reference_weights(inp, coordination_numbers)
    edge_weights = weights[graph.edges.indices.indices]  # (n_edges, 2, n_ref)
    _, valid, edge_z = _edge_geometry(inp)
    reference_c6 = inp.parameters.reference.reference_c6[edge_z[:, 0], edge_z[:, 1]]
    c6 = jnp.einsum(
        "ea,eab,eb->e", edge_weights[:, 0], reference_c6, edge_weights[:, 1]
    )
    return jnp.where(valid, c6, 0.0)


@jit
def d3_edge_energy(inp: D3Input) -> Array:
    """D3(BJ) energy contribution per directed edge [eV].

    Each unordered pair — periodic images included — appears twice in the edge
    list, so the caller halves the sum.
    """
    r2, valid, edge_z = _edge_geometry(inp)
    parameters = inp.parameters
    reference = parameters.reference

    coordination_numbers = d3_coordination_numbers(inp)
    c6 = d3_c6_coefficients(inp, coordination_numbers)

    r4r2 = reference.r4r2[edge_z]  # (n_edges, 2)
    # C8 = 3 * sqrt(Q_i Q_j) * C6, with the stored r4r2 already carrying the sqrt
    rr = 3.0 * r4r2[:, 0] * r4r2[:, 1]
    c8 = rr * c6
    r0 = parameters.a1 * jnp.sqrt(rr) + parameters.a2

    r6 = r2**3
    r8 = r6 * r2
    energy = -(parameters.s6 * c6 / (r6 + r0**6) + parameters.s8 * c8 / (r8 + r0**8))
    within = r2 < _edge_cutoff_squared(parameters.cutoff, inp)
    return jnp.where(valid & within, energy, 0.0)


def d3_energy(inp: D3Input) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Total pairwise D3(BJ) dispersion energy per system [eV].

    Args:
        inp: Graph input carrying particles, edges and D3 parameters.

    Returns:
        Per-system energies and an identity patch (D3 caches nothing).
    """
    _assert_supported_atomic_numbers(*_occupied_atomic_numbers(inp.graph.particles))
    edge_energy = d3_edge_energy(inp)
    total_energies = inp.graph.edge_batch_mask.sum_over(edge_energy) / 2
    return WithPatch(total_energies, IdPatch[Any]())


def validate_atomic_numbers(atomic_numbers: Array) -> None:
    """Eagerly raise if any atomic number falls outside the D3 parameterisation.

    A convenience for host-side, one-shot use where a concrete array is already
    available. The authoritative check is ``_assert_supported_atomic_numbers``,
    which runs inside [d3_energy][kups.potential.dispersion.d3.d3_energy] on
    every evaluation and therefore also covers traced code.

    Args:
        atomic_numbers: Atomic numbers to check.

    Raises:
        ValueError: If any value is outside ``1..MAX_ATOMIC_NUMBER``.
    """
    z = jnp.asarray(atomic_numbers)
    if z.size == 0:
        return
    lo, hi = int(jnp.min(z)), int(jnp.max(z))
    if lo < 1 or hi > MAX_ATOMIC_NUMBER:
        raise ValueError(
            f"D3 covers atomic numbers 1..{MAX_ATOMIC_NUMBER}; got {lo}..{hi}."
        )


class _ValidatedD3Potential[State, Gradients, Hessians, Ptch: Patch[Any]]:
    """Add eager host validation to a JAX-compatible D3 potential.

    ``runtime_assert`` remains authoritative in transformed evaluation paths. A
    direct call has concrete atomic numbers before the wrapped potential enters
    its internal ``jit``, so validate them here as well instead of silently
    discarding the assertion primitive.
    """

    def __init__(
        self,
        potential: Potential[State, Gradients, Hessians, Ptch],
        particles_view: View[State, Table[ParticleId, IsD3GraphParticles]],
    ) -> None:
        self.potential = potential
        self.particles_view = particles_view

    def __call__(
        self, state: State, patch: Ptch | None = None
    ) -> WithPatch[PotentialOut[Gradients, Hessians], Patch[State]]:
        _validate_atomic_numbers_if_concrete(
            *_occupied_atomic_numbers(self.particles_view(state))
        )
        return self.potential(state, patch)


def make_d3_potential[State, Gradients, Hessians](
    particles_view: View[State, Table[ParticleId, IsD3GraphParticles]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    parameter_view: View[State, D3Parameters],
    gradient_lens: Lens[D3GraphInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Patch[Any]]:
    """Create a pairwise D3(BJ) dispersion potential.

    The patch type is fixed rather than generic: D3 takes no ``probe``, so there
    is no incremental path whose patch type could vary.

    Uses [FullGraphSumComposer][kups.potential.common.graph.FullGraphSumComposer]
    rather than the incremental local composer: the coordination number couples
    atoms beyond the pair being moved, so an ``old edges``/``new edges``
    decomposition would silently omit the change in ``C6`` of the moved atom's
    neighbours.

    Args:
        particles_view: Extracts the particle table.
        systems_view: Extracts the system table (cells).
        neighborlist_view: Extracts the pair neighbor list, built at ``cutoff``.
        parameter_view: Extracts the D3 parameters.
        gradient_lens: Selects the differentiation degrees of freedom.
        hessian_lens: Selects which gradients get Hessians.
        hessian_idx_view: Hessian row/column indices.
        patch_idx_view: Index structure for cache updates.
        out_cache_lens: Where to cache the output, if anywhere.

    Returns:
        A configured D3 potential.
    """
    graph_constructor = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=neighborlist_view,
        probe=None,
    )
    potential = PotentialFromEnergy(
        energy_fn=d3_energy,
        composer=FullGraphSumComposer(
            graph_constructor=graph_constructor,
            parameter_view=parameter_view,
        ),
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
    return _ValidatedD3Potential(potential, particles_view)


if TYPE_CHECKING:
    _d3: EnergyFunction[Any, D3Input] = d3_energy
