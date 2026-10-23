# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Relaxation on the skin list sees the same pairs as with per-step builds."""

from typing import Any

import ase.build
import jax
import jax.numpy as jnp
import numpy as np

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_ONLY
from kups.application.relaxation.data import RelaxState, relax_state_from_ase
from kups.application.relaxation.simulation import make_relax_propagator
from kups.core.assertion import check_assertions, runtime_assert
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import AdaptiveNeighborList, SkinNeighborList
from kups.core.potential import Potential
from kups.core.propagator import propagate_and_fix
from kups.core.result import as_result_function
from kups.core.utils.jax import dataclass, field
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.relaxation.config import make_optimizer

_LJ = LennardJonesParameters.from_dict(
    cutoff=5.0, parameters={"Ar": (3.405, 0.010326)}, mixing_rule="lorentz_berthelot"
)
_LBFGS = [
    {"transform": "scale_by_ase_lbfgs", "memory_size": 10, "alpha": 70},
    {"transform": "max_step_size", "max_step_size": 0.2},
    {"transform": "scale", "step_size": -1},
]
# trial points of the line search leave the skin as well
_LBFGS_LINESEARCH = [
    {"transform": "scale_by_ase_lbfgs", "memory_size": 10},
    {"transform": "scale", "step_size": -1},
    {"transform": "scale_by_more_thuente_linesearch"},
]


@dataclass
class _Checked:
    """``potential``, asserting at every evaluation that it equals ``reference``."""

    potential: Potential[RelaxState, Any, Any, Any] = field(static=True)
    reference: Potential[RelaxState, Any, Any, Any] = field(static=True)

    def __call__(self, state, patch=None, *, include_compensate=False) -> Any:
        out = self.potential(state, patch, include_compensate=include_compensate)
        expected = self.reference(state, patch, include_compensate=include_compensate)
        leaves = zip(
            jax.tree.leaves(out.data), jax.tree.leaves(expected.data), strict=True
        )
        close = jnp.all(
            jnp.asarray([jnp.allclose(a, b, rtol=1e-10, atol=1e-10) for a, b in leaves])
        )
        # where a build outgrew its capacities, the step is fixed and run again
        runtime_assert(close | ~check_assertions(), "Differs from per-step builds.")
        return out


def _built(factory, gradient) -> tuple:
    """Graph LJ over ``factory``'s pairs."""
    potential = make_lennard_jones_from_state(
        identity_lens(RelaxState),
        parameters=_LJ,
        gradient=gradient,
        neighborlist_factory=factory,
    )
    return potential, _LJ.cutoff


def _relax(
    potential, cutoff, optimizer, gradient, reads_the_list: bool, steps: int = 12
) -> RelaxState:
    # the 2 x 2 x 2 primitive cell is narrower than twice the cutoff
    atoms = [
        ase.build.bulk("Ar", "fcc", a=5.0, cubic=True) * (3, 3, 3),
        ase.build.bulk("Ar", "fcc", a=5.0) * (2, 2, 2),
    ]
    for seed, a in enumerate(atoms):
        a.rattle(0.3, seed=seed)
    tables = [relax_state_from_ase(a) for a in atoms]
    particles, systems = Table.union([p for p, _ in tables], [s for _, s in tables])
    propagator, opt_init = make_relax_propagator(
        identity_lens(RelaxState), potential, make_optimizer(optimizer), gradient
    )
    opt_state = opt_init(particles, systems)
    state = RelaxState.create(particles, systems, cutoff, 0.3, opt_state)
    built = state.skin_list.reference.positions
    step = jax.jit(as_result_function(propagator))
    for i in range(steps):
        state = propagate_and_fix(step, jax.random.key(i), state)
    rebuilt = not np.array_equal(built, state.skin_list.reference.positions)
    assert rebuilt == reads_the_list
    return state


def test_relaxation_matches_per_step_builds():
    """Positions follow per-step builds; only the run that reads the list rebuilds
    it."""
    skin, per_step = (
        _relax(
            *_built(factory, POSITIONS_ONLY),
            _LBFGS,
            POSITIONS_ONLY,
            reads_the_list=reads_the_list,
        )
        for factory, reads_the_list in (
            (SkinNeighborList.from_state, True),
            (AdaptiveNeighborList.from_state, False),
        )
    )
    np.testing.assert_allclose(
        skin.particles.data.positions, per_step.particles.data.positions, atol=1e-8
    )


def test_line_search_evaluations_match_per_step_builds():
    """A line search turns round-off into other trial points, so paths that sum pairs
    in another order diverge. Every evaluation, trial points included, is checked
    against per-step builds instead."""
    skin, cutoff = _built(SkinNeighborList.from_state, FRECHET_FILTER)
    per_step, _ = _built(AdaptiveNeighborList.from_state, FRECHET_FILTER)
    _relax(
        _Checked(skin, per_step),
        cutoff,
        _LBFGS_LINESEARCH,
        FRECHET_FILTER,
        reads_the_list=True,
    )
