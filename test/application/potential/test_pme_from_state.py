# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""``make_pme_from_state`` against ``make_pme_potential``."""

import jax
import numpy as np
import pytest

from kups.application.potential.classical.pme import make_pme_from_state
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.core.data import Table
from kups.core.lens import identity_lens, view
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.potential import EMPTY_LENS
from kups.core.utils.jax import dataclass
from kups.potential.classical.ewald import PMEParameters, make_pme_potential
from kups.potential.common.graph import POINTCLOUD_GEOMETRY
from test.potential.test_pme import _random_batch


@dataclass
class _State:
    particles: Table
    systems: Table
    neighborlist_params: UniversalNeighborlistParameters
    pme_parameters: PMEParameters


def _state() -> _State:
    particles, systems = _random_batch(5, (40, 30))
    cutoffs = systems.map_data(lambda s: s.cutoff)
    capacities = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, cutoffs
    )
    pme = PMEParameters.make(particles, systems, 1e-6, real_cutoff=6.0)
    return _State(particles, systems, capacities, pme)


@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize("on_state", [False, True])
def test_matches_make_pme_potential(on_state, gradient):
    state = _state()
    lens = identity_lens(_State)
    filter = POSITIONS_AND_CELL if gradient else None
    if on_state:
        potential = make_pme_from_state(lens, gradient=filter)
    else:
        potential = make_pme_from_state(
            lens, parameters=state.pme_parameters, gradient=filter
        )
    reference = make_pme_potential(
        view(lambda s: s.particles),
        view(lambda s: s.systems),
        lambda s: AdaptiveNeighborList.from_state(s, lens, s.pme_parameters.cutoff),
        view(lambda s: s.pme_parameters),
        POINTCLOUD_GEOMETRY.nest(POSITIONS_AND_CELL) if gradient else EMPTY_LENS,
        EMPTY_LENS,
        EMPTY_LENS,
    )
    out, expected = jax.jit(lambda s: (potential(s).data, reference(s).data))(state)
    assert np.all(np.abs(np.asarray(expected.total_energies.data)) > 1e-3)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
