# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Behavioral pins for the `Decomposition` placement strategy.

`Replicated` is the whole-graph null object: `owned_only` and
`combine_across_shards` must both be the identity. The domain-decomposed
implementation (`Sharded`) is pinned by its own tests next to
`kups.core.domain`.
"""

import jax
import jax.numpy as jnp
import numpy as np

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.common.graph import PointCloud, Replicated


@dataclass
class _P:
    positions: jax.Array
    system: Index[SystemId]


@dataclass
class _S:
    cell: PeriodicCell


def _particles(n: int, system_ids: np.ndarray | None = None) -> Table[ParticleId, _P]:
    if system_ids is None:
        system_ids = np.zeros(n, dtype=int)
    n_sys = int(system_ids.max()) + 1
    return Table.arange(
        _P(
            positions=jnp.zeros((n, 3)),
            system=Index.integer(system_ids, n=n_sys, label=SystemId),
        ),
        label=ParticleId,
    )


def test_replicated_owned_only_and_combine_are_identity() -> None:
    parts = _particles(4)
    x = jnp.arange(4.0)
    d = Replicated[_P]()
    assert jnp.array_equal(d.owned_only(parts, x), x)
    assert jnp.array_equal(d.combine_across_shards(x), x)


def test_replicated_reduce_matches_direct_segment_sum() -> None:
    # reduce_nodes_to_systems under the default Replicated placement must be a
    # plain per-system segment sum.
    system_ids = np.array([0, 0, 1, 1, 1, 0])
    parts = _particles(len(system_ids), system_ids)
    cell = PeriodicCell(TriclinicFrame.from_matrix(4.0 * jnp.eye(3)[None].repeat(2, 0)))
    systems = Table((SystemId(0), SystemId(1)), _S(cell))
    x = jnp.arange(float(len(system_ids)))
    reduced = PointCloud(parts, systems).reduce_nodes_to_systems(x)
    expected = jax.ops.segment_sum(x, jnp.asarray(system_ids), 2)
    assert jnp.array_equal(reduced.data, expected)
