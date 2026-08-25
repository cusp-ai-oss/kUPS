# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Domain-decomposition mesh helpers: the axis-name convention and replicated placement."""

import jax

from kups.core.data.index import SupportsSorting


def shard_axis[Id: SupportsSorting](id: type[Id]) -> str:
    """Mesh axis name for sharding over an Id type (e.g. ``OriginDeviceId``)."""
    return id.__name__


def device_put_replicated[S](state: S, mesh: jax.sharding.Mesh) -> S:
    """Place every leaf of ``state`` on ``mesh`` fully replicated."""
    sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return jax.device_put(state, sharding)
