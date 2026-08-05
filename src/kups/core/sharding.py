# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Domain-decomposition mesh helpers: the axis-name convention and replicated placement."""

import jax

from kups.core.data.index import SupportsSorting
from kups.core.utils.jax import no_post_init


def shard_axis[Id: SupportsSorting](id: type[Id]) -> str:
    """Mesh axis name for sharding over an Id type (e.g. ``OriginDeviceId``)."""
    return id.__name__


def device_put_replicated[S](state: S, mesh: jax.sharding.Mesh) -> S:
    """Place every leaf of ``state`` on ``mesh`` fully replicated.

    ``jax.device_put`` round-trips its pytree argument through a sentinel
    unflatten (``flatten_axes`` for ``may_alias``, on jax<=0.7.x), which would
    re-run Table/Index post-init validation on non-array sentinel leaves —
    disabled around the placement (a pure data movement).
    """
    sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with no_post_init():
        return jax.device_put(state, sharding)
