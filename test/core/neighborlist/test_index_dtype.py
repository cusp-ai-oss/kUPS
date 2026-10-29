# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Neighbor-list builds keep their index arithmetic in the dtype of the keys' indices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import ClosedJaxpr, Jaxpr

from kups.core.capacity import FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.neighborlist import (
    AllDenseNearestNeighborList,
    CellListNeighborList,
    DenseNearestNeighborList,
)
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId

from ._builders import SamplePoints, SampleSystems, valid_edge_set

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"), reason="int32 is already the default"
)

HEAVY = {"sort", "scatter", "scatter-add", "cumsum", "cumprod"}


def _inputs(n_sys: int = 2, n_per: int = 64, dtype=None):
    rng = np.random.default_rng(0)
    lattice = np.array([[12.0, 0, 0], [2.0, 11.0, 0], [1.0, 1.5, 10.0]])
    n = n_sys * n_per
    system_ids = np.repeat(np.arange(n_sys), n_per)
    points = Table(
        tuple(ParticleId(i) for i in range(n)),
        SamplePoints(
            positions=jnp.asarray(rng.uniform(0, 1, (n, 3)) @ lattice),
            system=Index.integer(system_ids, label=SystemId, dtype=dtype),
            inclusion=Index.integer(system_ids, label=SystemId, dtype=dtype),
            exclusion=Index.integer(np.arange(n), dtype=dtype),
        ),
    )
    keys = tuple(SystemId(i) for i in range(n_sys))
    frame = TriclinicFrame.from_matrix(jnp.asarray(np.repeat(lattice[None], n_sys, 0)))
    systems = Table(keys, SampleSystems(cell=PeriodicCell(frame)))
    return points, systems, Table(keys, jnp.full(n_sys, 4.0))


def _neighbor_lists(cutoffs):
    cap = FixedCapacity
    return {
        "cell_list": CellListNeighborList(
            avg_candidates=cap(200),
            avg_edges=cap(60),
            cells=cap(64),
            avg_image_candidates=cap(200),
            cutoffs=cutoffs,
        ),
        "dense": DenseNearestNeighborList(
            avg_candidates=cap(64),
            avg_edges=cap(60),
            avg_image_candidates=cap(256),
            cutoffs=cutoffs,
        ),
        "all_dense": AllDenseNearestNeighborList(
            avg_edges=cap(60), avg_image_candidates=cap(256), cutoffs=cutoffs
        ),
    }


def _int64_heavy_ops(jaxpr: Jaxpr) -> list[str]:
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in HEAVY and any(
            getattr(v.aval, "dtype", None) == jnp.int64 for v in eqn.outvars
        ):
            found.append(eqn.primitive.name)
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else [param]:
                if isinstance(sub, ClosedJaxpr):
                    found += _int64_heavy_ops(sub.jaxpr)
                elif isinstance(sub, Jaxpr):
                    found += _int64_heavy_ops(sub)
    return found


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", ["cell_list", "dense", "all_dense"])
def test_int32_build_matches_default(name):
    points, systems, cutoffs = _inputs()

    def build():
        nl = as_result_function(_neighbor_lists(cutoffs)[name])
        result = jax.jit(nl)(points, systems)
        result.raise_assertion()
        return result.value

    expected = build()
    points, systems, cutoffs = _inputs(dtype=jnp.int32)
    edges = build()
    assert edges.indices.raw_indices.dtype == jnp.int32
    assert valid_edge_set(edges) == valid_edge_set(expected)

    nl = as_result_function(_neighbor_lists(cutoffs)[name])
    jaxpr = jax.make_jaxpr(nl)(points, systems)
    assert _int64_heavy_ops(jaxpr.jaxpr) == []
