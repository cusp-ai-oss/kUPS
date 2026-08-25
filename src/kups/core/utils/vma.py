# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Varying-manual-axes (vma) plumbing for kUPS' custom primitives.

Under ``shard_map(..., check_vma=True)`` every abstract value tracks which
manual mesh axes it varies over; an output aval that dropped them would type
the value as replicated, and the transpose's cotangent would then mismatch a
varying primal. jax has no public registry for custom primitives' vma rules
(jax-ml/jax#24726 tracks a public primitive API), so the segment sum/take
primitives route their vma handling through this module, the single place
that touches the jax-internal helpers.

Version knowledge lives here and nowhere else: ``aval.vma`` (jax 0.7) became
``aval.manual_axis_type.varying`` (jax 0.8+), and ``standard_insert_pvary``
became ``auto_insert_reshard`` (jax 0.11). A future rename fails loudly on
the attribute lookup below.
"""

from typing import Any

from jax._src import core as jax_core
from jax.core import ShapedArray

insert_pvary: Any = getattr(
    jax_core,
    "standard_insert_pvary",  # jax <= 0.10
    getattr(jax_core, "auto_insert_reshard", None),  # jax >= 0.11
)
"""Broadcast operands to the union of their varying manual axes.

Bind wrappers call this on their operands so the standard vma rule sees
consistent inputs; its transpose is the matching mesh reduction, which keeps
cotangent types aligned with the original operands. A no-op outside
``shard_map`` or with ``check_vma=False``.
"""
assert insert_pvary is not None, "jax exposes no vma operand-unifying helper"


def out_aval(
    name: str, shape: tuple[Any, ...], data: ShapedArray, *operands: Any
) -> ShapedArray:
    """Output aval of ``shape`` and ``data``'s dtype carrying the operands' vma.

    The vma set is the standard unification of ``data`` and ``operands``
    (empty outside a manual region, where a plain aval is returned). The
    varying case derives the aval from ``data`` via ``update`` so its (manual)
    sharding carries over — valid only rank-preservingly, which the callers
    guarantee and the assert pins.
    """
    vma = jax_core.standard_vma_rule(name, data, *operands)
    if not vma:
        return ShapedArray(shape, data.dtype)
    assert len(shape) == data.ndim, (shape, data.shape)
    mat = getattr(data, "manual_axis_type", None)
    if mat is None:  # jax 0.7
        return data.update(shape=shape, weak_type=False, vma=vma)
    return data.update(
        shape=shape, weak_type=False, manual_axis_type=mat.update(varying=vma)
    )


__all__ = ["insert_pvary", "out_aval"]
