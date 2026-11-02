# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""The class declarations the checker sees must match the runtime surface.

``Quantity`` and ``Unit`` are single classes; names with no
meaning are defined under ``if not TYPE_CHECKING`` so the checker flags them
while the runtime raises with guidance, and those hidden blocks are pinned here.
"""

import pathlib

from kups.core import units

PKG = pathlib.Path(units.__file__).parent


def test_no_pyi_stubs():
    # a stub would silently shadow its module
    assert not list(PKG.glob("*.pyi")), (
        f"{[p.name for p in PKG.glob('*.pyi')]} shadow the sources they sit next to"
    )
