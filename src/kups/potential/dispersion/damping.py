# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Becke-Johnson (rational) damping for the D3 dispersion correction.

The damped pair interaction is

$$
E_{ij} = -\left[
    \frac{s_6\,C^{ij}_6}{r_{ij}^6 + (R^{ij}_0)^6}
  + \frac{s_8\,C^{ij}_8}{r_{ij}^8 + (R^{ij}_0)^8}
\right],
\qquad
R^{ij}_0 = a_1 \sqrt{3\,Q_i Q_j} + a_2 ,
$$

with $C^{ij}_8 = 3\,Q_i Q_j\,C^{ij}_6$ and $Q_i$ the stored ``r4r2`` quantity.
Unlike zero damping, the rational form stays finite as $r \to 0$.

The four parameters are fitted per density functional. Values in
[BECKE_JOHNSON_PARAMETERS][kups.potential.dispersion.damping.BECKE_JOHNSON_PARAMETERS]
are quoted exactly as published, so $a_2$ is in **Bohr**; conversion to kUPS units
happens in ``D3Parameters``.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "BECKE_JOHNSON_PARAMETERS",
    "BeckeJohnsonDamping",
    "available_functionals",
    "damping_for_functional",
    "normalize_functional",
]


class BeckeJohnsonDamping(NamedTuple):
    """D3(BJ) damping parameters as published.

    Attributes:
        s6: Scaling of the ``C6`` term. ``1.0`` for every functional except
            double hybrids, which scale it down.
        s8: Scaling of the ``C8`` term.
        a1: Dimensionless damping parameter.
        a2: Damping parameter with units of length, **in Bohr** (as published).
        doi: Source of the fit, or ``""`` where the upstream parameter set names
            none and no primary source has been confirmed. An empty string is a
            statement that the attribution is unknown, not that it is missing by
            oversight -- do not fill it in without checking the paper.
    """

    s6: float
    s8: float
    a1: float
    a2: float
    doi: str


# Fitted values as published in the primary literature cited by each entry.
# Reproduced with attribution; see ``data/README.md`` for the licensing note.
BECKE_JOHNSON_PARAMETERS: dict[str, BeckeJohnsonDamping] = {
    "b3lyp": BeckeJohnsonDamping(1.0, 1.9889, 0.3981, 4.4211, "10.1002/jcc.21759"),
    "b97d": BeckeJohnsonDamping(1.0, 2.2609, 0.5545, 3.2297, "10.1002/jcc.21759"),
    "blyp": BeckeJohnsonDamping(1.0, 2.6996, 0.4298, 4.2359, "10.1002/jcc.21759"),
    "bp": BeckeJohnsonDamping(1.0, 3.2822, 0.3946, 4.8516, "10.1002/jcc.21759"),
    "b2plyp": BeckeJohnsonDamping(0.64, 0.9147, 0.3065, 5.0570, "10.1039/c0cp02984j"),
    "hf": BeckeJohnsonDamping(1.0, 0.9171, 0.3385, 2.8830, "10.1002/jcc.21759"),
    "pbe": BeckeJohnsonDamping(1.0, 0.7875, 0.4289, 4.4407, "10.1002/jcc.21759"),
    "pbe0": BeckeJohnsonDamping(1.0, 1.2177, 0.4145, 4.8593, "10.1002/jcc.21759"),
    "pbesol": BeckeJohnsonDamping(1.0, 2.9491, 0.4466, 6.1742, "10.1039/c0cp02984j"),
    "pw6b95": BeckeJohnsonDamping(1.0, 0.7257, 0.2076, 6.3750, "10.1039/c0cp02984j"),
    "revpbe": BeckeJohnsonDamping(1.0, 2.3550, 0.5238, 3.5016, "10.1002/jcc.21759"),
    "revtpss": BeckeJohnsonDamping(1.0, 1.4023, 0.4426, 4.4723, "10.1039/c7cp04913g"),
    # values match the upstream set exactly; unlike its neighbours that entry
    # carries no doi, and the fit has not been traced to a paper -- see `doi`
    "rpbe": BeckeJohnsonDamping(1.0, 0.8318, 0.1820, 4.0094, ""),
    "rpw86pbe": BeckeJohnsonDamping(1.0, 1.3845, 0.4613, 4.5062, "10.1002/jcc.21759"),
    "r2scan": BeckeJohnsonDamping(
        1.0, 0.78981345, 0.49484001, 5.73083694, "10.1063/5.0041008"
    ),
    "scan": BeckeJohnsonDamping(
        1.0, 0.0000, 0.5380, 5.4200, "10.1103/physrevb.94.115144"
    ),
    "tpss": BeckeJohnsonDamping(1.0, 1.9435, 0.4535, 4.4752, "10.1002/jcc.21759"),
    "tpssh": BeckeJohnsonDamping(1.0, 2.2382, 0.4529, 4.6550, "10.1039/c0cp02984j"),
}
"""D3(BJ) damping parameters by functional. ``a2`` is in Bohr."""


def normalize_functional(functional: str) -> str:
    """Fold a functional name to the key convention used here.

    Case, hyphens, underscores and spaces are ignored, so ``"PBE0"``,
    ``"pbe-0"`` and ``"pbe 0"`` all resolve to ``"pbe0"``.
    """
    return functional.lower().replace("-", "").replace("_", "").replace(" ", "")


def available_functionals() -> tuple[str, ...]:
    """Functionals with tabulated D3(BJ) parameters, sorted."""
    return tuple(sorted(BECKE_JOHNSON_PARAMETERS))


def damping_for_functional(functional: str) -> BeckeJohnsonDamping:
    """Look up published D3(BJ) parameters for a density functional.

    Args:
        functional: Functional name; matched case- and punctuation-insensitively.

    Returns:
        The published parameters, with ``a2`` in Bohr.

    Raises:
        KeyError: If the functional has no tabulated parameters. Pass ``s6``,
            ``s8``, ``a1`` and ``a2`` explicitly in that case.
    """
    key = normalize_functional(functional)
    try:
        return BECKE_JOHNSON_PARAMETERS[key]
    except KeyError:
        raise KeyError(
            f"No tabulated D3(BJ) parameters for {functional!r}. "
            f"Available: {', '.join(available_functionals())}. "
            "Pass s6/s8/a1/a2 explicitly to use another parameterisation."
        ) from None
