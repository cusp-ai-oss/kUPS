# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Factory utilities for building chained relaxation optimizers from config specs.

Lives in its own module (rather than ``kups.relaxation.optimizer``) so the
custom-transform registry can refer to entries from
``kups.relaxation.transforms`` without forming a circular import — the
transform implementations need ``Optimizer`` from
``kups.relaxation.optimizer`` themselves.
"""

from typing import Any

import optax

from kups.relaxation.optimizer import Optimizer, chain
from kups.relaxation.transforms.clip_by_global_norm import ClipByGlobalNorm
from kups.relaxation.transforms.fire import ScaleByFire
from kups.relaxation.transforms.fire2 import ScaleByFire2
from kups.relaxation.transforms.lbfgs import ScaleByAseLbfgs
from kups.relaxation.transforms.linesearch import (
    ScaleByBacktrackingLinesearch,
    ScaleByMoreThuenteLinesearch,
)
from kups.relaxation.transforms.max_step_size import MaxStepSize

Transform = str | dict[str, bool | int | float | str | list[Any] | None]
"""A single transform spec: either a name string or a dict with ``"transform"`` key."""

TransformationConfig = list[Transform]
"""Ordered list of transform specs to chain into an optimizer."""

_CUSTOM_TRANSFORMS: dict[str, Any] = {
    "scale_by_fire": ScaleByFire,
    "scale_by_fire2": ScaleByFire2,
    "max_step_size": MaxStepSize,
    "scale_by_ase_lbfgs": ScaleByAseLbfgs,
    "clip_by_global_norm": ClipByGlobalNorm,
    "scale_by_backtracking_linesearch": ScaleByBacktrackingLinesearch,
    "scale_by_more_thuente_linesearch": ScaleByMoreThuenteLinesearch,
}

_UNSUPPORTED_OPTAX = {
    "lbfgs": "scale_by_ase_lbfgs + scale_by_more_thuente_linesearch",
    "scale_by_zoom_linesearch": "scale_by_more_thuente_linesearch",
}
"""optax value-based ops (their ``update`` needs ``value``/``value_fn``, which the
per-system propagator does not supply) mapped to the kups transform(s) to use."""


def get_transform(
    transform: Transform,
) -> optax.GradientTransformation | Optimizer[Any, Any]:
    """Convert a transform config entry to an Optax GradientTransformation.

    Args:
        transform: Either a plain string name (e.g. ``"scale_by_adam"``) or a
            dict with a ``"transform"`` key and additional keyword arguments.

    Returns:
        The constructed GradientTransformation.

    Raises:
        ValueError: If the transform name is unknown, or resolves to an optax
            value-based line search whose ``update`` needs ``value``/``value_fn``
            (which ``RelaxationPropagator`` does not supply per system).
    """
    if isinstance(transform, str):
        name = transform
        kwargs: dict[str, Any] = {}
    else:
        transform = transform.copy()
        name = str(transform.pop("transform"))
        kwargs = transform

    if name in _CUSTOM_TRANSFORMS:
        return _CUSTOM_TRANSFORMS[name](**kwargs)
    if name in _UNSUPPORTED_OPTAX:
        raise ValueError(
            f"Unsupported optax transform '{name}'; use "
            f"{_UNSUPPORTED_OPTAX[name]} instead."
        )
    if not hasattr(optax, name):
        raise ValueError(f"Unknown transformation: {name}")
    return getattr(optax, name)(**kwargs)


def get_transformations(
    transformations: TransformationConfig,
) -> list[optax.GradientTransformation | Optimizer[Any, Any]]:
    """Convert a list of transform configs to Optax GradientTransformations.

    Args:
        transformations: List of transform specifications.

    Returns:
        List of GradientTransformations in the same order.
    """
    return [get_transform(t) for t in transformations]


def make_optimizer(transformations: TransformationConfig) -> Optimizer[Any, Any]:
    """Create a chained optimizer from a list of transform configs.

    Args:
        transformations: List of transform specifications.

    Returns:
        Chained Optax GradientTransformation.

    Example:
        >>> config = [
        ...     {"transform": "clip_by_global_norm", "max_norm": 1.0},
        ...     {"transform": "scale_by_fire", "dt_start": 0.1},
        ... ]
        >>> optimizer = make_optimizer(config)
    """
    return chain(*get_transformations(transformations))
