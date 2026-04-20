# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Factory utilities for building Optax optimizers from config specs."""

from typing import Any

import optax

from kups.relaxation.optax.fire import scale_by_fire
from kups.relaxation.optax.lbfgs import scale_by_ase_lbfgs
from kups.relaxation.optax.max_step_size import max_step_size

Transform = str | dict[str, bool | int | float | str | list | None]
"""A single transform spec: either a name string or a dict with ``"transform"`` key."""

TransformationConfig = list[Transform]
"""Ordered list of transform specs to chain into an optimizer."""

_CUSTOM_TRANSFORMS: dict[str, Any] = {
    "scale_by_fire": scale_by_fire,
    "max_step_size": max_step_size,
    "scale_by_ase_lbfgs": scale_by_ase_lbfgs,
}


def get_transform(transform: Transform) -> optax.GradientTransformation:
    """Convert a transform config entry to an Optax GradientTransformation.

    Args:
        transform: Either a plain string name (e.g. ``"scale_by_adam"``) or a
            dict with a ``"transform"`` key and additional keyword arguments.

    Returns:
        The constructed GradientTransformation.

    Raises:
        ValueError: If the transform name is not found in custom transforms or optax.
    """
    if isinstance(transform, str):
        name = transform
        kwargs: dict[str, Any] = {}
    else:
        transform = transform.copy()
        name = str(transform.pop("transform"))
        kwargs = transform

    if name in _CUSTOM_TRANSFORMS:
        constructor = _CUSTOM_TRANSFORMS[name]
    elif hasattr(optax, name):
        constructor = getattr(optax, name)
    else:
        raise ValueError(f"Unknown transformation: {name}")

    return constructor(**kwargs)


def get_transformations(
    transformations: TransformationConfig,
) -> list[optax.GradientTransformation]:
    """Convert a list of transform configs to Optax GradientTransformations.

    Args:
        transformations: List of transform specifications.

    Returns:
        List of GradientTransformations in the same order.
    """
    return [get_transform(t) for t in transformations]


def make_optimizer(
    transformations: TransformationConfig,
) -> optax.GradientTransformationExtraArgs:
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
    return optax.chain(*get_transformations(transformations))
