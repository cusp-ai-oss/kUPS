# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tojax simulation config in kups.application.simulations.potentials."""

import json
import zipfile
from typing import Any

import jax
import numpy as np
import pytest
from jax import export
from pydantic import ValidationError

from kups.application.simulations import potentials
from kups.application.simulations.potentials import TojaxPotentialConfig
from kups.core.utils.msgpack import serialize as msgpack_serialize


def test_tojax_config_task_defaults_to_none():
    assert TojaxPotentialConfig(model_path="model.zip").task_name is None


def test_tojax_config_rejects_unknown_task():
    with pytest.raises(ValidationError, match="task_name"):
        TojaxPotentialConfig.model_validate(
            {"model_path": "model.zip", "task_name": "omoll"}
        )


def test_tojax_config_forwards_task(tmp_path, monkeypatch):
    params = [np.array(1.0)]
    exported = export.export(jax.jit(lambda p, x: x * p[0]))(params, np.zeros(3))
    model_path = tmp_path / "model.zip"
    with zipfile.ZipFile(model_path, "w") as zf:
        zf.writestr("model.jax", exported.serialize())
        zf.writestr("metadata.json", json.dumps({"cutoff": 5.0}))
        zf.writestr("params.msgpack", msgpack_serialize(params))
    seen: dict[str, Any] = {}

    def make_tojaxed_from_state(state_lens, **kwargs):
        seen.update(kwargs)
        return "potential"

    monkeypatch.setattr(potentials, "make_tojaxed_from_state", make_tojaxed_from_state)
    config = TojaxPotentialConfig.model_validate(
        {"model_path": str(model_path), "task_name": "omol"}
    )

    potential, _ = config.build(state_lens=None, gradient=None)  # type: ignore[arg-type]

    assert potential == "potential"
    assert seen["parameters"].task_name == "omol"
