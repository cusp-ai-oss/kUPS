# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Annotated, Literal

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import optax
import rich
import rich.logging
from jax import Array
from nanoargs import NanoArgs
from pydantic import BaseModel, Field

from kups.application.relaxation.analysis import analyze_relax_file
from kups.application.relaxation.data import (
    RelaxParameters,
    RelaxParticles,
    RelaxRunConfig,
    RelaxSystems,
    relax_state_from_ase,
)
from kups.application.relaxation.simulation import (
    OptInit,
    make_relax_propagator,
    run_relax,
)
from kups.application.utils.path import get_model_path
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.mliap.torch import (
    TorchMliap,
    load_mace,
    load_uma,
    make_torch_mliap_from_state,
)
from kups.relaxation.config import make_optimizer

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich.logging.RichHandler()],
)


class MACEModelConfig(BaseModel):
    """Configuration for loading a PyTorch MACE checkpoint."""

    backend: Literal["mace"] = "mace"
    model_path: str | Path
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: Literal["float32", "float64"] = "float32"


class UMAModelConfig(BaseModel):
    """Configuration for loading a Meta FAIR Chemistry UMA checkpoint."""

    backend: Literal["uma"] = "uma"
    model_path: str | Path
    device: Literal["cpu", "cuda"] = "cuda"
    task_name: Literal["omat", "omol", "oc20", "odac", "omc"] = "omat"
    inference_settings: Literal["default", "turbo"] = "default"


type ModelConfig = Annotated[
    MACEModelConfig | UMAModelConfig, Field(discriminator="backend")
]


class Config(BaseModel):
    """Top-level configuration for torch-MLFF relaxation runs."""

    run: RelaxRunConfig
    relax: RelaxParameters
    inp_files: tuple[str | Path, ...]
    model: ModelConfig


@dataclass
class RelaxTorchState:
    """Simulation state for torch-backed MLFF relaxation."""

    particles: Table[ParticleId, RelaxParticles]
    systems: Table[SystemId, RelaxSystems]
    neighborlist_params: UniversalNeighborlistParameters
    opt_state: optax.OptState
    step: Array
    torch_mliap_model: TorchMliap


def _load_torch_model(config: MACEModelConfig | UMAModelConfig) -> TorchMliap:
    """Dispatch to the right backend loader, with ``compute_cell_gradients=True``
    so stress is available regardless of whether the optimiser decides to
    update the cell.
    """
    model_path = get_model_path(config.model_path)
    if isinstance(config, MACEModelConfig):
        return load_mace(
            model_path,
            device=config.device,
            dtype=config.dtype,
            compute_cell_gradients=True,
        )
    return load_uma(
        model_path,
        device=config.device,
        task_name=config.task_name,
        compute_cell_gradients=True,
        inference_settings=config.inference_settings,
    )


def init_state(config: Config, opt_init: OptInit) -> RelaxTorchState:
    """Initialise relaxation state from config."""
    torch_mliap_model = _load_torch_model(config.model)
    all_particles: list[Table[ParticleId, RelaxParticles]] = []
    all_systems: list[Table[SystemId, RelaxSystems]] = []
    for inp_file in config.inp_files:
        logging.info(f"Loading structure from {inp_file}")
        particles_i, systems_i = relax_state_from_ase(inp_file)
        all_particles.append(particles_i)
        all_systems.append(systems_i)
    particles, systems = Table.union(all_particles, all_systems)
    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, torch_mliap_model.cutoff
    )
    opt_state = opt_init(particles, systems)
    return RelaxTorchState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        opt_state=opt_state,
        step=jnp.array([0]),
        torch_mliap_model=torch_mliap_model,
    )


def run(config: Config) -> None:
    """Run a torch-MLFF relaxation."""
    key = jax.random.key(config.run.seed or time.time_ns())
    state_lens = identity_lens(RelaxTorchState)
    optimizer = make_optimizer(config.relax.optimizer)
    potential = make_torch_mliap_from_state(
        state_lens, compute_position_and_cell_gradients=True
    )
    propagator, opt_init = make_relax_propagator(
        state_lens, potential, optimizer, config.relax.optimize_cell
    )
    state = init_state(config, opt_init)
    logging.info("Starting relaxation")
    run_relax(key, propagator, state, config.run)


def main() -> None:
    """CLI entry point for torch-MLFF relaxation."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_relax_file(config.run.out_file))


if __name__ == "__main__":
    main()
