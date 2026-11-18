# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unified structure-relaxation entry point.

The force field is selected by the ``potential`` config (LJ / tojax-MLFF /
torch-MLFF). The potential is built directly with its parameters, so a single
force-field-agnostic ``StreamingRelaxState`` and driver serve every backend.
The input structures stream through ``run.batch_size`` slots.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # for torch backends

import jax
import rich
import rich.logging
from nanoargs import NanoArgs
from pydantic import BaseModel

from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_ONLY
from kups.application.relaxation.analysis import analyze_streaming_relax_file
from kups.application.relaxation.data import RelaxRunConfig
from kups.application.relaxation.streaming import (
    StreamingRelaxState,
    Structure,
    run_streaming_relax,
)
from kups.application.simulations.potentials import (
    MaceConfig,
    PotentialConfig,
    UmaConfig,
)
from kups.application.utils.particles import particles_from_ase
from kups.core.lens import identity_lens
from kups.relaxation.config import make_optimizer
from kups.relaxation.optimizer import ChainOptState

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich.logging.RichHandler()],
)


class Config(BaseModel):
    """Top-level configuration for a structure relaxation run."""

    run: RelaxRunConfig
    potential: PotentialConfig
    inp_files: tuple[str | Path, ...]


def run(config: Config) -> None:
    """Run a structure relaxation for the configured force field."""
    key = jax.random.key(config.run.seed or time.time_ns())
    state_lens = identity_lens(StreamingRelaxState[ChainOptState])
    optimizer = make_optimizer(config.run.optimizer)
    gradient = FRECHET_FILTER if config.run.optimize_cell else POSITIONS_ONLY
    potential, cutoff = config.potential.build(state_lens, gradient)

    structures: list[Structure] = []
    for inp_file in config.inp_files:
        logging.info(f"Loading structure from {inp_file}")
        particles, cell, _ = particles_from_ase(inp_file)
        structures.append((particles, cell))

    # Torch MLFF models need extra neighbor-list capacity headroom.
    multiplier = 2.0 if isinstance(config.potential, MaceConfig | UmaConfig) else 1.0
    logging.info("Starting relaxation")
    run_streaming_relax(
        key,
        structures,
        potential,
        cutoff,
        optimizer,
        gradient,
        config.run,
        neighborlist_multiplier=multiplier,
    )


def main() -> None:
    """CLI entry point for structure relaxation."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config.model_dump())
    run(config)
    rich.print(analyze_streaming_relax_file(config.run.out_file))


if __name__ == "__main__":
    main()
