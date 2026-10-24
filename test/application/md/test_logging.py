# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""kups_md writes energies and frames at their own intervals."""

import logging
import sys
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from kups.application.md.data import MdRunConfig, MdState
from kups.application.md.logging import MDLoggedData
from kups.application.simulations.md import Config, main, run
from kups.core.storage import HDF5StorageReader

from .._builders import ROWS, argon, md_parameters, write_cif


def _run(directory: Path, **intervals: int) -> tuple[MdState, Path]:
    """24 CSVR steps in blocks of 2 for two argon systems, from new input files."""
    directory.mkdir()
    out_file = directory / "md.h5"
    config = Config(
        run=MdRunConfig(
            out_file=out_file,
            num_steps=24,
            num_warmup_steps=0,
            block_size=2,
            seed=1,
            **intervals,
        ),
        md=md_parameters("csvr"),
        potential=ROWS,
        inp_files=tuple(write_cif(argon(n, n), directory / f"{n}.cif") for n in (2, 3)),
    )
    return run(config), out_file


def _assert_tree_equal(actual: object, expected: object) -> None:
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(x, y)


def test_intervals_subsample_one_trajectory(tmp_path: Path):
    """Rows every 4 and frames every 6 steps are every second row and every third
    frame of the same run logged every block, and both end on its final state."""
    state, every_block = _run(tmp_path / "every_block")
    _, sparse = _run(tmp_path / "sparse", thermo_every=4, frames_every=6)
    groups = []
    for path in (every_block, sparse):
        with HDF5StorageReader[MDLoggedData](path) as reader:
            groups.append(
                (
                    reader.focus_group(lambda s: s.step)[:],
                    reader.focus_group(lambda s: s.frames)[:],
                    reader.focus_group(lambda s: s.final)[...],
                )
            )
        with h5py.File(path) as file:
            assert file.attrs["actual_steps"] == file.attrs["total_steps"] == 12
    (rows, frames, final), (sparse_rows, sparse_frames, sparse_final) = groups
    np.testing.assert_array_equal(rows.step[:, 0], np.arange(2, 25, 2))
    np.testing.assert_array_equal(sparse_rows.step[:, 0], np.arange(4, 25, 4))
    np.testing.assert_array_equal(sparse_frames.step[:, 0], np.arange(6, 25, 6))
    _assert_tree_equal(sparse_rows, jax.tree.map(lambda x: x[1::2], rows))
    _assert_tree_equal(sparse_frames, jax.tree.map(lambda x: x[2::3], frames))
    _assert_tree_equal(sparse_final, final)
    _assert_tree_equal(final, (state.particles, state.systems))


def test_intervals_are_whole_blocks():
    """Intervals default to the block size; they and the production length must be
    multiples of it."""
    config = MdRunConfig(
        out_file="md.h5", num_steps=8, num_warmup_steps=0, block_size=4, seed=0
    )
    assert (config.thermo_every, config.frames_every) == (4, 4)
    with pytest.raises(ValidationError, match=r"run.frames_every \(6 steps\)"):
        MdRunConfig(
            out_file="md.h5",
            num_steps=8,
            num_warmup_steps=0,
            block_size=4,
            frames_every=6,
            seed=0,
        )
    with pytest.raises(ValidationError, match=r"run.num_steps \(6 steps\)"):
        MdRunConfig(
            out_file="md.h5", num_steps=6, num_warmup_steps=0, block_size=4, seed=0
        )


def test_short_runs_are_not_analyzed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """kups_md finishes a run with fewer rows than block averaging needs, unanalyzed."""
    config = {
        "run": {
            "out_file": str(tmp_path / "md.h5"),
            "num_steps": 4,
            "num_warmup_steps": 0,
            "block_size": 2,
            "seed": 0,
        },
        "md": md_parameters("verlet").model_dump(mode="json"),
        "potential": ROWS.model_dump(mode="json"),
        "inp_files": [str(write_cif(argon(2, 0), tmp_path / "argon.cif"))],
    }
    (tmp_path / "md.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["kups_md", str(tmp_path / "md.yaml")])
    with caplog.at_level(logging.WARNING):
        main()
    assert "Not analyzed: 2 rows" in caplog.text
