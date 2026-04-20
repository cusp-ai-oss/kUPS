# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for HDF5 storage functionality."""

import tempfile

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.lens import view
from kups.core.storage import (
    EveryNStep,
    GroupReader,
    HDF5StorageReader,
    HDF5StorageWriter,
    Once,
    WriterGroupConfig,
)
from kups.core.utils.jax import dataclass

from ..clear_cache import clear_cache  # noqa: F401


@dataclass
class SimpleState:
    position: jax.Array
    velocity: jax.Array
    energy: float


@dataclass
class NestedState:
    system: "SystemData"
    metadata: "MetaData"


@dataclass
class SystemData:
    positions: jax.Array
    forces: jax.Array
    temperature: float


@dataclass
class MetaData:
    step: int
    time: float


@pytest.fixture
def simple_state():
    return SimpleState(
        position=jnp.array([1.0, 2.0, 3.0]),
        velocity=jnp.array([0.1, 0.2, 0.3]),
        energy=10.5,
    )


@pytest.fixture
def nested_state():
    return NestedState(
        system=SystemData(
            positions=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            forces=jnp.array([[0.1, 0.2], [0.3, 0.4]]),
            temperature=298.15,
        ),
        metadata=MetaData(step=100, time=0.1),
    )


@pytest.fixture
def temp_file():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp:
        temp_path = temp.name
    try:
        yield temp_path
    finally:
        import os

        if os.path.exists(temp_path):
            os.unlink(temp_path)


class TestLoggingFrequencies:
    def test_once(self):
        """Once: should_log, leading_shape, dataset_index."""
        freq = Once()
        assert freq.should_log(0)
        assert not freq.should_log(1)
        assert not freq.should_log(10)
        assert freq.leading_shape(100) == ()
        assert freq.dataset_index(0) is ...

    def test_every_n_step(self):
        """EveryNStep: should_log, leading_shape, dataset_index."""
        freq = EveryNStep(n=5)
        assert freq.should_log(0)
        assert not freq.should_log(1)
        assert freq.should_log(5)
        assert freq.should_log(10)
        assert not freq.should_log(7)

        freq10 = EveryNStep(n=10)
        assert freq10.leading_shape(100) == (10,)
        assert freq10.leading_shape(95) == (10,)
        assert freq10.leading_shape(101) == (11,)

        assert freq.dataset_index(0) == 0
        assert freq.dataset_index(5) == 1
        assert freq.dataset_index(10) == 2


class TestHDF5StorageWriter:
    def test_write_every_and_every_n_and_once(self, simple_state, temp_file):
        """Test writing every step, every N steps, and once in a single workflow."""
        config = {
            "every": WriterGroupConfig(
                view=view(lambda s: {"pos": s.position, "vel": s.velocity}),
                logging_frequency=EveryNStep(1),
            ),
            "sparse": WriterGroupConfig(
                view=view(lambda s: {"pos": s.position}),
                logging_frequency=EveryNStep(3),
            ),
            "initial": WriterGroupConfig(
                view=view(lambda s: {"pos": s.position}),
                logging_frequency=Once(),
            ),
        }
        writer = HDF5StorageWriter(temp_file, config, simple_state, total_steps=10)
        with writer:
            for i in range(10):
                state = SimpleState(
                    position=simple_state.position + i,
                    velocity=simple_state.velocity * (i + 1),
                    energy=simple_state.energy,
                )
                writer.log(state, i)

        with HDF5StorageReader(temp_file) as reader:
            # Every step
            every = reader.focus_group("group['every']")[:]
            assert every["pos"].shape == (10, 3)
            assert every["vel"].shape == (10, 3)
            npt.assert_array_equal(every["pos"][0], simple_state.position)
            npt.assert_array_equal(every["pos"][9], simple_state.position + 9)

            # Every 3 steps
            sparse = reader.focus_group("group['sparse']")[:]
            assert sparse["pos"].shape == (4, 3)
            npt.assert_array_equal(sparse["pos"][0], simple_state.position + 0)
            npt.assert_array_equal(sparse["pos"][1], simple_state.position + 3)

            # Once
            initial = reader.focus_group("group['initial']")[...]
            assert initial["pos"].shape == (3,)
            npt.assert_array_equal(initial["pos"], simple_state.position)

    def test_actual_steps_attr(self, simple_state, temp_file):
        writer = HDF5StorageWriter(
            temp_file,
            WriterGroupConfig(
                view=view(lambda s: {"pos": s.position}),
                logging_frequency=EveryNStep(1),
            ),
            simple_state,
            total_steps=10,
        )
        with writer:
            for i in range(5):
                writer.log(simple_state, i)

        import h5py

        with h5py.File(temp_file, "r") as f:
            assert f.attrs["actual_steps"] == 5


class TestHDF5StorageReader:
    def test_focus_group_and_list_groups(self, simple_state, temp_file):
        """Test focus_group returns GroupReader and list_groups works."""
        config = {
            "a": WriterGroupConfig(
                view=view(lambda s: {"pos": s.position}),
                logging_frequency=EveryNStep(1),
            ),
            "b": WriterGroupConfig(
                view=view(lambda s: {"vel": s.velocity}),
                logging_frequency=Once(),
            ),
        }
        writer = HDF5StorageWriter(temp_file, config, simple_state, total_steps=3)
        with writer:
            for i in range(3):
                writer.log(simple_state, i)

        with HDF5StorageReader(temp_file) as reader:
            group_reader = reader.focus_group("group['a']")
            assert isinstance(group_reader, GroupReader)

            groups = reader.list_groups()
            assert len(groups) == 2
            assert "group['a']" in groups
            assert "group['b']" in groups


class TestGroupReader:
    def test_indexing(self, simple_state, temp_file):
        """Test single index, negative index, and slice indexing."""
        writer = HDF5StorageWriter(
            temp_file,
            WriterGroupConfig(
                view=view(lambda s: {"pos": s.position}),
                logging_frequency=EveryNStep(1),
            ),
            simple_state,
            total_steps=10,
        )
        with writer:
            for i in range(10):
                state = SimpleState(
                    position=simple_state.position + i,
                    velocity=simple_state.velocity,
                    energy=simple_state.energy,
                )
                writer.log(state, i)

        with HDF5StorageReader(temp_file) as reader:
            group_reader = reader.focus_group("group")
            npt.assert_array_equal(group_reader[2]["pos"], simple_state.position + 2)
            npt.assert_array_equal(group_reader[-1]["pos"], simple_state.position + 9)
            assert group_reader[::2]["pos"].shape == (5, 3)


class TestIntegration:
    def test_complete_workflow_and_nested_state(
        self, simple_state, nested_state, temp_file
    ):
        """Test complete write/read workflow with simple and nested states."""
        # Simple state workflow
        config = {
            "trajectory": WriterGroupConfig(
                view=view(lambda s: {"pos": s.position, "vel": s.velocity}),
                logging_frequency=EveryNStep(2),
            ),
            "initial": WriterGroupConfig(
                view=view(lambda s: {"energy": jnp.array(s.energy)}),
                logging_frequency=Once(),
            ),
        }
        writer = HDF5StorageWriter(temp_file, config, simple_state, total_steps=10)
        with writer:
            for i in range(10):
                state = SimpleState(
                    position=simple_state.position + i * 0.1,
                    velocity=simple_state.velocity + i * 0.01,
                    energy=simple_state.energy + i,
                )
                writer.log(state, i)

        with HDF5StorageReader(temp_file) as reader:
            traj = reader.focus_group("group['trajectory']")[:]
            assert traj["pos"].shape == (5, 3)
            npt.assert_array_almost_equal(traj["pos"][0], simple_state.position)

            init = reader.focus_group("group['initial']")[...]
            assert init["energy"].shape == ()
            assert init["energy"] == simple_state.energy

        # Nested state workflow (separate temp file)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            nested_path = tmp.name

        try:
            nested_config = {
                "system": WriterGroupConfig(
                    view=view(lambda s: {"pos": s.system.positions}),
                    logging_frequency=EveryNStep(1),
                ),
                "metadata": WriterGroupConfig(
                    view=view(lambda s: {"temp": jnp.array(s.system.temperature)}),
                    logging_frequency=EveryNStep(5),
                ),
            }
            writer2 = HDF5StorageWriter(
                nested_path, nested_config, nested_state, total_steps=10
            )
            with writer2:
                for i in range(10):
                    ns = NestedState(
                        system=SystemData(
                            positions=nested_state.system.positions + i,
                            forces=nested_state.system.forces,
                            temperature=nested_state.system.temperature + i * 10,
                        ),
                        metadata=nested_state.metadata,
                    )
                    writer2.log(ns, i)

            with HDF5StorageReader(nested_path) as reader:
                assert reader.focus_group("group['system']")[:]["pos"].shape == (
                    10,
                    2,
                    2,
                )
                assert reader.focus_group("group['metadata']")[:]["temp"].shape == (2,)
        finally:
            import os

            if os.path.exists(nested_path):
                os.unlink(nested_path)
