# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""HDF5-backed storage for simulation trajectories.

Provides async-writing :class:`HDF5StorageWriter` (Logger protocol) and
:class:`HDF5StorageReader` for reading back logged data.  Logging frequency
is controlled via :class:`LoggingFrequency` implementations (:class:`Once`,
:class:`EveryNStep`); per-group chunking and compression are controlled via
:class:`Compression`.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import EllipsisType
from typing import Any, Literal, Protocol, Self, cast, override

import h5py
import jax
import numpy as np

from kups.core.lens import View
from kups.core.utils.jax import PyTreeDef, jit, no_post_init

type Index = int | slice | EllipsisType | tuple[Index, ...]


class LoggingFrequency(Protocol):
    """Protocol for defining when and how data should be logged during simulation.

    Implementations control logging frequency, determine HDF5 dataset dimensions,
    and map simulation steps to dataset indices.
    """

    def should_log(self, step: int) -> bool: ...
    def leading_shape(self, total_steps: int) -> tuple[int, ...]: ...
    def dataset_index(self, step: int) -> Index: ...


class Once(LoggingFrequency):
    """Logs data only at step 0, creating scalar datasets without time dimension."""

    @override
    def should_log(self, step: int) -> bool:
        return step == 0

    @override
    def leading_shape(self, total_steps: int) -> tuple[int, ...]:
        return ()

    @override
    def dataset_index(self, step: int) -> EllipsisType:
        return ...


@dataclass
class EveryNStep(LoggingFrequency):
    """Logs data every N steps, creating datasets with a time dimension.

    Args:
        n: The interval between logging steps (e.g., n=10 logs at steps 0, 10, 20, ...).
    """

    n: int

    @override
    def should_log(self, step: int) -> bool:
        return step % self.n == 0

    @override
    def leading_shape(self, total_steps: int) -> tuple[int, ...]:
        num_logged = (total_steps + self.n - 1) // self.n
        return (num_logged,)

    @override
    def dataset_index(self, step: int) -> Index:
        return step // self.n


_DEFAULT_CHUNK_BYTES = 1 << 20  # ~1 MiB; chunk- and batch-sizing target
_MAX_AUTO_BATCH = 128  # cap so cheap (small-frame) groups still flush periodically


@dataclass(frozen=True)
class Compression:
    """HDF5 compression and chunking policy for a logging group.

    Datasets are chunked along the leading (time) axis so each chunk is roughly
    ``target_chunk_bytes``: large per-frame arrays get about single-frame chunks
    (keeping random-frame reads cheap) while small or repeated-across-time fields
    get many-frame chunks (capturing cross-step redundancy without per-chunk
    bloat). Scalar datasets are stored uncompressed.

    Args:
        codec: HDF5 compression filter, ``"gzip"`` or ``"lzf"``.
        level: gzip level (0-9); ignored for lzf.
        shuffle: Enable the byte-shuffle filter. Helps float arrays; may slightly
            reduce the ratio on highly repetitive data.
        target_chunk_bytes: Approximate per-chunk size in bytes.
    """

    codec: Literal["gzip", "lzf"] = "gzip"
    level: int = 4
    shuffle: bool = True
    target_chunk_bytes: int = _DEFAULT_CHUNK_BYTES

    def dataset_kwargs(
        self, leading_dims: tuple[int, ...], shape: tuple[int, ...], itemsize: int
    ) -> dict[str, Any]:
        """Build ``create_dataset`` keyword arguments for one array leaf.

        Args:
            leading_dims: Leading (time) dimensions prepended to ``shape``.
            shape: Shape of a single logged value.
            itemsize: Bytes per element.
        """
        if not leading_dims + shape:  # scalar: cannot chunk or compress
            return {}
        kwargs: dict[str, Any] = {
            "chunks": self._chunk_shape(leading_dims, shape, itemsize),
            "shuffle": self.shuffle,
            "compression": self.codec,
        }
        if self.codec == "gzip":
            kwargs["compression_opts"] = self.level
        return kwargs

    def _chunk_shape(
        self, leading_dims: tuple[int, ...], shape: tuple[int, ...], itemsize: int
    ) -> tuple[int, ...]:
        if not leading_dims:  # logged once: one chunk spanning the whole array
            return shape
        frame_bytes = max(itemsize * math.prod(shape), 1)
        frames = max(1, min(leading_dims[0], self.target_chunk_bytes // frame_bytes))
        return (frames, *shape)


@dataclass(frozen=True)
class WriterGroupConfig[State, Storage]:
    """Configuration for a single logging group.

    Args:
        view: A lens that extracts Storage data from the full State.
        logging_frequency: Controls when this data should be logged.
        compression: Compression/chunking policy, or ``None`` to store
            uncompressed contiguous datasets.
    """

    view: View[State, Storage]
    logging_frequency: LoggingFrequency
    compression: Compression | None = field(default_factory=Compression, kw_only=True)


@dataclass(frozen=True)
class GroupWriters[State, Storage](WriterGroupConfig[State, Storage]):
    """Internal class combining a WriterGroupConfig with its initialized HDF5 writer."""

    writer: Hdf5ObjWriter[Storage]

    @cached_property
    def batch_size(self) -> int:
        """Steps to buffer before a block write, sized so the largest leaf writes
        about one chunk per flush (capped so cheap groups still flush periodically).
        ``1`` for groups logged once (no time axis).
        """
        if not self.logging_frequency.leading_shape(1):  # logged once: no time axis
            return 1
        target = (
            self.compression.target_chunk_bytes
            if self.compression
            else _DEFAULT_CHUNK_BYTES
        )
        datasets = self.writer.datasets
        frame_bytes = max(
            (ds.dtype.itemsize * math.prod(ds.shape[1:]) for ds in datasets), default=1
        )
        num_frames = min((ds.shape[0] for ds in datasets), default=1)
        return max(1, min(num_frames, _MAX_AUTO_BATCH, target // max(frame_bytes, 1)))


@dataclass
class HDF5StorageWriter[State, WriterConfig]:
    """Logs simulation state to HDF5 files. Implements the Logger protocol.

    Usage as context manager (preferred):
        ```python
        writer = HDF5StorageWriter(out_path, config, initial_state, total_steps=1000)
        with writer:
            for step in range(1000):
                state = simulate_step(state)
                writer.log(state, step)
        ```

    The writer opens the file and starts a background I/O thread on ``__enter__``,
    and flushes, records ``actual_steps``, and closes the file on ``__exit__``.

    The background thread buffers each group's steps and writes them in contiguous
    blocks to amortise per-write overhead (the dominant cost of single-step writes).
    The block size is chosen per group so the largest leaf writes about one chunk
    per flush, bounding the buffer to roughly one chunk regardless of system size;
    buffers are flushed on ``__exit__``.
    """

    out_path: str | Path
    config: WriterConfig
    initial_state: State
    total_steps: int

    # Private runtime state (set in __enter__)
    _file: h5py.File | None = field(init=False, default=None, repr=False)
    _group_writers: list[GroupWriters[State, Any]] = field(
        init=False, default_factory=list, repr=False
    )
    _bg_writer: BackgroundWriter[State, WriterConfig] | None = field(
        init=False, default=None, repr=False
    )
    _bg_thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _bg_running: threading.Event = field(
        init=False, default_factory=threading.Event, repr=False
    )
    _actual_steps: int = field(init=False, default=0, repr=False)

    def __enter__(self) -> Self:
        self._file = h5py.File(self.out_path, "w")
        self._group_writers = _init_group_writers(
            self._file, self.config, self.initial_state, self.total_steps
        )
        # Start background writer thread
        self._bg_running = threading.Event()
        self._bg_running.set()
        self._bg_writer = BackgroundWriter(self, queue.Queue(), self._bg_running)
        self._bg_thread = threading.Thread(target=self._bg_writer.start, daemon=True)
        self._bg_thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._bg_writer is not None:
            self._bg_writer.stop()
        if self._bg_thread is not None:
            self._bg_thread.join()
            self._bg_thread = None
        self._bg_writer = None
        if self._file is not None:
            self._file.attrs["actual_steps"] = self._actual_steps
            self._file.close()
            self._file = None

    def log(self, state: State, step: int) -> None:
        """Queue state for async background writing."""
        self._actual_steps = step + 1
        assert self._bg_writer is not None, "Must be used inside a with-block"
        self._bg_writer.write(state, step)

    def _prepare_write(self, state: State, step: int) -> list[tuple[int, Index, Any]]:
        """Extract loggable data on the main thread (before JAX donation)."""
        to_log: list[tuple[int, Index, Any]] = []
        for i, group in enumerate(self._group_writers):
            if group.logging_frequency.should_log(step):
                index = group.logging_frequency.dataset_index(step)
                to_log.append((i, index, group.view(state)))
        return to_log

    def _write(self, to_write: list[tuple[int, Index, Any]]) -> None:
        for i, idx, data in to_write:
            self._group_writers[i].writer.write(data, idx)


def _init_group_writers[S, WC](
    hdf5_file: h5py.File,
    config: WC,  # type: ignore
    state: S,
    total_steps: int,
) -> list[GroupWriters[S, Any]]:
    """Shared init logic: create HDF5 groups and datasets from config."""
    confs_and_paths, conf_structure = jax.tree.flatten_with_path(
        config, is_leaf=lambda x: isinstance(x, WriterGroupConfig)
    )
    confs_and_paths: list[tuple[tuple[int, ...], WriterGroupConfig[S, Any]]]
    assert all(isinstance(c[1], WriterGroupConfig) for c in confs_and_paths), (
        "All leaves of WriterConfig must be WriterGroupConfig"
    )
    hdf5_file.create_dataset(
        "config_pytree", data=np.void(pickle.dumps(conf_structure))
    )
    hdf5_file.attrs["config_class_name"] = type(config).__qualname__
    group_names: list[str] = []
    group_writers: list[GroupWriters[S, Any]] = []
    for path, group_config in confs_and_paths:
        group_name = "group" + "".join(map(str, path))
        group = hdf5_file.create_group(group_name)
        view = jit(group_config.view)
        leading_dims = group_config.logging_frequency.leading_shape(total_steps)
        writer = Hdf5ObjWriter.init(
            group, view(state), leading_dims, group_config.compression
        )
        group_writers.append(
            GroupWriters(
                view,
                group_config.logging_frequency,
                writer,
                compression=group_config.compression,
            )
        )
        group_names.append(group_name)
    hdf5_file.attrs["group_names"] = json.dumps(group_names)
    return group_writers


@dataclass
class Hdf5ObjWriter[Storage]:
    """Low-level writer for a single HDF5 group that stores a pytree of JAX arrays."""

    datasets: list[h5py.Dataset]

    @staticmethod
    def init[S](
        hdf5_group: h5py.Group,
        state: S,
        leading_dims: tuple[int, ...],
        compression: Compression | None = None,
    ) -> Hdf5ObjWriter[S]:
        datasets: list[h5py.Dataset] = []
        paths: list[str] = []
        for path, tensor in jax.tree.leaves_with_path(state):
            if not isinstance(tensor, jax.Array):
                raise ValueError(
                    f"All leaves of the storage must be jax arrays, got {type(tensor)} at path {path}"
                )
            name = "array" + "".join(map(str, path))
            dataset_shape = leading_dims + tensor.shape
            ds_kwargs: dict[str, Any] = {}
            if compression is not None:
                ds_kwargs = compression.dataset_kwargs(
                    leading_dims, tensor.shape, np.dtype(tensor.dtype).itemsize
                )
            datasets.append(
                hdf5_group.create_dataset(
                    name, shape=dataset_shape, dtype=tensor.dtype, **ds_kwargs
                )
            )
            paths.append(name)
        hdf5_group.attrs["data_class_name"] = type(state).__qualname__
        tree_def = pickle.dumps(jax.tree_util.tree_structure(state))
        hdf5_group.create_dataset("tree_def", data=np.void(tree_def))
        hdf5_group.attrs["paths"] = json.dumps(paths)
        return Hdf5ObjWriter(datasets)

    def write(self, state: Storage, index: Index) -> None:
        for dataset, value in zip(self.datasets, jax.tree.leaves(state)):
            dataset[index] = np.asarray(value)

    def write_block(self, states: list[Storage], start: int) -> None:
        """Write a contiguous block of states to ``[start : start + len(states)]``.

        Stacks each leaf across ``states`` and writes one slice per dataset,
        amortising the per-call overhead of single-step writes.
        """
        leaves = [jax.tree.leaves(s) for s in states]
        stop = start + len(states)
        for j, dataset in enumerate(self.datasets):
            dataset[start:stop] = np.stack([np.asarray(step[j]) for step in leaves])


@dataclass
class HDF5StorageReader[Config]:
    """Reader for HDF5 files created by HDF5StorageWriter.

    Usage:
        ```python
        with HDF5StorageReader[MyConfig]("output.h5") as reader:
            data = reader.focus_group("group_name")[:]
        ```
    """

    path: str | Path

    # Private runtime state
    _file: h5py.File | None = field(init=False, default=None, repr=False)

    def __enter__(self) -> Self:
        self._file = h5py.File(self.path, "r")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        assert self._file is not None, "File not open; use as context manager"
        return self._file

    def focus_group[Storage](
        self, view_or_name: View[Config, WriterGroupConfig[Any, Storage]] | str
    ) -> GroupReader[Storage]:
        """Returns a reader for a specific logging group.

        Args:
            view_or_name: Either a string group name or a View lens.
        """
        if isinstance(view_or_name, str):
            return GroupReader[Storage](self.file[view_or_name])  # type: ignore - h5py is not very good with types
        view = view_or_name
        group_names = self.list_groups()
        try:
            if "config_pytree" in self.file:
                raw = bytes(self.file["config_pytree"][()])  # type: ignore - h5py typing
            else:
                raw = bytes(self.file.attrs["config_pytree"])  # type: ignore - legacy
            conf_treedef = pickle.loads(raw)
        except Exception as e:
            raise ValueError("Failed to read config pytree") from e
        try:
            group_name = cast(str, view(jax.tree.unflatten(conf_treedef, group_names)))
        except Exception as e:
            raise ValueError("Failed to focus config") from e
        group = self.file[group_name]
        assert isinstance(group, h5py.Group), "Focused path is not a group"
        return GroupReader[Storage](group)

    def list_groups(self) -> list[str]:
        try:
            group_names = json.loads(self.file.attrs["group_names"])  # type: ignore - h5py is not very good with types
            return group_names
        except Exception as e:
            raise ValueError("Failed to read group names") from e


@dataclass
class GroupReader[Storage]:
    """Reader for a single HDF5 logging group, providing array-like access."""

    group: h5py.Group

    @cached_property
    def paths(self) -> list[str]:
        return json.loads(self.group.attrs["paths"])  # type: ignore - h5py is not very good with types

    @cached_property
    def tree_def(self) -> PyTreeDef[Storage]:
        if "tree_def" in self.group:
            raw = bytes(self.group["tree_def"][()])  # type: ignore - h5py typing
        else:
            raw = bytes(self.group.attrs["tree_def"])  # type: ignore - legacy
        tree_def = pickle.loads(raw)
        return tree_def

    def read(self, index: Index) -> Storage:
        if index is None:
            index = slice(None)

        def read_dataset(
            path: str,
        ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
            dataset = self.group["".join(map(str, path))]
            return dataset[index]  # type: ignore - pylance doesn't understand h5py correctly.

        with no_post_init():
            return self.tree_def.unflatten(jax.tree.map(read_dataset, self.paths))

    def __getitem__(self, index: Index) -> Storage:
        return self.read(index)


@dataclass
class BackgroundWriter[State, WriterConfig]:
    """Background thread worker that asynchronously writes pre-extracted data to HDF5.

    When a group's batch size is >1 its per-step data is buffered and written in
    contiguous blocks, amortising per-write overhead. Buffers are flushed when full
    and once more when the thread stops.
    """

    storage_writer: HDF5StorageWriter[State, WriterConfig]
    data_queue: queue.Queue[list[tuple[int, Index, Any]]]
    running: threading.Event
    # Per-group buffer of (dataset_index, data) pending a block write.
    _buffers: defaultdict[int, list[tuple[Index, Any]]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )

    def start(self) -> None:
        """Main loop for the background writer thread."""
        logging.info("Writer thread started")
        while self.running.is_set():
            try:
                to_log = self.data_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._buffer(to_log)
            except Exception as e:
                logging.error(f"Error processing data: {e}")
            finally:
                self.data_queue.task_done()
        self._flush_all()  # drain remaining buffers on shutdown
        logging.info("Writer thread stopped")

    def write(self, state: State, step: int) -> None:
        """Queue state data for asynchronous writing to HDF5."""
        self.data_queue.put(self.storage_writer._prepare_write(state, step))

    def _buffer(self, to_log: list[tuple[int, Index, Any]]) -> None:
        group_writers = self.storage_writer._group_writers
        for i, index, data in to_log:
            buffer = self._buffers[i]
            buffer.append((index, data))
            if len(buffer) >= group_writers[i].batch_size:
                self._flush_group(i)

    def _flush_group(self, i: int) -> None:
        buffer = self._buffers.get(i)
        if not buffer:
            return
        writer = self.storage_writer._group_writers[i].writer
        indices = [index for index, _ in buffer]
        first = indices[0]
        if (
            len(buffer) > 1
            and isinstance(first, int)
            and all(x == first + k for k, x in enumerate(indices))
        ):
            writer.write_block([data for _, data in buffer], first)
        else:
            for index, data in buffer:
                writer.write(data, index)
        buffer.clear()

    def _flush_all(self) -> None:
        for i in list(self._buffers):
            self._flush_group(i)

    def stop(self) -> None:
        """Drain the queue, then signal the thread to flush buffers and stop."""
        self.data_queue.join()
        self.running.clear()
