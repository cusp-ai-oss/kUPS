# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for resolving model paths from the local filesystem or Hugging Face Hub."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HF_PREFIX = "hf://"


def get_model_path(path: str | Path) -> Path:
    """Resolve a model path, downloading from Hugging Face Hub if needed.

    Accepts either a local filesystem path or an ``hf://`` URI of the form
    ``hf://<owner>/<repo>/<filename>``. For ``hf://`` URIs the file is
    fetched via :func:`huggingface_hub.hf_hub_download` and the cached local
    path is returned. The resolved path is verified to exist on disk.

    Args:
        path: Local path or ``hf://<owner>/<repo>/<filename>`` URI.

    Returns:
        Local path to the resolved file.

    Raises:
        ImportError: If ``path`` is an ``hf://`` URI and the optional
            ``huggingface_hub`` dependency is not installed.
        ValueError: If ``path`` is an ``hf://`` URI that cannot be parsed
            into ``<owner>/<repo>/<filename>``.
        FileNotFoundError: If the resolved path does not exist.
    """
    path_str = str(path)
    if path_str.startswith(HF_PREFIX):
        try:
            from huggingface_hub import (
                hf_hub_download,  # pyright: ignore[reportMissingImports]
            )
        except ImportError as e:
            raise ImportError(
                "Resolving 'hf://' model paths requires the optional "
                "'huggingface_hub' dependency. Install it with "
                "`pip install kups[hf]` or `pip install huggingface_hub`."
            ) from e
        parts = path_str[len(HF_PREFIX) :].split("/", 2)
        if len(parts) < 3 or not all(parts):
            raise ValueError(
                f"Invalid hf:// URI {path_str!r}; expected "
                "'hf://<owner>/<repo>/<filename>'."
            )
        owner, repo, filename = parts
        repo_id = f"{owner}/{repo}"
        logger.info("Retrieving %s from Hugging Face Hub (%s).", filename, repo_id)
        resolved = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    else:
        resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Resolved model path does not exist: {resolved}")
    return resolved
