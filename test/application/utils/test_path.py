# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.utils.path."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kups.application.utils.path import get_model_path


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub ``huggingface_hub`` in :data:`sys.modules` since the real package
    is an optional dependency and not installed in the test environment.
    """
    fake = MagicMock(name="huggingface_hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return fake


class TestLocalPath:
    def test_existing_file_returned(self, tmp_path: Path) -> None:
        target = tmp_path / "model.bin"
        target.write_bytes(b"")
        assert get_model_path(target) == target
        assert get_model_path(str(target)) == target

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_model_path(tmp_path / "missing.bin")


class TestHuggingFacePath:
    def test_downloads_and_returns_path(
        self, tmp_path: Path, fake_hf: MagicMock
    ) -> None:
        target = tmp_path / "weights.safetensors"
        target.write_bytes(b"")
        fake_hf.hf_hub_download.return_value = str(target)

        result = get_model_path("hf://acme/my-model/weights.safetensors")

        assert result == target
        fake_hf.hf_hub_download.assert_called_once_with(
            repo_id="acme/my-model", filename="weights.safetensors"
        )

    def test_filename_with_subdirs(self, tmp_path: Path, fake_hf: MagicMock) -> None:
        target = tmp_path / "file.bin"
        target.write_bytes(b"")
        fake_hf.hf_hub_download.return_value = str(target)

        get_model_path("hf://acme/my-model/sub/dir/file.bin")

        fake_hf.hf_hub_download.assert_called_once_with(
            repo_id="acme/my-model", filename="sub/dir/file.bin"
        )

    def test_downloaded_file_missing_raises(
        self, tmp_path: Path, fake_hf: MagicMock
    ) -> None:
        fake_hf.hf_hub_download.return_value = str(tmp_path / "ghost.bin")
        with pytest.raises(FileNotFoundError):
            get_model_path("hf://owner/repo/ghost.bin")

    @pytest.mark.parametrize(
        "uri",
        [
            "hf://onlyone",
            "hf://owner/repo",
            "hf://owner/repo/",
            "hf://owner//file.bin",
            "hf:///repo/file.bin",
        ],
    )
    def test_invalid_uri_raises(self, uri: str, fake_hf: MagicMock) -> None:
        with pytest.raises(ValueError, match="hf://"):
            get_model_path(uri)
        fake_hf.hf_hub_download.assert_not_called()

    def test_missing_dependency_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ImportError, match="huggingface_hub"):
            get_model_path("hf://owner/repo/file.bin")
