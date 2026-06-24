# Copyright 2026 Marimo. All rights reserved.
"""`_copy_lazy_caches_to_export` — bundling executed-export caches.

The function is filesystem-driven: kernel teardown records an export
manifest of the cache keys a session produced, and the export step
copies exactly those files into `<out_dir>/public/cache/` where the
WASM store's HTTP fallback fetches them. These tests exercise it
against a synthetic `__marimo__/cache` tree; no kernel run required.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from marimo._cli.export.commands import _copy_lazy_caches_to_export
from marimo._utils.marimo_path import MarimoPath
from marimo._utils.paths import notebook_output_dir

if TYPE_CHECKING:
    from pathlib import Path


def _notebook(tmp_path: Path) -> MarimoPath:
    nb = tmp_path / "nb.py"
    nb.write_text("import marimo\napp = marimo.App()\n")
    return MarimoPath(str(nb))


def _cache_dir(tmp_path: Path) -> Path:
    cache = notebook_output_dir(tmp_path) / "cache"
    cache.mkdir(parents=True)
    return cache


def test_no_manifest_is_a_noop(tmp_path: Path) -> None:
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    _copy_lazy_caches_to_export(_notebook(tmp_path), out_dir)
    assert not (out_dir / "public" / "cache").exists()


def test_manifest_keys_copied_and_manifest_consumed(tmp_path: Path) -> None:
    cache = _cache_dir(tmp_path)
    (cache / "lazy").mkdir()
    (cache / "lazy" / "E_abc.jsonl").write_bytes(b"manifest-line\n")
    (cache / "lazy" / "blob.npy").write_bytes(b"\x93NUMPY")
    manifest = cache / ".lazy_export_manifest.json"
    manifest.write_text(json.dumps(["lazy/E_abc.jsonl", "lazy/blob.npy"]))

    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    _copy_lazy_caches_to_export(_notebook(tmp_path), out_dir)

    dst = out_dir / "public" / "cache"
    assert (dst / "lazy" / "E_abc.jsonl").read_bytes() == b"manifest-line\n"
    assert (dst / "lazy" / "blob.npy").read_bytes() == b"\x93NUMPY"
    # One-shot: the manifest is consumed so a later re-export of an
    # unexecuted notebook doesn't bundle stale caches.
    assert not manifest.exists()


def test_missing_listed_file_skipped(tmp_path: Path) -> None:
    cache = _cache_dir(tmp_path)
    (cache / "present.bin").write_bytes(b"ok")
    (cache / ".lazy_export_manifest.json").write_text(
        json.dumps(["present.bin", "evicted.bin"])
    )

    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    _copy_lazy_caches_to_export(_notebook(tmp_path), out_dir)

    dst = out_dir / "public" / "cache"
    assert (dst / "present.bin").exists()
    assert not (dst / "evicted.bin").exists()


def test_keys_outside_session_not_bundled(tmp_path: Path) -> None:
    """Only manifest-listed keys ship — other cache files on disk stay."""
    cache = _cache_dir(tmp_path)
    (cache / "mine.bin").write_bytes(b"mine")
    (cache / "other.bin").write_bytes(b"other-session")
    (cache / ".lazy_export_manifest.json").write_text(json.dumps(["mine.bin"]))

    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    _copy_lazy_caches_to_export(_notebook(tmp_path), out_dir)

    dst = out_dir / "public" / "cache"
    assert (dst / "mine.bin").exists()
    assert not (dst / "other.bin").exists()
