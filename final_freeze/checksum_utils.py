"""Checksum helpers for final freeze packages."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze.freeze_io import progress_iter


def compute_sha256(path: Path) -> str:
    """Compute SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_checksum_row(path: Path, root: Path) -> Dict[str, Any]:
    """Return one checksum manifest row."""
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": compute_sha256(path),
    }


def checksum_rows_for_files(files: List[Path], root: Path, show_progress: bool = True) -> List[Dict[str, Any]]:
    """Compute checksum rows for files under a root."""
    rows = []
    for path in progress_iter(files, show_progress, "checksums", "file"):
        if path.exists() and path.is_file():
            rows.append(file_checksum_row(path, root))
    return rows


def write_checksum_text(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write sha256sum-style checksums."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["%s  %s" % (row.get("sha256"), row.get("path")) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

