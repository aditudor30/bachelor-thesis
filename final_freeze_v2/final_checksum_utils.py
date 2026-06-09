"""Checksum helpers for final freeze v2."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import progress_iter


def compute_sha256(path: Path) -> str:
    """Compute SHA256 for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_sha256(path: Path, large_file_mb_threshold: float = 200.0, checksum_large_files: bool = False) -> Optional[str]:
    """Compute checksum unless file is too large and large checksums are disabled."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    size_mb = float(path.stat().st_size) / (1024.0 * 1024.0)
    if size_mb > float(large_file_mb_threshold) and not checksum_large_files:
        return None
    return compute_sha256(path)


def checksum_rows_for_files(
    files: List[Path],
    root: Path,
    show_progress: bool = True,
    large_file_mb_threshold: float = 200.0,
    checksum_large_files: bool = False,
) -> List[Dict[str, Any]]:
    """Build checksum rows for files."""
    rows = []
    for path in progress_iter(files, show_progress, "checksums", "file"):
        if not path.exists() or not path.is_file():
            continue
        sha = maybe_sha256(path, large_file_mb_threshold, checksum_large_files)
        rows.append(
            {
                "path": str(path.relative_to(root)) if str(path).startswith(str(root)) else str(path),
                "size_bytes": path.stat().st_size,
                "sha256": "" if sha is None else sha,
            }
        )
    return rows


def write_checksum_text(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write sha256sum-compatible text for rows with checksums."""
    lines = []
    for row in rows:
        if row.get("sha256"):
            lines.append("%s  %s" % (row.get("sha256"), row.get("path")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
