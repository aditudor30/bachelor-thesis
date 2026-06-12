"""Manifest construction for frozen candidates and packages."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.freeze_upload.freeze_io import compute_sha256, count_nonempty_lines, write_json


def build_candidate_manifest(
    candidate_name: str,
    source_path: Path,
    frozen_path: Path,
    validation: Dict[str, Any],
    notes: str,
) -> Dict[str, Any]:
    """Build one immutable-candidate manifest."""
    exists = frozen_path.exists() and frozen_path.is_file()
    return {
        "candidate_name": candidate_name,
        "source_track1_path": str(source_path),
        "frozen_track1_path": str(frozen_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "sha256": compute_sha256(frozen_path) if exists else None,
        "file_size_bytes": frozen_path.stat().st_size if exists else 0,
        "line_count": count_nonempty_lines(frozen_path) if exists else 0,
        "track1_validation_errors": int(validation.get("num_errors", 1) or 0),
        "status": "ready" if exists and int(validation.get("num_errors", 1) or 0) == 0 else "not_ready",
        "notes": notes,
    }


def write_candidate_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Write one candidate manifest."""
    write_json(path, manifest)


def build_package_manifest(packages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the combined package manifest."""
    return {"created_at": datetime.utcnow().isoformat() + "Z", "packages": packages}

