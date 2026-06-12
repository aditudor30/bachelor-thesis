"""Copy existing Track1 outputs into stable frozen candidate directories."""

import shutil
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.freeze_upload.freeze_config import candidate_output_root, candidate_specs, write_resolved_config
from deep_oc_sort_3d.freeze_upload.freeze_io import compute_sha256, progress_iter
from deep_oc_sort_3d.freeze_upload.track1_validator import validate_and_write
from deep_oc_sort_3d.freeze_upload.upload_manifest import build_candidate_manifest, write_candidate_manifest


def freeze_candidates(
    config: Dict[str, Any],
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Freeze configured source Track1 files without modifying their contents."""
    write_resolved_config(config)
    results = []
    for spec in progress_iter(candidate_specs(config), progress, "freeze upload candidates"):
        results.append(_freeze_one(config, spec, progress, overwrite, skip_existing))
    return {"candidates": results}


def validate_frozen_candidates(config: Dict[str, Any], progress: bool = True) -> Dict[str, Any]:
    """Revalidate already frozen candidates and refresh their manifests."""
    results = []
    for spec in progress_iter(candidate_specs(config), progress, "validate frozen candidates"):
        name = str(spec.get("candidate_name"))
        root = candidate_output_root(config, name)
        frozen = root / "track1.txt"
        validation = validate_and_write(frozen, root / "validation_summary.json", config, progress=progress)
        notes = _notes(name)
        manifest = build_candidate_manifest(name, Path(str(spec.get("source_track1_path", ""))), frozen, validation, notes)
        write_candidate_manifest(root / "manifest.json", manifest)
        _write_sha(root, manifest)
        results.append({"candidate_name": name, "validation": validation, "manifest": manifest})
    return {"candidates": results}


def _freeze_one(
    config: Dict[str, Any],
    spec: Dict[str, Any],
    progress: bool,
    overwrite: bool,
    skip_existing: bool,
) -> Dict[str, Any]:
    name = str(spec.get("candidate_name"))
    source = Path(str(spec.get("source_track1_path", "")))
    root = candidate_output_root(config, name)
    frozen = root / "track1.txt"
    root.mkdir(parents=True, exist_ok=True)
    if not source.exists() or not source.is_file():
        validation = validate_and_write(frozen, root / "validation_summary.json", config, progress=progress)
        manifest = build_candidate_manifest(name, source, frozen, validation, _notes(name))
        write_candidate_manifest(root / "manifest.json", manifest)
        _write_sha(root, manifest)
        return {"candidate_name": name, "status": "missing_source", "manifest": manifest, "validation": validation}
    if source.stat().st_size <= 0:
        raise ValueError("Source Track1 file is empty: %s" % source)
    if frozen.exists() and not overwrite and not skip_existing:
        raise FileExistsError("Frozen candidate already exists; use --overwrite or --skip-existing: %s" % frozen)
    if not (frozen.exists() and skip_existing and not overwrite):
        shutil.copy2(str(source), str(frozen))
    source_sha = compute_sha256(source)
    frozen_sha = compute_sha256(frozen)
    if source_sha != frozen_sha:
        raise ValueError("Frozen copy checksum differs from source for %s" % name)
    validation = validate_and_write(frozen, root / "validation_summary.json", config, progress=progress)
    manifest = build_candidate_manifest(name, source, frozen, validation, _notes(name))
    manifest["source_sha256"] = source_sha
    manifest["content_unchanged"] = source_sha == frozen_sha
    write_candidate_manifest(root / "manifest.json", manifest)
    _write_sha(root, manifest)
    return {"candidate_name": name, "status": manifest.get("status"), "manifest": manifest, "validation": validation}


def _write_sha(root: Path, manifest: Dict[str, Any]) -> None:
    value = manifest.get("sha256")
    line = "%s  track1.txt\n" % value if value else ""
    (root / "sha256.txt").write_text(line, encoding="utf-8")


def _notes(name: str) -> str:
    if name == "v3_gap_aware_soft":
        return "ByteTrack local + gap_aware_soft motion filtering"
    return "V2 current pseudo3D fullcam coverage-first candidate"
