"""Build official 023-027 Track1 files from old and extension outputs."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.official_023_027.official_config import (
    VARIANT_KEYS,
    class_remap,
    extension_track1_path,
    frozen_variant_root,
    old_track1_path,
    output_root,
    scene_ids,
    source_track1_path,
    variant_official_root,
)
from deep_oc_sort_3d.official_023_027.official_track1_io import (
    compute_sha256,
    read_json,
    read_track1_rows,
    write_json,
    write_track1_rows,
)
from deep_oc_sort_3d.official_023_027.official_track1_remap import remap_rows_to_official, stable_deduplicate_rows
from deep_oc_sort_3d.official_023_027.official_track1_validator import validate_and_write


def build_official_track1_candidates(
    config: Dict[str, Any],
    mode: str = "incremental",
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Merge, remap, round, validate and freeze V2/V3 official candidates."""
    if mode not in ("incremental", "rerun_all"):
        raise ValueError("Unsupported mode: %s" % mode)
    mapping_error = _mapping_error(config)
    if mapping_error is not None:
        results = [
            {"variant": variant, "candidate_name": _candidate_name(variant), "status": "class_mapping_audit_failed", "error": mapping_error}
            for variant in VARIANT_KEYS
        ]
        compliance = _build_compliance_audit(config, results, mode)
        write_json(output_root(config) / "audit" / "compliance_audit.json", compliance)
        return {"mode": mode, "variants": results, "compliance": compliance}
    results = []
    for variant in VARIANT_KEYS:
        results.append(_build_one(config, variant, mode, progress, overwrite, skip_existing))
    compliance = _build_compliance_audit(config, results, mode)
    write_json(output_root(config) / "audit" / "compliance_audit.json", compliance)
    return {"mode": mode, "variants": results, "compliance": compliance}


def _build_one(
    config: Dict[str, Any],
    variant: str,
    mode: str,
    progress: bool,
    overwrite: bool,
    skip_existing: bool,
) -> Dict[str, Any]:
    processing_root = variant_official_root(config, variant)
    processing_path = processing_root / "track1.txt"
    frozen_root = frozen_variant_root(config, variant)
    frozen_path = frozen_root / "track1.txt"
    if frozen_path.exists() and skip_existing and not overwrite:
        validation = validate_and_write(frozen_path, frozen_root / "validation_summary.json", config, progress=progress)
        return _summary_from_existing(variant, frozen_path, validation, mode)
    if frozen_path.exists() and not overwrite:
        raise FileExistsError("Official frozen candidate exists; use --overwrite or --skip-existing: %s" % frozen_path)
    extension = extension_track1_path(config, variant)
    legacy_source = _preferred_legacy_source(config, variant)
    sources = [extension] if mode == "rerun_all" else [legacy_source, extension]
    missing = [str(path) for path in sources if not path.exists() or path.stat().st_size <= 0]
    if missing:
        return {
            "variant": variant,
            "status": "missing_new_test_scene_outputs",
            "mode": mode,
            "missing_sources": missing,
            "source_paths": [str(path) for path in sources],
        }
    rows = []
    source_counts = {}
    for source in sources:
        source_rows = read_track1_rows(source, progress=progress)
        rows.extend(source_rows)
        source_counts[str(source)] = len(source_rows)
    remapped, remap_summary = remap_rows_to_official(rows, class_remap(config))
    deduplicated, dedup_summary = stable_deduplicate_rows(remapped)
    decimals = int(config.get("official_track1", {}).get("round_float_decimals", 2))
    write_track1_rows(processing_path, deduplicated, decimals=decimals)
    validation = validate_and_write(processing_path, processing_root / "validation_summary.json", config, progress=progress)
    frozen_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(processing_path), str(frozen_path))
    frozen_validation = validate_and_write(frozen_path, frozen_root / "validation_summary.json", config, progress=progress)
    sha = compute_sha256(frozen_path)
    build_ready = frozen_validation.get("status") == "ok" and remap_summary.get("status") == "ok"
    manifest = {
        "candidate_name": _candidate_name(variant),
        "mode": mode,
        "source_paths": [str(path) for path in sources],
        "source_row_counts": source_counts,
        "official_track1_path": str(processing_path),
        "frozen_track1_path": str(frozen_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "sha256": sha,
        "file_size_bytes": frozen_path.stat().st_size,
        "line_count": len(deduplicated),
        "track1_validation_errors": frozen_validation.get("num_errors"),
        "class_mapping": "official",
        "internal_to_official": class_remap(config),
        "float_rounding_decimals": decimals,
        "deduplication": dedup_summary,
        "remap_summary": remap_summary,
        "status": "ready" if build_ready else "not_ready",
    }
    write_json(processing_root / "manifest.json", manifest)
    write_json(frozen_root / "manifest.json", manifest)
    (processing_root / "sha256.txt").write_text("%s  track1.txt\n" % sha, encoding="utf-8")
    (frozen_root / "sha256.txt").write_text("%s  track1.txt\n" % sha, encoding="utf-8")
    return {
        "variant": variant,
        "candidate_name": _candidate_name(variant),
        "status": manifest.get("status"),
        "mode": mode,
        "rows": len(deduplicated),
        "scene_ids": frozen_validation.get("scene_ids", []),
        "per_scene_rows": frozen_validation.get("per_scene_rows", {}),
        "per_class_rows": frozen_validation.get("per_class_rows", {}),
        "validation_errors": frozen_validation.get("num_errors"),
        "duplicate_rows_removed": dedup_summary.get("duplicate_rows_removed"),
        "sha256": sha,
        "track1_path": str(frozen_path),
    }


def _build_compliance_audit(config: Dict[str, Any], rows: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    expected = scene_ids(config)
    checks = {}
    for row in rows:
        name = str(row.get("candidate_name", row.get("variant")))
        checks[name] = {
            "status": row.get("status"),
            "validation_errors": row.get("validation_errors"),
            "all_scene_ids_present": row.get("scene_ids") == expected,
            "scene_ids": row.get("scene_ids"),
            "class_mapping": "official",
            "float_rounding_decimals": int(config.get("official_track1", {}).get("round_float_decimals", 2)),
        }
    return {
        "status": "ok" if checks and all(value.get("status") == "ready" for value in checks.values()) else "error",
        "mode": mode,
        "expected_scene_ids": expected,
        "class_remap_applied_at_final_export_only": True,
        "candidates": checks,
    }


def _summary_from_existing(variant: str, path: Path, validation: Dict[str, Any], mode: str) -> Dict[str, Any]:
    manifest = read_json(path.parent / "manifest.json")
    ready = validation.get("status") == "ok" and manifest.get("status") == "ready"
    return {
        "variant": variant,
        "candidate_name": _candidate_name(variant),
        "status": "ready" if ready else "not_ready",
        "mode": mode,
        "rows": validation.get("total_rows"),
        "scene_ids": validation.get("scene_ids", []),
        "per_scene_rows": validation.get("per_scene_rows", {}),
        "per_class_rows": validation.get("per_class_rows", {}),
        "validation_errors": validation.get("num_errors"),
        "sha256": compute_sha256(path),
        "track1_path": str(path),
    }


def _candidate_name(variant: str) -> str:
    return "v2_current_official" if variant == "v2_current" else "v3_gap_aware_soft_official"


def _mapping_error(config: Dict[str, Any]) -> Any:
    internal = {int(key): str(value) for key, value in config.get("class_mapping", {}).get("internal", {}).items()}
    official = {int(key): str(value) for key, value in config.get("class_mapping", {}).get("official", {}).items()}
    remap = class_remap(config)
    if set(remap.keys()) != set(range(7)) or set(remap.values()) != set(range(7)):
        return "internal_to_official_is_not_a_complete_bijection"
    if not all(internal.get(source) == official.get(target) for source, target in remap.items()):
        return "class_names_do_not_match_after_remap"
    return None


def _preferred_legacy_source(config: Dict[str, Any], variant: str) -> Path:
    frozen = old_track1_path(config, variant)
    if frozen.is_file() and frozen.stat().st_size > 0:
        return frozen
    return source_track1_path(config, variant)
