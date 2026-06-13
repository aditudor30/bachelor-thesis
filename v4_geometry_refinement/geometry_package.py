"""Freeze and package the selected V4 geometry candidate."""

import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from deep_oc_sort_3d.official_023_027.official_track1_io import compute_sha256
from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import output_root, variant_root
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import prepare_directory, read_json, write_json
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_validator import validate_v4_track1


def package_selected_geometry_variant(
    config: Dict[str, Any],
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Copy, revalidate, checksum and zip the selected Track1 file."""
    root = output_root(config)
    selected = read_json(root / "comparison" / "selected_variant.json")
    variant = selected.get("selected_variant")
    package_root = root / "frozen_candidate"
    comparison_root = package_root / "comparison"
    if not variant:
        report = {
            "ready": False,
            "selected_variant": None,
            "verdict": selected.get("verdict", "v4_geometry_refinement_invalid_fix_required"),
            "reason": "No V4 variant was selected; V3.1 remains unchanged.",
        }
        write_json(comparison_root / "upload_readiness.json", {"v4_geometry_refined_official": report, "recommendation": report["reason"]})
        write_json(comparison_root / "verdict.json", {"label": report["verdict"], "selected_variant": None})
        return report
    if not prepare_directory(package_root, overwrite=overwrite, skip_existing=skip_existing):
        existing = read_json(comparison_root / "upload_readiness.json")
        return existing.get("v4_geometry_refined_official", existing)

    source = variant_root(config, str(variant)) / "track1.txt"
    frozen_track1 = package_root / "v4_geometry_refined_official" / "track1.txt"
    frozen_track1.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(frozen_track1))
    validation = validate_v4_track1(
        frozen_track1,
        package_root / "v4_geometry_refined_official" / "validation_summary.json",
        config,
        progress=progress,
    )
    zip_path = package_root / "packages" / "v4_geometry_refined_official_023_027_track1.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(str(frozen_track1), arcname="track1.txt")
    size_mb = float(zip_path.stat().st_size) / (1024.0 * 1024.0)
    max_size = float(config.get("packaging", {}).get("max_zip_size_mb", 50.0))
    ready = validation.get("status") == "ok" and int(validation.get("num_errors", 1)) == 0 and size_mb <= max_size
    unique_tracks = validation.get("identity_preservation", {}).get("candidate_unique_tracks")
    candidate = {
        "ready": ready,
        "selected_variant": variant,
        "track1_path": str(frozen_track1),
        "zip_path": str(zip_path),
        "zip_size_mb": size_mb,
        "sha256": compute_sha256(frozen_track1),
        "rows": validation.get("total_rows"),
        "unique_tracks": unique_tracks,
        "scene_ids": validation.get("scene_ids"),
        "per_scene_rows": validation.get("per_scene_rows"),
        "per_class_rows": validation.get("per_class_rows"),
        "validation_errors": validation.get("num_errors"),
        "class_mapping": "official",
        "float_rounding_decimals": int(config.get("official_track1", {}).get("round_float_decimals", 2)),
        "identity_preservation": validation.get("identity_preservation"),
    }
    recommendation = "Upload only if official submission budget allows a fourth candidate after V2 official, V3.1 and V3 official."
    readiness = {"v4_geometry_refined_official": candidate, "recommendation": recommendation}
    manifest = {
        "selected_variant": variant,
        "source_track1": str(source),
        "track1_path": str(frozen_track1),
        "rows": validation.get("total_rows"),
        "unique_tracks": unique_tracks,
        "sha256": candidate["sha256"],
    }
    write_json(frozen_track1.parent / "manifest.json", manifest)
    (frozen_track1.parent / "sha256.txt").write_text("%s  track1.txt\n" % candidate["sha256"], encoding="utf-8")
    write_json(zip_path.parent / "package_manifest.json", {
        "zip_path": str(zip_path), "zip_size_mb": size_mb, "contains": ["track1.txt"], "ready": ready,
    })
    (zip_path.parent / "package_checksums.txt").write_text("%s  %s\n" % (compute_sha256(zip_path), zip_path.name), encoding="utf-8")
    write_json(comparison_root / "upload_readiness.json", readiness)
    write_json(comparison_root / "verdict.json", {
        "label": "v4_geometry_refinement_ready_for_upload" if ready else "v4_geometry_refinement_invalid_fix_required",
        "selected_variant": variant,
        "reasons": [] if ready else ["frozen_package_validation_or_size_gate_failed"],
    })
    return candidate
