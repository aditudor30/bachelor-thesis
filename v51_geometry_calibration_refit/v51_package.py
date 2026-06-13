"""Freeze and package only a selected, valid, true fit-train V5.1 candidate."""

import shutil
import zipfile
from typing import Any, Dict

from deep_oc_sort_3d.official_023_027.official_track1_io import compute_sha256
from deep_oc_sort_3d.v51_geometry_calibration_refit.track1_validator import validate_v51_track1
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_config import output_root, variant_root
from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import prepare_directory, read_json, write_json


def package_selected_v51(
    config: Dict[str, Any], progress: bool = True, overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    root = output_root(config)
    selection = read_json(root / "comparison" / "selected_variant.json")
    corrections = read_json(root / "learned_corrections" / "selected_corrections.json")
    variant = selection.get("selected_variant")
    frozen_root = root / "frozen_candidate"
    comparison_root = frozen_root / "comparison"
    if not variant or corrections.get("fit_source") != "fit_train":
        candidate = {
            "ready": False, "selected_variant": None, "validation_errors": None,
            "fit_source": corrections.get("fit_source"), "zip_path": None,
        }
        write_json(comparison_root / "upload_readiness.json", {
            "v51_geometry_calibrated_official": candidate,
            "recommendation": "V5 remains the calibrated candidate because V5.1 did not pass the strict fit_train and validation gates.",
        })
        write_json(comparison_root / "verdict.json", {"label": selection.get("verdict"), "selected_variant": None})
        return candidate
    if not prepare_directory(frozen_root, overwrite=overwrite, skip_existing=skip_existing):
        existing = read_json(comparison_root / "upload_readiness.json")
        return existing.get("v51_geometry_calibrated_official", existing)
    source = variant_root(config, str(variant)) / "track1.txt"
    target = frozen_root / "v51_geometry_calibrated_official" / "track1.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(target))
    validation = validate_v51_track1(target, target.parent / "validation_summary.json", config, progress=progress)
    zip_path = frozen_root / "packages" / "v51_geometry_calibrated_official_023_027_track1.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(str(target), arcname="track1.txt")
    zip_size_mb = float(zip_path.stat().st_size) / (1024.0 * 1024.0)
    ready = (
        validation.get("status") == "ok" and int(validation.get("num_errors", 1)) == 0
        and zip_size_mb <= float(config.get("packaging", {}).get("max_zip_size_mb", 50.0))
    )
    identity = validation.get("identity_preservation", {})
    candidate = {
        "ready": ready, "selected_variant": variant, "track1_path": str(target), "zip_path": str(zip_path),
        "validation_errors": validation.get("num_errors"), "scene_ids": validation.get("scene_ids"),
        "class_mapping": "official", "float_rounding_decimals": int(config.get("official_track1", {}).get("round_float_decimals", 2)),
        "rows": validation.get("total_rows"), "unique_tracks": identity.get("candidate_unique_tracks"),
        "fit_source": corrections.get("fit_source"), "zip_size_mb": zip_size_mb,
        "sha256": compute_sha256(target), "identity_preservation": identity,
    }
    write_json(target.parent / "manifest.json", {
        "selected_variant": variant, "source_track1": str(source), "track1_path": str(target),
        "rows": candidate["rows"], "unique_tracks": candidate["unique_tracks"],
        "fit_source": candidate["fit_source"], "sha256": candidate["sha256"],
        "gt_or_depth_used_on_test": False,
    })
    (target.parent / "sha256.txt").write_text("%s  track1.txt\n" % candidate["sha256"], encoding="utf-8")
    write_json(zip_path.parent / "package_manifest.json", {
        "zip_path": str(zip_path), "zip_size_mb": zip_size_mb, "contains": ["track1.txt"], "ready": ready,
    })
    (zip_path.parent / "package_checksums.txt").write_text("%s  %s\n" % (compute_sha256(zip_path), zip_path.name), encoding="utf-8")
    write_json(comparison_root / "upload_readiness.json", {
        "v51_geometry_calibrated_official": candidate,
        "recommendation": "Upload only if the official submission budget allows a robust refit candidate after V2 and V5/V3.1.",
    })
    label = selection.get("verdict") if ready else "v51_invalid_fix_required"
    write_json(comparison_root / "verdict.json", {"label": label, "selected_variant": variant})
    if not ready:
        write_json(root / "comparison" / "verdict.json", {
            "label": "v51_invalid_fix_required", "selected_variant": variant,
            "reasons": ["frozen_package_validation_or_size_gate_failed"],
        })
    return candidate
