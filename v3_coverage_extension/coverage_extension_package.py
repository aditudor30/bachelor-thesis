"""Freeze and package the selected V3.1 official candidate."""

import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.official_023_027.official_track1_io import compute_sha256, read_track1_rows
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import output_root, variant_root
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import read_json, write_json
from deep_oc_sort_3d.v3_coverage_extension.official_track1_validator import validate_variant_track1


def package_selected_candidate(config: Dict[str, Any], progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Copy, revalidate, checksum and zip the selected balanced variant."""
    root = output_root(config)
    selected_data = read_json(root / "comparison" / "selected_variant.json")
    selected = selected_data.get("selected_variant")
    if selected is None:
        result = {"ready": False, "reason": "no_selected_variant"}
        write_json(root / "frozen_candidate" / "comparison" / "upload_readiness.json", {"v3_coverage_extended_official": result, "recommendation": "Keep V3 official."})
        return result
    source = variant_root(config, str(selected)) / "track1_official.txt"
    frozen = root / "frozen_candidate" / "v3_coverage_extended_official"
    packages = root / "frozen_candidate" / "packages"
    zip_path = packages / "v3_coverage_extended_official_023_027_track1.zip"
    if (frozen.exists() or zip_path.exists()) and not overwrite:
        raise FileExistsError("Frozen V3.1 output exists; use --overwrite or --skip-existing")
    if overwrite and frozen.exists():
        shutil.rmtree(str(frozen))
    frozen.mkdir(parents=True, exist_ok=True)
    packages.mkdir(parents=True, exist_ok=True)
    target = frozen / "track1.txt"
    shutil.copy2(str(source), str(target))
    validation = validate_variant_track1(target, frozen / "validation_summary.json", config, progress=progress)
    sha256 = compute_sha256(target)
    (frozen / "sha256.txt").write_text(sha256 + "  track1.txt\n", encoding="utf-8")
    if zip_path.exists() and overwrite:
        zip_path.unlink()
    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(str(target), arcname="track1.txt")
    rows = read_track1_rows(target, progress=False)
    unique_tracks = len(set((row.scene_id, row.class_id, row.object_id) for row in rows))
    zip_size_mb = float(zip_path.stat().st_size) / (1024.0 * 1024.0)
    max_zip = float(config.get("packaging", {}).get("max_zip_size_mb", 50.0))
    ready = validation.get("status") == "ok" and zip_size_mb <= max_zip
    manifest = {
        "selected_variant": selected, "source_track1": str(source), "track1_path": str(target),
        "zip_path": str(zip_path), "rows": len(rows), "unique_tracks": unique_tracks,
        "zip_size_mb": zip_size_mb, "sha256": sha256, "validation_errors": validation.get("num_errors", -1),
    }
    write_json(frozen / "manifest.json", manifest)
    write_json(packages / "package_manifest.json", manifest)
    (packages / "package_checksums.txt").write_text("%s  %s\n" % (compute_sha256(zip_path), zip_path.name), encoding="utf-8")
    readiness = {
        "v3_coverage_extended_official": {
            "ready": ready, "selected_variant": selected, "track1_path": str(target), "zip_path": str(zip_path),
            "validation_errors": validation.get("num_errors", -1), "scene_ids": validation.get("scene_ids", []),
            "class_mapping": "official", "float_rounding_decimals": int(config.get("official_track1", {}).get("round_float_decimals", 2)),
            "rows": len(rows), "unique_tracks": unique_tracks, "zip_size_mb": zip_size_mb, "sha256": sha256,
        },
        "recommendation": "Upload only if official submission budget allows a third candidate after V2 official and V3 official.",
    }
    write_json(root / "frozen_candidate" / "comparison" / "upload_readiness.json", readiness)
    write_json(root / "frozen_candidate" / "comparison" / "verdict.json", {"label": "v3_coverage_extension_ready_for_upload" if ready else "v3_coverage_extension_invalid_fix_required"})
    return readiness
