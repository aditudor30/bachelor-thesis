"""Separate official upload package builder for Step 22A."""

import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.official_023_027.official_config import VARIANT_KEYS, frozen_output_root, frozen_variant_root, output_root
from deep_oc_sort_3d.official_023_027.official_track1_io import compute_sha256, count_nonempty_lines, read_json, write_json


def package_official_candidates(
    config: Dict[str, Any],
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Create and verify one flat zip per official candidate."""
    package_root = frozen_output_root(config) / "packages"
    package_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANT_KEYS:
        rows.append(_package_one(config, variant, package_root, overwrite, skip_existing))
    manifest = {
        "status": "ok" if rows and all(row.get("status") == "ok" for row in rows) else "error",
        "packages": rows,
        "max_zip_size_mb": _max_zip_size(config),
    }
    write_json(package_root / "package_manifest.json", manifest)
    checksum_lines = ["%s  %s" % (row.get("zip_sha256"), Path(str(row.get("zip_path"))).name) for row in rows if row.get("zip_sha256")]
    (package_root / "package_checksums.txt").write_text("\n".join(checksum_lines) + ("\n" if checksum_lines else ""), encoding="utf-8")
    processing_packages = output_root(config) / "packages"
    processing_packages.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = Path(str(row.get("zip_path", "")))
        if source.exists():
            shutil.copy2(str(source), str(processing_packages / source.name))
    shutil.copy2(str(package_root / "package_manifest.json"), str(processing_packages / "package_manifest.json"))
    shutil.copy2(str(package_root / "package_checksums.txt"), str(processing_packages / "package_checksums.txt"))
    return manifest


def _package_one(config: Dict[str, Any], variant: str, package_root: Path, overwrite: bool, skip_existing: bool) -> Dict[str, Any]:
    name = "v2_current_official_023_027_track1.zip" if variant == "v2_current" else "v3_gap_aware_soft_official_023_027_track1.zip"
    source = frozen_variant_root(config, variant) / "track1.txt"
    validation_path = frozen_variant_root(config, variant) / "validation_summary.json"
    manifest_path = frozen_variant_root(config, variant) / "manifest.json"
    output = package_root / name
    if not source.exists() or source.stat().st_size <= 0:
        return {"variant": variant, "status": "missing_track1", "track1_path": str(source), "zip_path": str(output)}
    validation = read_json(validation_path)
    candidate_manifest = read_json(manifest_path)
    if validation.get("status") != "ok" or candidate_manifest.get("status") != "ready":
        return {
            "variant": variant,
            "status": "invalid_track1",
            "track1_path": str(source),
            "validation_path": str(validation_path),
            "manifest_path": str(manifest_path),
            "validation_errors": validation.get("num_errors"),
            "candidate_status": candidate_manifest.get("status"),
            "zip_path": str(output),
        }
    if output.exists() and not overwrite and not skip_existing:
        raise FileExistsError("Package exists; use --overwrite or --skip-existing: %s" % output)
    if output.exists() and overwrite:
        output.unlink()
    if not (output.exists() and skip_existing and not overwrite):
        with zipfile.ZipFile(str(output), "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(str(source), arcname="track1.txt")
    verification = _verify_zip(output, count_nonempty_lines(source))
    size_mb = float(output.stat().st_size) / (1024.0 * 1024.0) if output.exists() else 0.0
    within_limit = size_mb <= _max_zip_size(config)
    return {
        "variant": variant,
        "status": "ok" if verification.get("valid") and within_limit else "error",
        "track1_path": str(source),
        "zip_path": str(output),
        "zip_size_bytes": output.stat().st_size if output.exists() else 0,
        "zip_size_mb": size_mb,
        "max_zip_size_mb": _max_zip_size(config),
        "within_size_limit": within_limit,
        "zip_sha256": compute_sha256(output) if output.exists() else None,
        "verification": verification,
    }


def _verify_zip(path: Path, expected_lines: int) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"valid": False, "reason": "missing_or_empty_zip"}
    with zipfile.ZipFile(str(path), "r") as archive:
        names = archive.namelist()
        if names != ["track1.txt"]:
            return {"valid": False, "reason": "zip_must_contain_only_track1_txt", "names": names}
        line_count = 0
        with archive.open("track1.txt", "r") as handle:
            for line in handle:
                if line.strip():
                    line_count += 1
    return {"valid": line_count == expected_lines, "names": names, "line_count": line_count, "expected_line_count": expected_lines}


def _max_zip_size(config: Dict[str, Any]) -> float:
    return float(config.get("packaging", {}).get("max_zip_size_mb", 50.0))
