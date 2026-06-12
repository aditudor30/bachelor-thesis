"""Build and verify separate upload zip packages."""

import zipfile
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.freeze_upload.freeze_config import candidate_output_root, candidate_specs, output_root
from deep_oc_sort_3d.freeze_upload.freeze_io import compute_sha256, count_nonempty_lines, progress_iter, write_json
from deep_oc_sort_3d.freeze_upload.upload_manifest import build_package_manifest


def package_frozen_candidates(
    config: Dict[str, Any],
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
    create_zip: bool = True,
) -> Dict[str, Any]:
    """Create one flat zip per candidate and verify its content."""
    package_root = output_root(config) / "packages"
    package_root.mkdir(parents=True, exist_ok=True)
    packages = []
    for spec in progress_iter(candidate_specs(config), progress, "package frozen candidates"):
        packages.append(_package_one(config, spec, package_root, overwrite, skip_existing, create_zip))
    manifest = build_package_manifest(packages)
    write_json(package_root / "package_manifest.json", manifest)
    lines = []
    for package in packages:
        if package.get("zip_sha256"):
            lines.append("%s  %s" % (package.get("zip_sha256"), Path(str(package.get("zip_path"))).name))
    (package_root / "package_checksums.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest


def _package_one(
    config: Dict[str, Any],
    spec: Dict[str, Any],
    package_root: Path,
    overwrite: bool,
    skip_existing: bool,
    create_zip: bool,
) -> Dict[str, Any]:
    name = str(spec.get("candidate_name"))
    track1 = candidate_output_root(config, name) / "track1.txt"
    zip_path = package_root / (name + "_track1.zip")
    if not track1.exists() or track1.stat().st_size <= 0:
        return {"candidate_name": name, "status": "missing_frozen_track1", "zip_path": str(zip_path)}
    if not create_zip:
        return {"candidate_name": name, "status": "zip_disabled", "track1_path": str(track1), "zip_path": None}
    if zip_path.exists() and not overwrite and not skip_existing:
        raise FileExistsError("Package already exists; use --overwrite or --skip-existing: %s" % zip_path)
    if zip_path.exists() and overwrite:
        zip_path.unlink()
    if not (zip_path.exists() and skip_existing and not overwrite):
        with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(str(track1), arcname="track1.txt")
    verification = _verify_zip(zip_path, count_nonempty_lines(track1))
    return {
        "candidate_name": name,
        "status": "ok" if verification.get("valid") else "error",
        "track1_path": str(track1),
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
        "zip_sha256": compute_sha256(zip_path) if zip_path.exists() else None,
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
    return {
        "valid": line_count == expected_lines,
        "names": ["track1.txt"],
        "line_count": line_count,
        "expected_line_count": expected_lines,
    }
