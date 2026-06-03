"""Packaging helpers for final Track 1 submission artifacts."""

import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 checksum for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_submission_manifest(package_root: Path, files: List[Path], show_progress: bool = True) -> Dict[str, Any]:
    """Create a manifest for packaged files."""
    entries = []
    for path in _progress_iter(files, show_progress, "create manifest", "file"):
        if not path.exists() or not path.is_file():
            continue
        entries.append(
            {
                "path": str(path.relative_to(package_root)),
                "size_bytes": path.stat().st_size,
                "sha256": compute_file_sha256(path),
            }
        )
    return {
        "package_root": str(package_root),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "files": entries,
    }


def package_track1_submission(
    track1_path: Union[str, Path],
    output_package_root: Union[str, Path],
    config_paths: Optional[List[Union[str, Path]]] = None,
    reports: Optional[List[Union[str, Path]]] = None,
    baseline_name: str = "baseline_v1_geometry_only",
    overwrite: bool = False,
    make_zip: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Create a clean submission package folder."""
    source_track1 = Path(track1_path)
    package_root = Path(output_package_root)
    if not source_track1.exists():
        raise ValueError("Missing track1 file: %s" % source_track1)
    if package_root.exists() and overwrite:
        shutil.rmtree(str(package_root))
    if package_root.exists() and any(package_root.iterdir()) and not overwrite:
        raise ValueError("Package root exists and is not empty. Use overwrite=True: %s" % package_root)
    package_root.mkdir(parents=True, exist_ok=True)
    copied = []
    copied.append(_copy_file(source_track1, package_root / "track1.txt"))
    for path in _progress_iter(_path_list(config_paths), show_progress, "copy configs", "file"):
        if path.exists():
            copied.append(_copy_file(path, package_root / "configs" / path.name))
    for path in _progress_iter(_path_list(reports), show_progress, "copy reports", "file"):
        if path.exists():
            copied.append(_copy_file(path, package_root / "reports" / path.name))
    readme = package_root / "README_SUBMISSION.md"
    _write_package_readme(readme, baseline_name)
    copied.append(readme)
    manifest = create_submission_manifest(package_root, copied, show_progress=show_progress)
    manifest_path = package_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    copied.append(manifest_path)
    checksums_path = package_root / "checksums.txt"
    _write_checksums(package_root, copied, checksums_path, show_progress=show_progress)
    zip_path = None
    if make_zip:
        zip_path = _make_zip(package_root)
    return {
        "baseline_name": baseline_name,
        "package_root": str(package_root),
        "track1_path": str(package_root / "track1.txt"),
        "manifest_path": str(manifest_path),
        "checksums_path": str(checksums_path),
        "num_files": len(copied) + 1,
        "zip_path": None if zip_path is None else str(zip_path),
    }


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    return destination


def _path_list(values: Optional[List[Union[str, Path]]]) -> List[Path]:
    if values is None:
        return []
    return [Path(value) for value in values]


def _write_package_readme(path: Path, baseline_name: str) -> None:
    lines = [
        "# Submission Package",
        "",
        "- baseline_name: `%s`" % baseline_name,
        "- main_file: `track1.txt`",
        "- model: YOLO11m medium curriculum",
        "- association: geometry-only overlap + transition",
        "- ReID: no",
        "- neural 3D bbox head: no",
        "",
        "This package contains the validated Track 1 text file, configs, reports, manifest, and checksums.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checksums(package_root: Path, files: List[Path], output_path: Path, show_progress: bool) -> None:
    lines = []
    for path in _progress_iter(files, show_progress, "compute checksums", "file"):
        if path.exists() and path.is_file():
            lines.append("%s  %s" % (compute_file_sha256(path), path.relative_to(package_root)))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_zip(package_root: Path) -> Path:
    zip_path = package_root.with_suffix(".zip")
    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(str(path), arcname=str(path.relative_to(package_root)))
    return zip_path


def _progress_iter(values: List[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        print("%s: %d/%d %s" % (desc, index + 1, total, value))
        yield value
