"""Create compact final packages for selected baselines and report assets."""

import shutil
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze.checksum_utils import checksum_rows_for_files, write_checksum_text
from deep_oc_sort_3d.final_freeze.freeze_config import output_root_from_config
from deep_oc_sort_3d.final_freeze.freeze_io import (
    copy_file_if_exists,
    load_yaml,
    progress_iter,
    write_csv_rows,
    write_json,
)


def package_final_outputs_from_config(config_path: Path, show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Build final freeze packages."""
    config = load_yaml(config_path)
    return package_final_outputs(config, show_progress=show_progress, overwrite=overwrite)


def package_final_outputs(config: Dict[str, Any], show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Create submission-safe and report asset packages."""
    output_root = output_root_from_config(config)
    package_root = output_root / "packages"
    specs = _package_specs(config, output_root)
    packages = []
    for spec in progress_iter(specs, show_progress, "package final outputs", "package"):
        packages.append(build_package(spec, package_root, show_progress=show_progress, overwrite=overwrite))
    write_json({"packages": packages}, package_root / "package_manifest.json")
    write_csv_rows(_flatten_package_rows(packages), package_root / "package_manifest.csv")
    return {"packages": packages}


def build_package(spec: Dict[str, Any], package_root: Path, show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Build one package from file entries."""
    package_name = str(spec.get("name", "package"))
    root = package_root / package_name
    if overwrite and root.exists():
        shutil.rmtree(str(root))
    root.mkdir(parents=True, exist_ok=True)
    copied_rows: List[Dict[str, Any]] = []
    for entry in spec.get("files", []):
        source = Path(str(entry.get("source", "")))
        relative = Path(str(entry.get("relative", source.name)))
        destination = root / relative
        copied = copy_file_if_exists(source, destination)
        copied_rows.append(
            {
                "package": package_name,
                "kind": entry.get("kind", ""),
                "source": str(source),
                "destination": "" if copied is None else str(copied),
                "exists": copied is not None,
            }
        )
    _write_package_readme(spec, root)
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    checksum_rows = checksum_rows_for_files(files, root, show_progress=show_progress)
    write_csv_rows(checksum_rows, root / "checksums.csv")
    write_checksum_text(checksum_rows, root / "CHECKSUMS.sha256")
    write_json({"package": package_name, "files": copied_rows, "checksums": checksum_rows}, root / "manifest.json")
    return {"name": package_name, "root": str(root), "files": copied_rows, "num_files": len(files)}


def _package_specs(config: Dict[str, Any], output_root: Path) -> List[Dict[str, Any]]:
    paths = config.get("paths", {})
    v1_track1_root = Path(str(paths.get("v1_track1_root", "")))
    v2_track1_root = Path(str(paths.get("v2_track1_root", "")))
    compact_root = Path(str(paths.get("v2_compact_root", paths.get("person_cleanup_root", "")))) / "runs" / "export_compact"
    return [
        {
            "name": "submission_safe_v1",
            "title": "Submission-safe V1 geometry-only baseline",
            "description": "Compact package with the validated Track1 file and validation report for the V1 submission-safe baseline.",
            "files": [
                {"kind": "track1", "source": v1_track1_root / "track1.txt", "relative": "track1.txt"},
                {"kind": "validation", "source": _validation_path(v1_track1_root), "relative": "validation_report.json"},
                {"kind": "final_table", "source": output_root / "tables" / "final_baseline_comparison.csv", "relative": "tables/final_baseline_comparison.csv"},
                {"kind": "report", "source": output_root / "reports" / "REPRODUCIBILITY.md", "relative": "reports/REPRODUCIBILITY.md"},
            ],
        },
        {
            "name": "mvp_3d_v2",
            "title": "V2 pseudo3D fullcam MVP",
            "description": "Package with the 3D provenance-backed MVP Track1 file, validation report, and final report summaries.",
            "files": [
                {"kind": "track1", "source": v2_track1_root / "track1.txt", "relative": "track1.txt"},
                {"kind": "validation", "source": _validation_path(v2_track1_root), "relative": "validation_report.json"},
                {"kind": "pseudo3d_table", "source": output_root / "tables" / "pseudo3d_summary.csv", "relative": "tables/pseudo3d_summary.csv"},
                {"kind": "final_report", "source": output_root / "reports" / "FINAL_BASELINE_FREEZE_REPORT.md", "relative": "reports/FINAL_BASELINE_FREEZE_REPORT.md"},
                {"kind": "reproducibility", "source": output_root / "reports" / "REPRODUCIBILITY.md", "relative": "reports/REPRODUCIBILITY.md"},
            ],
        },
        {
            "name": "safe_compact_v2",
            "title": "Safe compact V2 export variant",
            "description": "Package with the compact export variant retained as a safe optional baseline.",
            "files": [
                {"kind": "track1", "source": compact_root / "track1_submission" / "track1.txt", "relative": "track1.txt"},
                {"kind": "validation", "source": _validation_path(compact_root / "track1_submission"), "relative": "validation_report.json"},
                {"kind": "final_table", "source": output_root / "tables" / "final_baseline_comparison.csv", "relative": "tables/final_baseline_comparison.csv"},
            ],
        },
        {
            "name": "report_assets",
            "title": "Report-ready tables, figures, and manifests",
            "description": "Small package containing final freeze tables, copied figures, generated metric charts, and report drafts.",
            "files": _report_asset_entries(output_root),
        },
    ]


def _report_asset_entries(output_root: Path) -> List[Dict[str, Any]]:
    roots = [output_root / "tables", output_root / "figures", output_root / "reports", output_root / "manifests"]
    entries: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                entries.append({"kind": root.name, "source": path, "relative": str(path.relative_to(output_root))})
    return entries


def _validation_path(root: Path) -> Path:
    for name in ["track1_validation_report.json", "validation_report.json", "final_checks/final_validation_report.json"]:
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / "track1_validation_report.json"


def _flatten_package_rows(packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for package in packages:
        for row in package.get("files", []):
            rows.append(dict(row))
    return rows


def _write_package_readme(spec: Dict[str, Any], root: Path) -> None:
    lines = [
        "# %s" % spec.get("title", spec.get("name", "Package")),
        "",
        str(spec.get("description", "")),
        "",
        "This package is generated by Step 17A final freeze tooling. It intentionally contains compact artifacts only; large intermediate detections, videos, and H5 depth maps are not copied.",
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
