"""Build final baseline manifests without copying large artifacts."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.final_freeze.checksum_utils import compute_sha256
from deep_oc_sort_3d.final_freeze.freeze_config import output_root_from_config
from deep_oc_sort_3d.final_freeze.freeze_io import (
    NOT_AVAILABLE,
    count_text_rows,
    metric_value,
    progress_iter,
    read_json,
    write_csv_rows,
    write_json,
)


def build_final_outputs_manifest(config: Dict[str, Any], show_progress: bool = True) -> Dict[str, Any]:
    """Build final outputs manifest for key artifacts."""
    output_root = output_root_from_config(config)
    rows = []
    for baseline in progress_iter(_baseline_specs(config), show_progress, "build final manifest", "baseline"):
        rows.extend(manifest_rows_for_baseline(baseline))
    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "output_root": str(output_root),
        "files": rows,
    }
    write_json(manifest, output_root / "manifests" / "final_outputs_manifest.json")
    write_csv_rows(rows, output_root / "manifests" / "final_outputs_manifest.csv")
    _write_baseline_directories(config, rows, output_root)
    return manifest


def manifest_rows_for_baseline(baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return file manifest rows for one baseline spec."""
    rows = []
    for item in baseline.get("files", []):
        path = Path(str(item.get("path", "")))
        rows.append(file_manifest_row(baseline.get("name", ""), baseline.get("role", ""), item.get("kind", ""), path, bool(item.get("checksum", True))))
    return rows


def file_manifest_row(baseline_name: str, role: str, kind: str, path: Path, checksum: bool = True) -> Dict[str, Any]:
    """Return one file manifest row."""
    exists = path.exists()
    row: Dict[str, Any] = {
        "baseline_name": baseline_name,
        "role": role,
        "kind": kind,
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else NOT_AVAILABLE,
        "sha256": NOT_AVAILABLE,
        "track1_rows": count_text_rows(path) if path.name == "track1.txt" else NOT_AVAILABLE,
    }
    if exists and path.is_file() and checksum:
        row["sha256"] = compute_sha256(path)
    return row


def baseline_metadata_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize baseline paths, validation status, and row counts."""
    output = {}
    for spec in _baseline_specs(config):
        name = str(spec.get("name", ""))
        track1 = _file_by_kind(spec, "track1")
        validation = _file_by_kind(spec, "validation_report")
        report = read_json(validation) if validation is not None else {}
        output[name] = {
            "role": spec.get("role"),
            "track1_path": None if track1 is None else str(track1),
            "track1_rows": NOT_AVAILABLE if track1 is None else count_text_rows(track1),
            "validation_status": metric_value((report or {}).get("status")),
            "validation_errors": metric_value((report or {}).get("num_errors")),
        }
    return output


def _baseline_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths = config.get("paths", {})
    return [
        {
            "name": "v1_geometry_only",
            "role": "submission_safe_baseline",
            "files": [
                {"kind": "track1", "path": Path(str(paths.get("v1_track1_root", ""))) / "track1.txt"},
                {"kind": "validation_report", "path": _validation_path(Path(str(paths.get("v1_track1_root", ""))))},
                {"kind": "final_export_root", "path": Path(str(paths.get("v1_final_export_root", ""))), "checksum": False},
                {"kind": "global_root", "path": Path(str(paths.get("v1_global_root", ""))), "checksum": False},
            ],
        },
        {
            "name": "v2_pseudo3d_fullcam",
            "role": "provenance_3d_mvp",
            "files": [
                {"kind": "track1", "path": Path(str(paths.get("v2_track1_root", ""))) / "track1.txt"},
                {"kind": "validation_report", "path": _validation_path(Path(str(paths.get("v2_track1_root", ""))))},
                {"kind": "final_export_root", "path": Path(str(paths.get("v2_final_export_root", ""))), "checksum": False},
                {"kind": "global_root", "path": Path(str(paths.get("v2_global_root", ""))), "checksum": False},
            ],
        },
        {
            "name": "v2_export_compact",
            "role": "safe_compact_variant",
            "files": [
                {"kind": "track1", "path": Path(str(paths.get("v2_compact_root", paths.get("person_cleanup_root", "")))) / "runs" / "export_compact" / "track1_submission" / "track1.txt"},
                {"kind": "validation_report", "path": _validation_path(Path(str(paths.get("v2_compact_root", paths.get("person_cleanup_root", "")))) / "runs" / "export_compact" / "track1_submission")},
            ],
        },
        {
            "name": "reid_diagnostic",
            "role": "diagnostic_ablation",
            "files": [
                {"kind": "reid_summary", "path": Path(str(paths.get("reid_root", ""))) / "summaries" / "pipeline_summary.json"},
                {"kind": "reid_ablation_decision", "path": Path(str(paths.get("reid_ablation_root", ""))) / "comparison" / "final_variant_decision.json"},
            ],
        },
    ]


def _write_baseline_directories(config: Dict[str, Any], rows: List[Dict[str, Any]], output_root: Path) -> None:
    metadata = baseline_metadata_summary(config)
    for name, summary in metadata.items():
        baseline_rows = [row for row in rows if row.get("baseline_name") == name]
        root = output_root / "baselines" / name
        root.mkdir(parents=True, exist_ok=True)
        write_json({"summary": summary, "files": baseline_rows}, root / "baseline_summary.json")
        _write_baseline_readme(name, summary, root / "README.md")


def _write_baseline_readme(name: str, summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# %s" % name,
        "",
        "- Role: `%s`." % summary.get("role"),
        "- Track1 path: `%s`." % summary.get("track1_path"),
        "- Track1 rows: `%s`." % summary.get("track1_rows"),
        "- Validation status: `%s`." % summary.get("validation_status"),
        "- Validation errors: `%s`." % summary.get("validation_errors"),
        "",
        "This directory is a lightweight freeze pointer. It does not copy large intermediate outputs.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validation_path(root: Path) -> Path:
    for name in ["track1_validation_report.json", "validation_report.json", "final_checks/final_validation_report.json"]:
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / "track1_validation_report.json"


def _file_by_kind(spec: Dict[str, Any], kind: str) -> Optional[Path]:
    for item in spec.get("files", []):
        if item.get("kind") == kind:
            return Path(str(item.get("path", "")))
    return None
