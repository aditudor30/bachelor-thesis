"""Compare baseline_v1_geometry_only and baseline_v2_pseudo3d outputs."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.audit3d.audit3d_io import iter_data_files, read_json_if_exists, write_csv, write_json, write_markdown


def compare_baseline_v1_v2(
    baseline_v1_root_paths: Dict[str, Any],
    baseline_v2_root_paths: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare available baseline_v1 and baseline_v2 artifacts."""
    v1 = _collect_baseline_metrics(baseline_v1_root_paths)
    v2 = _collect_baseline_metrics(baseline_v2_root_paths)
    return {
        "baseline_v1": v1,
        "baseline_v2": v2,
        "deltas": _deltas(v1, v2),
        "assessment": _assessment(v1, v2),
    }


def write_comparison_report_md(summary: Dict[str, Any], path: Path) -> None:
    """Write Markdown baseline comparison report."""
    write_markdown(_report_text(summary), path)


def write_comparison_json(summary: Dict[str, Any], path: Path) -> None:
    """Write comparison summary JSON."""
    write_json(summary, path)


def write_comparison_csv(summary: Dict[str, Any], path: Path) -> None:
    """Write comparison summary CSV."""
    rows = []
    for metric, value in summary.get("deltas", {}).items():
        rows.append({"metric": metric, "value": value})
    write_csv(rows, path)


def _collect_baseline_metrics(paths: Dict[str, Any]) -> Dict[str, Any]:
    track1_root = Path(paths.get("track1_root", ""))
    final_root = Path(paths.get("final_export_root", ""))
    global_root = Path(paths.get("global_mtmc_root", ""))
    observations_summary = read_json_if_exists(paths.get("observations_summary", ""))
    validation = _first_json([track1_root / "track1_validation.json", track1_root / "validation.json"])
    track1_path = track1_root / "track1.txt"
    generic_root = final_root / "generic_tracking_export"
    return {
        "track1_rows": _line_count(track1_path),
        "track1_validation_errors": validation.get("num_errors"),
        "track1_validation_status": validation.get("status"),
        "generic_rows": _csv_row_count(generic_root),
        "global_scene_count": _scene_count(global_root),
        "source_metadata_completeness": observations_summary.get("source_metadata_completeness", {}),
        "pseudo3d_used_rate": observations_summary.get("pseudo3d_used_rate"),
        "fallback_original_used": observations_summary.get("fallback_original_used"),
        "no_3d_records": observations_summary.get("no_3d_records"),
        "per_class": _generic_class_counts(generic_root),
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _deltas(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "track1_rows_delta": _delta(v1.get("track1_rows"), v2.get("track1_rows")),
        "generic_rows_delta": _delta(v1.get("generic_rows"), v2.get("generic_rows")),
        "track1_validation_errors_delta": _delta(v1.get("track1_validation_errors"), v2.get("track1_validation_errors")),
        "v2_pseudo3d_used_rate": v2.get("pseudo3d_used_rate"),
        "v2_fallback_original_used": v2.get("fallback_original_used"),
        "v2_no_3d_records": v2.get("no_3d_records"),
    }


def _assessment(v1: Dict[str, Any], v2: Dict[str, Any]) -> str:
    errors = v2.get("track1_validation_errors")
    if errors in (0, "0"):
        return "baseline_v2 produced a Track1-valid export; compare tracking quality before treating it as better than baseline_v1."
    if errors is None:
        return "baseline_v2 comparison is incomplete because Track1 validation was not found."
    return "baseline_v2 is not submission-ready yet because Track1 validation has errors."


def _report_text(summary: Dict[str, Any]) -> str:
    v1 = summary.get("baseline_v1", {})
    v2 = summary.get("baseline_v2", {})
    deltas = summary.get("deltas", {})
    return "\n".join(
        [
            "# Baseline V1 vs V2 Comparison",
            "",
            "## Track1",
            "",
            "- v1 rows: %s" % v1.get("track1_rows"),
            "- v2 rows: %s" % v2.get("track1_rows"),
            "- v1 validation errors: %s" % v1.get("track1_validation_errors"),
            "- v2 validation errors: %s" % v2.get("track1_validation_errors"),
            "",
            "## Provenance",
            "",
            "- v2 pseudo3D used rate: %s" % v2.get("pseudo3d_used_rate"),
            "- v2 fallback original used: %s" % v2.get("fallback_original_used"),
            "- v2 no-3D records: %s" % v2.get("no_3d_records"),
            "",
            "## Deltas",
            "",
            "- Track1 rows delta: %s" % deltas.get("track1_rows_delta"),
            "- Generic rows delta: %s" % deltas.get("generic_rows_delta"),
            "",
            "## Assessment",
            "",
            str(summary.get("assessment", "")),
            "",
        ]
    )


def _first_json(paths: List[Path]) -> Dict[str, Any]:
    for path in paths:
        data = read_json_if_exists(path)
        if data:
            return data
    return {}


def _line_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _csv_row_count(root: Path) -> int:
    total = 0
    for path in iter_data_files(root, [".csv"]):
        total += len(_read_csv(path))
    return total


def _scene_count(root: Path) -> int:
    if not root.exists():
        return 0
    return len([path for path in root.rglob("*") if path.is_dir() and path.name.startswith("Warehouse_")])


def _generic_class_counts(root: Path) -> Dict[str, int]:
    counts = {}
    for path in iter_data_files(root, [".csv"]):
        for row in _read_csv(path):
            key = str(row.get("class_id", row.get("class_name", "")))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (IOError, OSError):
        return []


def _delta(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None

