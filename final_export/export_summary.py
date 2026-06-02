"""Summary helpers for final MVP export."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def summarize_propagation_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-camera global-id propagation rows."""
    assigned = sum([int(row.get("assigned_records", 0)) for row in rows])
    unassigned = sum([int(row.get("unassigned_records", 0)) for row in rows])
    input_records = sum([int(row.get("input_records", 0)) for row in rows])
    output_records = sum([int(row.get("output_records", 0)) for row in rows])
    return {
        "files": len(rows),
        "ok_files": len([row for row in rows if row.get("status") == "ok"]),
        "error_files": len([row for row in rows if row.get("status") == "error"]),
        "input_records": input_records,
        "output_records": output_records,
        "assigned_records": assigned,
        "unassigned_records": unassigned,
        "assignment_ratio": _ratio(assigned, assigned + unassigned),
        "unique_global_tracks": _sum_unique_global_tracks(rows),
        "per_subset": _sum_by(rows, "subset", "output_records"),
        "per_scene": _sum_by(rows, "scene_name", "output_records"),
        "per_camera": _sum_by(rows, "camera_id", "output_records"),
        "rows": rows,
    }


def summarize_generic_exports(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-scene generic export rows."""
    return {
        "files": len(rows),
        "rows_written": sum([int(row.get("rows_written", 0)) for row in rows]),
        "unique_global_tracks_sum": sum([int(row.get("unique_global_tracks", 0)) for row in rows]),
        "per_scene_rows": {str(row.get("scene_name", "")): int(row.get("rows_written", 0)) for row in rows},
        "rows": rows,
    }


def write_summary_json(summary: Dict[str, Any], path: Path) -> None:
    """Write summary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    """Write compact summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow({"metric": key, "value": value})


def print_final_export_summary(summary: Dict[str, Any]) -> None:
    """Print compact final export summary."""
    print("files: %s" % summary.get("files"))
    print("input_records: %s" % summary.get("input_records"))
    print("output_records: %s" % summary.get("output_records"))
    print("assigned_records: %s" % summary.get("assigned_records"))
    print("unassigned_records: %s" % summary.get("unassigned_records"))
    print("assignment_ratio: %s" % summary.get("assignment_ratio"))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _sum_unique_global_tracks(rows: List[Dict[str, Any]]) -> int:
    return sum([int(row.get("unique_global_tracks", 0)) for row in rows])


def _sum_by(rows: List[Dict[str, Any]], group_key: str, value_key: str) -> Dict[str, int]:
    output = {}
    for row in rows:
        key = str(row.get(group_key, ""))
        output[key] = output.get(key, 0) + int(row.get(value_key, 0))
    return output
