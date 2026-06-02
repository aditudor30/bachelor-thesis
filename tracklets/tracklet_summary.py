"""Batch summaries for local tracklet files."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.tracklets.tracklet_eval import evaluate_tracklets
from deep_oc_sort_3d.tracklets.tracklet_io import read_tracklets_file


def summarize_tracklet_files(tracklet_files: List[Path]) -> Dict[str, Any]:
    """Summarize a collection of tracklet files without loading unrelated data."""
    file_rows = []
    all_tracklets = []
    warnings = []
    for path in tracklet_files:
        try:
            tracklets = read_tracklets_file(path)
            metrics = evaluate_tracklets(tracklets)
            file_rows.append(_file_row(path, metrics))
            all_tracklets.extend(tracklets)
        except Exception as exc:
            warnings.append("%s: %s" % (path, exc))
            file_rows.append({"path": str(path), "status": "error", "error_message": str(exc)})
    global_metrics = evaluate_tracklets(all_tracklets)
    return {
        "total_files": len(tracklet_files),
        "files": file_rows,
        "total_tracklets": global_metrics.get("num_tracklets", 0),
        "valid_tracklets": global_metrics.get("valid_for_mtmc", 0),
        "quality_flags": global_metrics.get("quality_flags", {}),
        "per_class": global_metrics.get("per_class_tracklets", {}),
        "per_scene": global_metrics.get("per_scene_tracklets", {}),
        "per_camera": global_metrics.get("per_camera_tracklets", {}),
        "mean_length": global_metrics.get("mean_length"),
        "median_length": global_metrics.get("median_length"),
        "purity_mean": global_metrics.get("purity_mean"),
        "no_3d_tracklets": global_metrics.get("no_3d_tracklets"),
        "warnings": warnings,
    }


def write_tracklet_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    """Write file-level tracklet summary rows as CSV."""
    rows = summary.get("files", [])
    fields = [
        "path",
        "status",
        "num_tracklets",
        "valid_for_mtmc",
        "mean_length",
        "median_length",
        "purity_mean",
        "no_3d_tracklets",
        "short_tracklets",
        "low_confidence_tracklets",
        "error_message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_tracklet_summary_json(summary: Dict[str, Any], path: Path) -> None:
    """Write global tracklet summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def print_tracklet_summary(summary: Dict[str, Any]) -> None:
    """Print compact tracklet summary."""
    print("total files: %s" % summary.get("total_files"))
    print("total tracklets: %s" % summary.get("total_tracklets"))
    print("valid tracklets: %s" % summary.get("valid_tracklets"))
    print("mean length: %s" % summary.get("mean_length"))
    print("median length: %s" % summary.get("median_length"))
    print("purity mean: %s" % summary.get("purity_mean"))
    print("quality flags: %s" % json.dumps(summary.get("quality_flags", {}), sort_keys=True))
    print("per class: %s" % json.dumps(summary.get("per_class", {}), sort_keys=True))
    warnings = summary.get("warnings", [])
    if warnings:
        print("warnings: %d" % len(warnings))
        for warning in warnings[:10]:
            print("  %s" % warning)


def _file_row(path: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path),
        "status": "ok",
        "num_tracklets": metrics.get("num_tracklets"),
        "valid_for_mtmc": metrics.get("valid_for_mtmc"),
        "mean_length": metrics.get("mean_length"),
        "median_length": metrics.get("median_length"),
        "purity_mean": metrics.get("purity_mean"),
        "no_3d_tracklets": metrics.get("no_3d_tracklets"),
        "short_tracklets": metrics.get("short_tracklets"),
        "low_confidence_tracklets": metrics.get("low_confidence_tracklets"),
        "error_message": "",
    }
