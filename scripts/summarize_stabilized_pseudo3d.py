"""Summarize Step 15D stabilized pseudo-3D outputs."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.audit3d.audit3d_io import read_json_if_exists, write_csv, write_json, write_markdown
from deep_oc_sort_3d.pseudo3d.pseudo3d_stabilization_report import build_stabilization_report


def run(args: Any) -> Dict[str, Any]:
    root = Path(args.root)
    smoothing = read_json_if_exists(root / "smoothing_reports" / "summary_smoothing_report.json")
    comparison = read_json_if_exists(root / "smoothness" / "raw_vs_stabilized_smoothness.json")
    stabilized_smoothness = read_json_if_exists(root / "smoothness" / "stabilized_smoothness_summary.json")
    eval_summary = read_json_if_exists(root / "evaluation" / "summary_eval.json")
    projection = read_json_if_exists(root / "projection_checks" / "stabilized_projection_summary.json")
    if not projection:
        projection = read_json_if_exists(root / "projection_checks" / "projection_summary.json")
    summary = dict(smoothing)
    summary.update(
        {
            "smoothness_comparison": comparison.get("smoothness_delta", {}),
            "stabilized_smoothness": stabilized_smoothness,
            "evaluation": _compact_eval(eval_summary),
            "projection": projection,
        }
    )
    delta = summary.get("smoothness_comparison", {})
    if isinstance(delta, dict):
        summary["stabilized_invalid_rate"] = delta.get("stabilized_invalid_rate")
    report_text = build_stabilization_report(summary)
    write_json(summary, root / "summaries" / "step15d_summary.json")
    write_csv(_summary_rows(summary), root / "summaries" / "step15d_summary.csv")
    write_markdown(report_text, root / "report" / "PSEUDO3D_STABILIZATION_REPORT.md")
    write_json(summary, root / "report" / "PSEUDO3D_STEP15D_SUMMARY.json")
    print("Step 15D summary written to %s" % (root / "report" / "PSEUDO3D_STABILIZATION_REPORT.md"))
    return summary


def _compact_eval(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not summary:
        return {}
    return {
        "num_predictions": summary.get("num_predictions"),
        "num_evaluated": summary.get("num_evaluated"),
        "num_missing_gt": summary.get("num_missing_gt"),
        "center_error": _key_stats(summary.get("center_error", {})),
        "depth_error": _key_stats(summary.get("depth_error", {})),
        "dimension_error": _key_stats(summary.get("dimension_error", {})),
        "yaw_error": _key_stats(summary.get("yaw_error", {})),
        "projection_success_rate": summary.get("projection_success_rate"),
    }


def _key_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    return {"median": stats.get("median"), "p95": stats.get("p95"), "p99": stats.get("p99"), "max": stats.get("max")}


def _summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key in ("num_predictions", "success_rate", "num_tracks", "num_center_smoothed", "num_depth_smoothed", "num_jump_corrected", "num_small_bbox_guarded", "stabilized_invalid_rate"):
        rows.append({"metric": key, "value": summary.get(key)})
    delta = summary.get("smoothness_comparison", {})
    if isinstance(delta, dict):
        for key, value in delta.items():
            rows.append({"metric": "smoothness_%s" % key, "value": value})
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize stabilized pseudo-3D outputs.")
    parser.add_argument("--root", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
