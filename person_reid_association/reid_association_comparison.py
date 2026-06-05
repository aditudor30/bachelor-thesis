"""Comparison utilities for ReID-guided Person association runs."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_reid_association.reid_association_io import read_json, write_csv_rows, write_json
from deep_oc_sort_3d.person_reid_association.reid_association_metrics import (
    collect_reid_association_metrics,
    compute_reid_association_deltas,
)
from deep_oc_sort_3d.person_reid_association.reid_association_selection import select_best_reid_association
from deep_oc_sort_3d.person_reid_association.reid_association_io import load_yaml


def compare_reid_person_association_runs(
    config_path: Path,
    output_root: Optional[Path] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Compare completed ReID-guided Person association runs with reference baselines."""
    _unused_progress = progress
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    root = output_root
    if root is None:
        root = Path(
            str(
                config.get("person_reid_association", {}).get(
                    "output_root",
                    paths.get("output_root", "output/person_reid_association/baseline_v2_pseudo3d_fullcam"),
                )
            )
        )
    v2 = collect_reid_association_metrics(
        "v2_current",
        Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam"))),
        Path(str(paths.get("v2_track1_root", "output/track1_submission/baseline_v2_pseudo3d_fullcam"))),
        Path(str(paths.get("v2_diagnostics_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam/diagnostics"))),
    )
    v1 = collect_reid_association_metrics(
        "v1_baseline",
        Path(str(paths.get("v1_final_export_root", "output/final_mvp_exports/yolo11m_medium_conf001_transition"))),
        Path(str(paths.get("v1_track1_root", "output/track1_submission/yolo11m_medium_conf001_transition"))),
        Path(str(paths.get("v1_diagnostics_root", "output/final_mvp_exports/yolo11m_medium_conf001_transition/diagnostics"))),
    )
    export_compact = _optional_reference(
        "v2_export_compact",
        Path(str(paths.get("v2_export_compact_root", "output/person_cleanup/baseline_v2_pseudo3d_fullcam/runs/export_compact"))),
    )
    geometry_association = _optional_reference(
        "person_geometry_association",
        Path(str(paths.get("person_geometry_association_root", "output/person_association/baseline_v2_pseudo3d_fullcam/experiments/diagnostic_only"))),
    )
    runs = []
    for item in config.get("runs", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if not name:
            continue
        run_root = root / "runs" / name
        status = _run_status(run_root)
        if status.get("status") != "ok":
            runs.append(_missing_or_failed_run(name, status, v2))
            continue
        metrics = collect_reid_association_metrics(
            name,
            run_root / "final_export",
            run_root / "track1_submission",
            run_root / "diagnostics",
        )
        metrics["run_status"] = status.get("status")
        metrics.update(compute_reid_association_deltas(metrics, v2, "vs_v2"))
        metrics.update(compute_reid_association_deltas(metrics, v1, "vs_v1"))
        if export_compact:
            metrics.update(compute_reid_association_deltas(metrics, export_compact, "vs_export_compact"))
        runs.append(metrics)
    recommendation = select_best_reid_association(runs, v2, config.get("criteria", {}))
    summary = {
        "v2_current": v2,
        "v1_baseline": v1,
        "v2_export_compact": export_compact,
        "person_geometry_association": geometry_association,
        "runs": runs,
        "best_reid_person_association_recommendation": recommendation,
    }
    _write_outputs(summary, root)
    return summary


def _optional_reference(name: str, root: Path) -> Dict[str, Any]:
    if not root.exists():
        return {}
    return collect_reid_association_metrics(
        name,
        root / "final_export",
        root / "track1_submission",
        root / "diagnostics",
    )


def _write_outputs(summary: Dict[str, Any], root: Path) -> None:
    comparison_root = root / "comparison"
    diagnostics_root = root / "diagnostics"
    runs = summary.get("runs", [])
    write_json(summary, comparison_root / "reid_person_association_summary.json")
    write_json(summary.get("best_reid_person_association_recommendation", {}), comparison_root / "best_reid_person_association_recommendation.json")
    write_csv_rows(_flat_rows(runs), comparison_root / "reid_person_association_summary.csv")
    write_csv_rows(_flat_rows(runs), comparison_root / "per_run_metrics.csv")
    write_csv_rows(_delta_rows(runs, "vs_v2"), comparison_root / "metric_deltas_vs_v2_current.csv")
    write_csv_rows(_delta_rows(runs, "vs_v1"), comparison_root / "metric_deltas_vs_v1.csv")
    write_csv_rows(_per_class_rows(runs), comparison_root / "per_class_metrics.csv")
    write_csv_rows(_non_person_rows(summary), diagnostics_root / "non_person_impact_check.csv")
    write_csv_rows(_merge_rows(root), diagnostics_root / "reid_merge_summary_by_run.csv")


def _run_status(run_root: Path) -> Dict[str, Any]:
    data = read_json(run_root / "summaries" / "run_status.json")
    if data is not None:
        return data
    if not run_root.exists():
        return {"run_name": run_root.name, "status": "missing", "error_message": "run directory missing"}
    return {"run_name": run_root.name, "status": "missing_status", "error_message": "run_status.json missing"}


def _missing_or_failed_run(name: str, status: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "run_name": name,
        "run_status": status.get("status"),
        "error_message": status.get("error_message"),
        "accepted_by_selection_criteria": False,
        "selection_score": -1e9,
        "track1_validation_errors": None,
        "track1_rows": None,
        "generic_rows": None,
        "person_rows": None,
        "non_person_rows": None,
        "person_fragmentation_approx": None,
        "global_purity_mean": None,
        "false_merge_rate": None,
        "merges_applied": None,
        "candidate_pairs_generated": None,
        "pairs_with_both_reid": None,
        "pairs_missing_reid": None,
        "per_class_rows": {},
        "per_scene_rows": {},
    }
    row.update(compute_reid_association_deltas(row, v2, "vs_v2"))
    return row


def _flat_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in rows]


def _delta_rows(rows: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    output = []
    marker = "%s_" % prefix
    for row in rows:
        item = {"run_name": row.get("run_name")}
        for key, value in row.items():
            if key.startswith(marker):
                item[key] = value
        output.append(item)
    return output


def _per_class_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        per_class = row.get("per_class_rows", {})
        if not isinstance(per_class, dict):
            continue
        for class_name, count in sorted(per_class.items()):
            output.append({"run_name": row.get("run_name"), "class_name": class_name, "rows": count})
    return output


def _non_person_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    v2 = summary.get("v2_current", {})
    rows = []
    for run in summary.get("runs", []):
        rows.append(
            {
                "run_name": run.get("run_name"),
                "v2_non_person_rows": v2.get("non_person_rows"),
                "run_non_person_rows": run.get("non_person_rows"),
                "delta": run.get("vs_v2_non_person_rows_delta"),
            }
        )
    return rows


def _merge_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((root / "runs").glob("*/diagnostics/reid_merge_summary.json")):
        run_name = path.parent.parent.name
        data = read_json(path) or {}
        row = {"run_name": run_name}
        row.update(data)
        rows.append(row)
    return rows
