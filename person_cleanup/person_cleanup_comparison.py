"""Comparison utilities for Person cleanup runs."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_cleanup.person_cleanup_io import load_yaml, read_csv_rows, write_csv_rows, write_json
from deep_oc_sort_3d.person_cleanup.person_cleanup_metrics import collect_person_cleanup_metrics, compute_cleanup_deltas
from deep_oc_sort_3d.person_cleanup.person_cleanup_selection import select_best_person_cleanup


def compare_person_cleanup_runs(config_path: Path, output_root: Optional[Path] = None, progress: bool = True) -> Dict[str, Any]:
    """Compare completed Person cleanup runs."""
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    root = output_root
    if root is None:
        root = Path(config.get("person_cleanup", {}).get("output_root", paths.get("output_root", "output/person_cleanup/baseline_v2_pseudo3d_fullcam")))
    v2 = collect_person_cleanup_metrics(
        "v2_current",
        Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam"))),
        Path(str(paths.get("v2_track1_root", "output/track1_submission/baseline_v2_pseudo3d_fullcam"))),
        Path(str(paths.get("v2_global_root", "output/global_mtmc_transition/baseline_v2_pseudo3d_fullcam"))),
    )
    v1 = collect_person_cleanup_metrics(
        "v1_baseline",
        Path(str(paths.get("v1_final_export_root", "output/final_mvp_exports/yolo11m_medium_conf001_transition"))),
        Path(str(paths.get("v1_track1_root", "output/track1_submission/yolo11m_medium_conf001_transition"))),
        Path(str(paths.get("v1_global_root", "output/global_mtmc_transition/yolo11m_medium_conf001_transition"))),
    )
    runs = []
    for item in config.get("runs", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if not name:
            continue
        run_root = root / "runs" / name
        metrics = collect_person_cleanup_metrics(name, run_root / "final_export", run_root / "track1_submission", Path(str(paths.get("v2_global_root", ""))))
        metrics.update(compute_cleanup_deltas(metrics, v2, "vs_v2"))
        metrics.update(compute_cleanup_deltas(metrics, v1, "vs_v1"))
        runs.append(metrics)
    recommendation = select_best_person_cleanup(runs, v2, config.get("criteria", {}))
    summary = {"v2_current": v2, "v1_baseline": v1, "runs": runs, "best_person_cleanup_recommendation": recommendation}
    _write_outputs(summary, root)
    return summary


def _write_outputs(summary: Dict[str, Any], root: Path) -> None:
    comparison_root = root / "comparison"
    diagnostics_root = root / "diagnostics"
    runs = summary.get("runs", [])
    write_json(summary, comparison_root / "person_cleanup_summary.json")
    write_json(summary.get("best_person_cleanup_recommendation", {}), comparison_root / "best_person_cleanup_recommendation.json")
    write_csv_rows(_flat_rows(runs), comparison_root / "person_cleanup_summary.csv")
    write_csv_rows(_flat_rows(runs), comparison_root / "per_run_metrics.csv")
    write_csv_rows(_delta_rows(runs, "vs_v2"), comparison_root / "metric_deltas_vs_v2_current.csv")
    write_csv_rows(_delta_rows(runs, "vs_v1"), comparison_root / "metric_deltas_vs_v1.csv")
    write_csv_rows(_per_class_rows(runs), comparison_root / "per_class_metrics.csv")
    write_csv_rows(_per_scene_rows(runs), comparison_root / "per_scene_metrics.csv")
    write_csv_rows(_non_person_rows(summary), diagnostics_root / "non_person_impact_check.csv")
    write_csv_rows(_rows_vs_purity_rows(runs), diagnostics_root / "person_rows_vs_purity.csv")
    write_csv_rows(_fragmentation_rows(runs), diagnostics_root / "person_fragmentation_vs_false_merge.csv")
    _copy_run_report_csv(root, "dropped_tracks_analysis.csv", diagnostics_root / "dropped_tracks_analysis.csv")
    _copy_run_report_csv(root, "merged_tracks_analysis.csv", diagnostics_root / "merged_tracks_analysis.csv")
    write_csv_rows(_dropped_rows(summary), diagnostics_root / "dropped_rows_analysis.csv")


def _flat_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        output.append({key: value for key, value in row.items() if not isinstance(value, dict)})
    return output


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


def _per_scene_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        per_scene = row.get("per_scene_rows", {})
        if not isinstance(per_scene, dict):
            continue
        for scene_name, count in sorted(per_scene.items()):
            output.append({"run_name": row.get("run_name"), "scene_name": scene_name, "rows": count})
    return output


def _non_person_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    v2 = summary.get("v2_current", {})
    output = []
    for row in summary.get("runs", []):
        output.append(
            {
                "run_name": row.get("run_name"),
                "v2_non_person_rows": v2.get("non_person_rows"),
                "run_non_person_rows": row.get("non_person_rows"),
                "delta": row.get("vs_v2_non_person_rows_delta"),
            }
        )
    return output


def _rows_vs_purity_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "run_name": row.get("run_name"),
            "person_rows": row.get("person_rows"),
            "track1_rows": row.get("track1_rows"),
            "person_purity": row.get("person_purity"),
            "global_purity_mean": row.get("global_purity_mean"),
        }
        for row in rows
    ]


def _fragmentation_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "run_name": row.get("run_name"),
            "person_fragmentation_approx": row.get("person_fragmentation_approx"),
            "person_false_merge_rate": row.get("person_false_merge_rate"),
            "false_merge_rate": row.get("false_merge_rate"),
        }
        for row in rows
    ]


def _copy_run_report_csv(root: Path, filename: str, output_path: Path) -> None:
    rows = []
    for path in sorted((root / "runs").glob("*/summaries/%s" % filename)):
        run_name = path.parent.parent.name
        try:
            file_rows, _fields = read_csv_rows(path)
        except (OSError, ValueError):
            file_rows = []
        for row in file_rows:
            copied = dict(row)
            copied["run_name"] = run_name
            rows.append(copied)
    write_csv_rows(rows, output_path)


def _dropped_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for run in summary.get("runs", []):
        rows.append(
            {
                "run_name": run.get("run_name"),
                "person_rows_delta": run.get("vs_v2_person_rows_delta"),
                "non_person_rows_delta": run.get("vs_v2_non_person_rows_delta"),
                "track1_rows_delta": run.get("vs_v2_track1_rows_delta"),
            }
        )
    return rows
