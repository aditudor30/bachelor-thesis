"""Comparison utilities for global tuning runs."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.global_tuning.tuning_config import load_sweep_config
from deep_oc_sort_3d.global_tuning.tuning_io import write_csv_rows, write_json
from deep_oc_sort_3d.global_tuning.tuning_metrics import collect_run_metrics, compute_metric_deltas
from deep_oc_sort_3d.global_tuning.tuning_selection import select_best_run


def compare_global_tuning_runs(
    sweep_config_path: Path,
    output_root: Optional[Path] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Compare all completed runs described by a sweep config."""
    sweep = load_sweep_config(sweep_config_path)
    paths = sweep.get("paths", {})
    if output_root is None:
        output_root = Path(paths.get("output_root", "output/global_tuning/debug"))
    comparison_root = output_root / "comparison"
    diagnostics_root = output_root / "diagnostics"
    v2 = collect_run_metrics(
        "v2_current",
        Path("."),
        global_root=_path_or_missing(paths.get("baseline_v2_current_global_root")),
        final_export_root=_path_or_missing(paths.get("baseline_v2_current_final_export_root")),
        track1_root=_path_or_missing(paths.get("baseline_v2_current_track1_root")),
    )
    v1 = collect_run_metrics(
        "v1_baseline",
        Path("."),
        global_root=_path_or_missing(paths.get("baseline_v1_global_root")),
        final_export_root=_path_or_missing(paths.get("baseline_v1_final_export_root")),
        track1_root=_path_or_missing(paths.get("baseline_v1_track1_root")),
    )
    run_metrics = []
    for run in sweep.get("runs", []):
        if not isinstance(run, dict):
            continue
        name = str(run.get("name", ""))
        if not name:
            continue
        run_root = output_root / "runs" / name
        metrics = collect_run_metrics(name, run_root)
        metrics.update(compute_metric_deltas(metrics, v2, "vs_v2"))
        metrics.update(compute_metric_deltas(metrics, v1, "vs_v1"))
        run_metrics.append(metrics)
    summary = {
        "v2_current": v2,
        "v1_baseline": v1,
        "runs": run_metrics,
    }
    selection = select_best_run(run_metrics, v2, sweep.get("selection_criteria", {}))
    summary["best_run_recommendation"] = selection
    _write_comparison_outputs(summary, comparison_root, diagnostics_root)
    return summary


def _write_comparison_outputs(summary: Dict[str, Any], comparison_root: Path, diagnostics_root: Path) -> None:
    runs = summary.get("runs", [])
    write_json(summary, comparison_root / "global_tuning_summary.json")
    write_csv_rows(_flatten_rows(runs), comparison_root / "global_tuning_summary.csv")
    write_csv_rows(_flatten_rows(runs), comparison_root / "per_run_metrics.csv")
    write_json(summary.get("best_run_recommendation", {}), comparison_root / "best_run_recommendation.json")
    write_csv_rows(_delta_rows(runs, "vs_v2"), comparison_root / "metric_deltas_vs_v2_current.csv")
    write_csv_rows(_delta_rows(runs, "vs_v1"), comparison_root / "metric_deltas_vs_v1.csv")
    write_csv_rows(_per_class_rows(runs), comparison_root / "per_class_metrics.csv")
    write_csv_rows(_per_scene_rows(runs), comparison_root / "per_scene_metrics.csv")
    write_csv_rows(_fragmentation_vs_false_merge_rows(runs), diagnostics_root / "fragmentation_vs_false_merge.csv")
    write_csv_rows(_rows_vs_purity_rows(runs), diagnostics_root / "rows_vs_purity.csv")
    write_csv_rows(_transition_rows(runs), diagnostics_root / "transition_edges_analysis.csv")
    write_csv_rows(_singleton_rows(runs), diagnostics_root / "singleton_global_tracks_analysis.csv")
    write_csv_rows(_short_rows(runs), diagnostics_root / "short_global_tracks_analysis.csv")
    write_csv_rows(_per_class_rows(runs), diagnostics_root / "per_class_fragmentation_tuning.csv")
    write_csv_rows(_per_scene_rows(runs), diagnostics_root / "per_scene_fragmentation_tuning.csv")


def _path_or_missing(value: Any) -> Path:
    if value in (None, ""):
        return Path("__missing_global_tuning_path__")
    return Path(str(value))


def _flatten_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, dict):
                continue
            flat[key] = value
        output.append(flat)
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
        per_class = row.get("per_class_tracks", {})
        per_class_fragmentation = row.get("per_class_fragmentation", {})
        per_class_purity = row.get("per_class_purity", {})
        if not isinstance(per_class, dict):
            continue
        for class_name, count in sorted(per_class.items()):
            output.append(
                {
                    "run_name": row.get("run_name"),
                    "class_name": class_name,
                    "global_tracks": count,
                    "fragmentation_approx": _dict_get(per_class_fragmentation, class_name),
                    "purity": _dict_get(per_class_purity, class_name),
                }
            )
    return output


def _per_scene_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        per_scene = row.get("per_scene_tracks", {})
        if not isinstance(per_scene, dict):
            continue
        for scene_name, count in sorted(per_scene.items()):
            output.append({"run_name": row.get("run_name"), "scene_name": scene_name, "global_tracks": count})
    return output


def _fragmentation_vs_false_merge_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "run_name": row.get("run_name"),
            "fragmentation_approx": row.get("fragmentation_approx"),
            "false_merge_rate": row.get("false_merge_rate"),
            "global_purity_mean": row.get("global_purity_mean"),
        }
        for row in rows
    ]


def _rows_vs_purity_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "run_name": row.get("run_name"),
            "track1_rows": row.get("track1_rows"),
            "generic_rows": row.get("generic_rows"),
            "global_purity_mean": row.get("global_purity_mean"),
        }
        for row in rows
    ]


def _transition_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "run_name": row.get("run_name"),
            "accepted_edges": row.get("accepted_edges"),
            "transition_edges_total": row.get("transition_edges_total"),
            "transition_edges_accepted": row.get("transition_edges_accepted"),
        }
        for row in rows
    ]


def _singleton_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "run_name": row.get("run_name"),
            "num_global_tracks": row.get("num_global_tracks"),
            "singleton_tracks": row.get("singleton_tracks"),
            "singleton_ratio": row.get("singleton_ratio"),
        }
        for row in rows
    ]


def _short_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "run_name": row.get("run_name"),
            "track1_unique_object_ids": row.get("track1_unique_object_ids"),
            "track1_short_object_ids": row.get("track1_short_object_ids"),
            "track1_rows_per_object_median": row.get("track1_rows_per_object_median"),
        }
        for row in rows
    ]


def _dict_get(value: Any, key: Any) -> Any:
    if not isinstance(value, dict):
        return ""
    return value.get(key, "")
