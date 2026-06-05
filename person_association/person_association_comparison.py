"""Comparison utilities for Person-aware association runs."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_association.person_association_io import load_yaml, read_json, write_csv_rows, write_json
from deep_oc_sort_3d.person_association.person_association_metrics import (
    collect_person_association_metrics,
    compute_association_deltas,
)


DEFAULT_CRITERIA = {
    "require_track1_errors_zero": True,
    "require_non_person_unchanged": True,
    "max_false_merge_increase": 0.005,
    "max_purity_drop": 0.003,
    "min_person_fragmentation_reduction": 0.05,
}


def compare_person_association_runs(
    config_path: Path,
    output_root: Optional[Path] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Compare completed Person association runs with V2/V1 references."""
    _unused_progress = progress
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    root = output_root
    if root is None:
        root = Path(config.get("person_association", {}).get("output_root", paths.get("output_root", "output/person_association/baseline_v2_pseudo3d_fullcam")))
    v2 = collect_person_association_metrics(
        "v2_current",
        Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam"))),
        Path(str(paths.get("v2_track1_root", "output/track1_submission/baseline_v2_pseudo3d_fullcam"))),
    )
    v1 = collect_person_association_metrics(
        "v1_baseline",
        Path(str(paths.get("v1_final_export_root", "output/final_mvp_exports/yolo11m_medium_conf001_transition"))),
        Path(str(paths.get("v1_track1_root", "output/track1_submission/yolo11m_medium_conf001_transition"))),
    )
    export_compact_root = Path(str(paths.get("v2_export_compact_root", "output/person_cleanup/baseline_v2_pseudo3d_fullcam/runs/export_compact")))
    export_compact = {}
    if export_compact_root.exists():
        export_compact = collect_person_association_metrics(
            "v2_export_compact",
            export_compact_root / "final_export",
            export_compact_root / "track1_submission",
        )
    runs = []
    for item in config.get("runs", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if not name:
            continue
        run_root = root / "experiments" / name
        metrics = collect_person_association_metrics(
            name,
            run_root / "final_export",
            run_root / "track1_submission",
            run_root / "diagnostics" / "person_merge_summary.json",
        )
        metrics.update(compute_association_deltas(metrics, v2, "vs_v2"))
        metrics.update(compute_association_deltas(metrics, v1, "vs_v1"))
        if export_compact:
            metrics.update(compute_association_deltas(metrics, export_compact, "vs_export_compact"))
        runs.append(metrics)
    recommendation = select_best_person_association(runs, v2, config.get("criteria", {}))
    summary = {
        "v2_current": v2,
        "v1_baseline": v1,
        "v2_export_compact": export_compact,
        "runs": runs,
        "best_person_association_recommendation": recommendation,
    }
    _write_outputs(summary, root)
    return summary


def select_best_person_association(
    runs: List[Dict[str, Any]],
    v2_current: Dict[str, Any],
    criteria: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select and label the best Person-aware association run."""
    cfg = dict(DEFAULT_CRITERIA)
    if criteria:
        cfg.update(criteria)
    evaluated = []
    for run in runs:
        row = dict(run)
        row.update(compute_association_deltas(row, v2_current, "vs_v2"))
        row["accepted_by_selection_criteria"] = _acceptable(row, cfg)
        row["selection_score"] = _score(row)
        evaluated.append(row)
    acceptable = [row for row in evaluated if row.get("accepted_by_selection_criteria")]
    if not acceptable:
        verdict = "person_association_not_beneficial"
        if any((row.get("applied_merge_mapping_size") or 0) for row in evaluated):
            verdict = "person_association_too_risky"
        return {"verdict": verdict, "best_run": None, "runs": evaluated, "criteria": cfg}
    best = sorted(acceptable, key=lambda row: float(row.get("selection_score", -1e9)), reverse=True)[0]
    frag_reduction = best.get("vs_v2_person_fragmentation_reduction")
    if frag_reduction is None or float(frag_reduction) < float(cfg.get("min_person_fragmentation_reduction", 0.05)):
        verdict = "person_association_valid_but_minor_gain"
    else:
        verdict = "person_association_improves_v2"
    if not best.get("applied_merge_mapping_size"):
        verdict = "needs_reid_or_appearance_features"
    return {"verdict": verdict, "best_run": best.get("run_name"), "runs": evaluated, "criteria": cfg}


def _acceptable(row: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    if bool(criteria.get("require_track1_errors_zero", True)):
        if row.get("track1_validation_errors") not in (0, 0.0, "0"):
            return False
    if bool(criteria.get("require_non_person_unchanged", True)):
        delta = row.get("vs_v2_non_person_rows_delta")
        if delta is None or abs(float(delta)) > 0.0:
            return False
    purity_delta = row.get("vs_v2_global_purity_mean_delta")
    if purity_delta is not None and float(purity_delta) < -float(criteria.get("max_purity_drop", 0.003)):
        return False
    false_delta = row.get("vs_v2_false_merge_rate_delta")
    if false_delta is not None and float(false_delta) > float(criteria.get("max_false_merge_increase", 0.005)):
        return False
    return True


def _score(row: Dict[str, Any]) -> float:
    frag_reduction = float(row.get("vs_v2_person_fragmentation_reduction") or 0.0)
    track_reduction = -(float(row.get("vs_v2_person_unique_tracks_delta") or 0.0))
    purity_penalty = max(0.0, -(float(row.get("vs_v2_global_purity_mean_delta") or 0.0))) * 1000.0
    false_penalty = max(0.0, float(row.get("vs_v2_false_merge_rate_delta") or 0.0)) * 1000.0
    validation_penalty = 0.0 if row.get("track1_validation_errors") in (0, 0.0, "0") else 1e9
    return frag_reduction * 1000.0 + track_reduction - purity_penalty - false_penalty - validation_penalty


def _write_outputs(summary: Dict[str, Any], root: Path) -> None:
    comparison_root = root / "comparison"
    diagnostics_root = root / "diagnostics"
    runs = summary.get("runs", [])
    write_json(summary, comparison_root / "person_association_summary.json")
    write_json(summary.get("best_person_association_recommendation", {}), comparison_root / "best_person_association_recommendation.json")
    write_csv_rows(_flat_rows(runs), comparison_root / "person_association_summary.csv")
    write_csv_rows(_flat_rows(runs), comparison_root / "per_run_metrics.csv")
    write_csv_rows(_delta_rows(runs, "vs_v2"), comparison_root / "metric_deltas_vs_v2_current.csv")
    write_csv_rows(_delta_rows(runs, "vs_v1"), comparison_root / "metric_deltas_vs_v1.csv")
    write_csv_rows(_per_class_rows(runs), comparison_root / "per_class_metrics.csv")
    write_csv_rows(_non_person_rows(summary), diagnostics_root / "non_person_impact_check.csv")
    write_csv_rows(_merge_rows(root), diagnostics_root / "merge_summary_by_run.csv")


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
    for path in sorted((root / "experiments").glob("*/diagnostics/person_merge_summary.json")):
        run_name = path.parent.parent.name
        data = read_json(path) or {}
        row = {"run_name": run_name}
        row.update(data)
        rows.append(row)
    return rows

