"""Comparable metrics for learned Person association sweep variants."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deep_oc_sort_3d.person_association.person_association_metrics import (
    collect_person_association_metrics,
)
from deep_oc_sort_3d.learned_association_application.scorer_association_io import (
    read_csv_rows,
    read_json,
    safe_float,
    safe_int,
)


METRIC_FIELDS = (
    "track1_valid", "track1_errors", "track1_rows", "global_tracks", "person_global_tracks",
    "multi_camera_tracks", "person_multi_camera_tracks", "singleton_tracks", "person_singleton_tracks",
    "accepted_edges", "mlp_edges_accepted", "reid_edges_accepted", "transition_edges_accepted",
    "global_purity_mean", "person_purity_mean", "false_merge_rate", "person_false_merge_rate",
    "fragmentation_approx", "person_fragmentation", "non_person_rows", "duplicate_keys",
    "nan_inf_count", "non_positive_dimensions",
)


def collect_variant_metrics(run_name: str, run_root: Path) -> Dict[str, Any]:
    """Collect existing project metrics and Step 20C-specific counters."""
    merge_summary_path = run_root / "merge_summary.json"
    base = collect_person_association_metrics(
        run_name,
        run_root / "final_export",
        run_root / "track1_submission",
        merge_summary_path,
    )
    validation = read_json(run_root / "validation" / "track1_validation_report.json") or {}
    merge = read_json(merge_summary_path) or {}
    person_multi, singleton, person_singleton = _track_counts(run_root / "final_export" / "generic_tracking_export")
    metrics = dict(base)
    metrics.update(
        {
            "track1_valid": str(validation.get("status", "")).lower() == "ok" and int(validation.get("num_errors") or 0) == 0,
            "track1_errors": validation.get("num_errors"),
            "global_tracks": base.get("global_unique_tracks"),
            "person_global_tracks": base.get("person_unique_tracks"),
            "person_multi_camera_tracks": person_multi,
            "singleton_tracks": singleton,
            "person_singleton_tracks": person_singleton,
            "accepted_edges": merge.get("accepted_edges"),
            "mlp_edges_accepted": merge.get("accepted_edges"),
            "reid_edges_accepted": merge.get("accepted_edges_with_reid", merge.get("accepted_edges")),
            "transition_edges_accepted": merge.get("transition_edges_accepted"),
            "person_purity_mean": base.get("person_purity"),
            "person_fragmentation": base.get("person_fragmentation_approx"),
            "duplicate_keys": validation.get("duplicate_key_count"),
            "nan_inf_count": validation.get("nan_or_inf_values"),
            "non_positive_dimensions": validation.get("non_positive_dimensions"),
        }
    )
    return metrics


def collect_baseline_metrics(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Collect four requested baselines when their artifacts are available."""
    paths = config.get("paths", {})
    specs = {
        "v2_current": (
            Path(str(paths.get("v2_final_export_root"))),
            Path(str(paths.get("v2_track1_root"))),
            Path(str(paths.get("v2_global_root", ""))) / "summaries" / "run_metrics.json",
        ),
        "v2_export_compact": _discover_existing_run(Path(str(paths.get("export_compact_root", ""))), "export_compact"),
        "threshold_080": _discover_existing_run(Path(str(paths.get("reid_finetuned_association_root", ""))) / "sweep_runs", "threshold_080"),
        "combined_safe_080": _discover_existing_run(Path(str(paths.get("reid_finetuned_association_root", ""))) / "sweep_runs", "combined_safe_080"),
    }
    result = {}
    for name, spec in specs.items():
        if spec is None:
            result[name] = {"run_name": name, "status": "not_available"}
            continue
        final_root, track1_root, merge_path = spec
        if not final_root.exists():
            result[name] = {"run_name": name, "status": "not_available"}
            continue
        base = collect_person_association_metrics(name, final_root, track1_root, merge_path if merge_path.exists() else None)
        validation = _load_validation(track1_root)
        base.update(
            {
                "status": "ok",
                "track1_valid": str(validation.get("status", "")).lower() == "ok" and int(validation.get("num_errors") or 0) == 0,
                "track1_errors": validation.get("num_errors"),
                "global_tracks": base.get("global_unique_tracks"),
                "person_global_tracks": base.get("person_unique_tracks"),
                "person_purity_mean": base.get("person_purity"),
                "person_fragmentation": base.get("person_fragmentation_approx"),
            }
        )
        result[name] = base
    return result


def metric_deltas(run: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compute signed numeric deltas for selection and reports."""
    result = {}
    for key in METRIC_FIELDS:
        left = _number(run.get(key))
        right = _number(baseline.get(key))
        result["%s_delta" % key] = None if left is None or right is None else left - right
    return result


def build_comparison_rows(
    variants: Sequence[Dict[str, Any]], baselines: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Attach requested baseline deltas to each variant metrics row."""
    rows = []
    for variant in variants:
        row = dict(variant)
        for baseline_name in ("v2_current", "combined_safe_080"):
            for key, value in metric_deltas(variant, baselines.get(baseline_name, {})).items():
                row["vs_%s_%s" % (baseline_name, key)] = value
        rows.append(row)
    return rows


def _discover_existing_run(root: Path, name: str) -> Optional[Tuple[Path, Path, Path]]:
    candidates = [root / name, root / "experiments" / name, root / "runs" / name]
    for candidate in candidates:
        final_root = candidate / "final_export"
        if final_root.exists():
            return (final_root, candidate / "track1_submission", candidate / "diagnostics" / "person_merge_summary.json")
    return None


def _track_counts(generic_root: Path) -> Tuple[int, int, int]:
    cameras = {}  # type: Dict[Tuple[str, str, str, str], set]
    counts = {}  # type: Dict[Tuple[str, str, str, str], int]
    if not generic_root.exists():
        return 0, 0, 0
    for path in generic_root.rglob("*.csv"):
        subset = path.parent.name
        for row in read_csv_rows(path):
            key = (subset, str(row.get("scene_name", "")), str(row.get("class_id", "")), str(row.get("global_track_id", "")))
            counts[key] = counts.get(key, 0) + 1
            cameras.setdefault(key, set()).add(str(row.get("camera_id", "")))
    person_multi = len([key for key, values in cameras.items() if key[2] == "0" and len(values) > 1])
    singleton = len([value for value in counts.values() if value <= 1])
    person_singleton = len([value for key, value in counts.items() if key[2] == "0" and value <= 1])
    return person_multi, singleton, person_singleton


def _load_validation(track1_root: Path) -> Dict[str, Any]:
    return read_json(track1_root / "track1_validation_report.json") or {}


def _number(value: Any) -> Optional[float]:
    return safe_float(value, None)
