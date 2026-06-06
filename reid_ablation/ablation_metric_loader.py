"""Load metrics for ReID ablation variants."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.person_association.person_association_metrics import collect_person_association_metrics
from deep_oc_sort_3d.person_reid_association.reid_association_metrics import collect_reid_association_metrics
from deep_oc_sort_3d.reid_ablation.ablation_io import (
    NOT_AVAILABLE,
    load_yaml,
    progress_iter,
    read_json,
    safe_int,
    write_csv_rows,
    write_json,
)


def collect_reid_ablation_metrics_from_config(config_path: Path, progress: bool = True) -> Dict[str, Any]:
    """Collect all configured ablation metrics and write collected outputs."""
    config = load_yaml(config_path)
    output_root = Path(str(config.get("reid_ablation_decision", {}).get("output_root", "output/reid_ablation_decision/baseline_v2_pseudo3d_fullcam")))
    rows = collect_variant_metrics(config, progress=progress)
    write_json({"variants": rows}, output_root / "collected_metrics" / "variant_metrics.json")
    write_csv_rows(rows, output_root / "collected_metrics" / "variant_metrics.csv")
    return {"output_root": str(output_root), "variants": rows}


def collect_variant_metrics(config: Dict[str, Any], progress: bool = True) -> List[Dict[str, Any]]:
    """Collect metric rows for all configured variants."""
    rows = []
    for variant in progress_iter(config.get("variants", []), progress, "collect ReID ablation metrics", "variant"):
        if not isinstance(variant, dict):
            continue
        rows.append(load_variant_metric(variant, config))
    return rows


def load_variant_metric(variant: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Load one variant metric row."""
    name = str(variant.get("name", ""))
    source_type = str(variant.get("type", ""))
    paths = config.get("paths", {})
    final_root, track1_root, diagnostics_root = resolve_variant_roots(name, source_type, paths)
    if final_root is None or track1_root is None or not final_root.exists():
        return make_variant_metric_row(name, source_type, {}, {"available": False, "missing_reason": "final_export_root_missing"})
    try:
        if source_type in ("reid_only", "reid_plus_compact"):
            metrics = collect_reid_association_metrics(name, final_root, track1_root, diagnostics_root or final_root.parent / "diagnostics")
        else:
            merge_summary = None if diagnostics_root is None else _merge_summary_path(source_type, diagnostics_root)
            metrics = collect_person_association_metrics(name, final_root, track1_root, merge_summary)
    except Exception as exc:
        return make_variant_metric_row(name, source_type, {}, {"available": False, "missing_reason": str(exc)})
    extras = collect_variant_extras(name, source_type, final_root, track1_root, diagnostics_root, paths)
    extras["available"] = True
    return make_variant_metric_row(name, source_type, metrics, extras)


def make_variant_metric_row(
    variant_name: str,
    source_type: str,
    metrics: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert raw metrics into the common ablation schema."""
    extra = extras or {}
    row = {
        "variant_name": variant_name,
        "source_type": source_type,
        "available": bool(extra.get("available", True)),
        "missing_reason": extra.get("missing_reason", ""),
        "track1_valid": _track1_valid(metrics),
        "track1_errors": _value(metrics.get("track1_validation_errors")),
        "track1_rows": _value(metrics.get("track1_rows")),
        "person_rows": _value(metrics.get("person_rows")),
        "non_person_rows": _value(metrics.get("non_person_rows")),
        "global_purity": _value(metrics.get("global_purity_mean")),
        "person_purity": _value(metrics.get("person_purity")),
        "false_merge_rate": _value(metrics.get("false_merge_rate")),
        "fragmentation_approx": _value(metrics.get("fragmentation_approx")),
        "person_fragmentation": _value(metrics.get("person_fragmentation_approx")),
        "num_reid_merges": _value(extra.get("num_reid_merges", metrics.get("merges_applied", 0))),
        "num_geometry_merges": _value(extra.get("num_geometry_merges", metrics.get("applied_merge_mapping_size", 0))),
        "num_export_dropped_tracks": _value(extra.get("num_export_dropped_tracks", 0)),
        "num_export_dropped_rows": _value(extra.get("num_export_dropped_rows", 0)),
        "candidate_pairs": _value(metrics.get("candidate_pairs_generated", metrics.get("candidate_rows"))),
        "pairs_with_reid": _value(metrics.get("pairs_with_both_reid")),
        "pairs_passing_reid_threshold": _value(metrics.get("pairs_passing_reid_threshold")),
        "final_export_root": str(extra.get("final_export_root", metrics.get("final_export_root", ""))),
        "track1_root": str(extra.get("track1_root", metrics.get("track1_root", ""))),
    }
    return row


def collect_variant_extras(
    name: str,
    source_type: str,
    final_root: Path,
    track1_root: Path,
    diagnostics_root: Optional[Path],
    paths: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect source-specific counts that are not in standard metrics."""
    _unused_paths = paths
    extras: Dict[str, Any] = {"final_export_root": str(final_root), "track1_root": str(track1_root)}
    run_root = final_root.parent
    if source_type in ("export_compact", "reid_plus_compact"):
        cleanup = read_json(run_root / "summaries" / "person_cleanup_export_summary.json") or {}
        extras.update(_cleanup_counts(cleanup))
    if source_type == "reid_only" or source_type == "reid_plus_compact":
        merge = read_json(run_root / "diagnostics" / "reid_merge_summary.json") or {}
        extras["num_reid_merges"] = merge.get("mapping_size", 0)
        extras["num_geometry_merges"] = 0
    elif source_type == "geometry_only":
        merge = read_json(run_root / "diagnostics" / "person_merge_summary.json") or {}
        extras["num_geometry_merges"] = merge.get("mapping_size", 0)
        extras["num_reid_merges"] = 0
    else:
        extras["num_reid_merges"] = 0
        extras["num_geometry_merges"] = 0
    if diagnostics_root is not None:
        extras["diagnostics_root"] = str(diagnostics_root)
    return extras


def resolve_variant_roots(name: str, source_type: str, paths: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Resolve final export, Track1, and diagnostics roots for a variant."""
    if source_type == "v1":
        return (
            Path(str(paths.get("v1_final_export_root", ""))),
            Path(str(paths.get("v1_track1_root", ""))),
            None,
        )
    if source_type == "v2_current":
        return (
            Path(str(paths.get("v2_final_export_root", ""))),
            Path(str(paths.get("v2_track1_root", ""))),
            None,
        )
    if source_type == "export_compact":
        root = Path(str(paths.get("person_cleanup_root", ""))) / "runs" / "export_compact"
        return (root / "final_export", root / "track1_submission", root / "summaries")
    if source_type == "geometry_only":
        root = _geometry_run_root(paths, name)
        return (root / "final_export", root / "track1_submission", root / "diagnostics")
    if source_type == "reid_only" or source_type == "reid_plus_compact":
        root = Path(str(paths.get("person_reid_association_root", ""))) / "runs" / name
        return (root / "final_export", root / "track1_submission", root / "diagnostics")
    return (None, None, None)


def _geometry_run_root(paths: Dict[str, Any], variant_name: str) -> Path:
    root = Path(str(paths.get("person_geometry_association_root", "")))
    if variant_name != "person_geometry_only_best":
        return root / "experiments" / variant_name
    recommendation = read_json(root / "comparison" / "best_person_association_recommendation.json") or {}
    best = recommendation.get("best_run")
    if best in (None, "", "None"):
        best = "diagnostic_only"
    return root / "experiments" / str(best)


def _merge_summary_path(source_type: str, diagnostics_root: Path) -> Optional[Path]:
    if source_type == "geometry_only":
        return diagnostics_root / "person_merge_summary.json"
    return None


def _cleanup_counts(summary: Dict[str, Any]) -> Dict[str, Any]:
    generic = summary.get("generic_report", {}) if isinstance(summary.get("generic_report"), dict) else {}
    frame = summary.get("frame_report", {}) if isinstance(summary.get("frame_report"), dict) else {}
    return {
        "num_export_dropped_tracks": summary.get("drop_keys", 0),
        "num_export_dropped_rows": generic.get("rows_dropped", frame.get("rows_dropped", 0)),
    }


def _track1_valid(metrics: Dict[str, Any]) -> Any:
    errors = safe_int(metrics.get("track1_validation_errors"), None)
    if errors is None:
        return NOT_AVAILABLE
    return errors == 0


def _value(value: Any) -> Any:
    if value in (None, ""):
        return NOT_AVAILABLE
    return value
