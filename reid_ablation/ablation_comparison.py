"""Compare ReID ablation variants against V2 current."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.reid_ablation.ablation_io import (
    NOT_AVAILABLE,
    load_yaml,
    safe_float,
    safe_int,
    write_csv_rows,
    write_json,
)
from deep_oc_sort_3d.reid_ablation.ablation_metric_loader import collect_variant_metrics


DEFAULT_THRESHOLDS = {
    "max_false_merge_increase": 0.005,
    "max_purity_drop": 0.003,
    "min_fragmentation_reduction_for_upgrade": 0.01,
    "min_track1_rows_reduction_for_compact": 1000,
    "require_non_person_unchanged": True,
    "require_track1_valid": True,
}


def build_reid_ablation_comparison_from_config(config_path: Path, progress: bool = True) -> Dict[str, Any]:
    """Collect and compare all variants from a config file."""
    config = load_yaml(config_path)
    output_root = Path(str(config.get("reid_ablation_decision", {}).get("output_root", "output/reid_ablation_decision/baseline_v2_pseudo3d_fullcam")))
    rows = collect_variant_metrics(config, progress=progress)
    comparison = compare_reid_ablation_variants(rows, config.get("decision_thresholds", {}))
    write_json({"variants": rows}, output_root / "collected_metrics" / "variant_metrics.json")
    write_csv_rows(rows, output_root / "collected_metrics" / "variant_metrics.csv")
    write_json(comparison, output_root / "comparison" / "reid_ablation_comparison.json")
    write_csv_rows(comparison.get("variants", []), output_root / "comparison" / "reid_ablation_comparison.csv")
    write_csv_rows(_source_rows(comparison.get("variants", [])), output_root / "comparison" / "improvement_source_analysis.csv")
    write_csv_rows(_filter_source(comparison.get("variants", []), "none"), output_root / "diagnostics" / "noop_runs.csv")
    write_csv_rows(_filter_type(comparison.get("variants", []), "reid_only"), output_root / "diagnostics" / "reid_only_effect.csv")
    write_csv_rows(_filter_type(comparison.get("variants", []), "export_compact"), output_root / "diagnostics" / "export_compact_effect.csv")
    write_csv_rows(_filter_type(comparison.get("variants", []), "reid_plus_compact"), output_root / "diagnostics" / "reid_plus_compact_effect.csv")
    return comparison


def compare_reid_ablation_variants(rows: List[Dict[str, Any]], thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add deltas, safety flags, and improvement-source labels."""
    cfg = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)
    v2 = _find_variant(rows, "v2_current")
    export_compact = _find_variant(rows, "export_compact")
    compared = []
    for row in rows:
        item = dict(row)
        item.update(_deltas_vs_v2(item, v2))
        item["is_noop"] = is_noop_variant(item)
        item["is_safe"] = is_safe_variant(item, cfg)
        item["improvement_source"] = classify_improvement_source(item, export_compact)
        item["real_upgrade"] = is_real_upgrade(item, cfg)
        item["is_submission_candidate"] = bool(item.get("is_safe")) and bool(item.get("track1_valid") is True)
        item["recommendation"] = recommendation_for_variant(item)
        compared.append(item)
    return {
        "thresholds": cfg,
        "v2_current": v2,
        "v2_export_compact": export_compact,
        "variants": compared,
    }


def is_noop_variant(row: Dict[str, Any]) -> bool:
    """Return True when a variant has no measurable effect."""
    if row.get("source_type") in ("v1", "v2_current"):
        return False
    reid_merges = _num(row.get("num_reid_merges")) or 0.0
    geometry_merges = _num(row.get("num_geometry_merges")) or 0.0
    export_rows = _num(row.get("num_export_dropped_rows")) or 0.0
    track_delta = _num(row.get("track1_rows_delta_vs_v2")) or 0.0
    frag_delta = _num(row.get("fragmentation_delta_vs_v2")) or 0.0
    person_frag_delta = _num(row.get("person_fragmentation_delta_vs_v2")) or 0.0
    if reid_merges > 0 or geometry_merges > 0 or export_rows > 0:
        return False
    return track_delta == 0.0 and frag_delta == 0.0 and person_frag_delta == 0.0


def is_safe_variant(row: Dict[str, Any], thresholds: Dict[str, Any]) -> bool:
    """Return True when a variant satisfies conservative safety gates."""
    if row.get("source_type") in ("v1", "v2_current"):
        return row.get("track1_valid") is not False
    if bool(thresholds.get("require_track1_valid", True)) and row.get("track1_valid") is not True:
        return False
    if bool(thresholds.get("require_non_person_unchanged", True)):
        non_person_delta = _num(row.get("non_person_rows_delta_vs_v2"))
        if non_person_delta is None or abs(non_person_delta) > 0.0:
            return False
    purity_delta = _num(row.get("global_purity_delta_vs_v2"))
    if purity_delta is not None and purity_delta < -float(thresholds.get("max_purity_drop", 0.003)):
        return False
    false_delta = _num(row.get("false_merge_delta_vs_v2"))
    if false_delta is not None and false_delta > float(thresholds.get("max_false_merge_increase", 0.005)):
        return False
    return True


def is_real_upgrade(row: Dict[str, Any], thresholds: Dict[str, Any]) -> bool:
    """Return True for safe variants with measurable non-noop gain."""
    if row.get("is_noop") or not row.get("is_safe"):
        return False
    frag_delta = _num(row.get("fragmentation_delta_vs_v2"))
    person_frag_delta = _num(row.get("person_fragmentation_delta_vs_v2"))
    track_delta = _num(row.get("track1_rows_delta_vs_v2"))
    base_frag = _num(row.get("v2_fragmentation_approx"))
    if base_frag is not None and base_frag > 0 and frag_delta is not None:
        if (-frag_delta / base_frag) >= float(thresholds.get("min_fragmentation_reduction_for_upgrade", 0.01)):
            return True
    if person_frag_delta is not None and person_frag_delta < 0:
        return True
    if track_delta is not None and -track_delta >= float(thresholds.get("min_track1_rows_reduction_for_compact", 1000)):
        return True
    return False


def classify_improvement_source(row: Dict[str, Any], export_compact: Dict[str, Any]) -> str:
    """Label where the observed improvement came from."""
    source_type = str(row.get("source_type", ""))
    reid_merges = _num(row.get("num_reid_merges")) or 0.0
    geometry_merges = _num(row.get("num_geometry_merges")) or 0.0
    export_rows = _num(row.get("num_export_dropped_rows")) or 0.0
    track_delta = _num(row.get("track1_rows_delta_vs_v2")) or 0.0
    frag_delta = _num(row.get("person_fragmentation_delta_vs_v2")) or 0.0
    if source_type == "v1":
        return "legacy_geometry_submission"
    if source_type == "v2_current":
        return "current_3d_mvp"
    if source_type == "export_compact":
        return "export_compact" if export_rows > 0 or track_delta < 0 else "none"
    if source_type == "geometry_only":
        if geometry_merges > 0 and (frag_delta < 0 or track_delta < 0):
            return "geometry_only"
        if geometry_merges > 0:
            return "minor_geometry_activity_no_measurable_gain"
        return "none"
    if source_type == "reid_only":
        if reid_merges > 0 and (frag_delta < 0 or track_delta < 0):
            return "reid"
        if reid_merges > 0:
            return "minor_reid_activity_no_measurable_gain"
        return "none"
    if source_type == "reid_plus_compact":
        if _same_as_export_compact(row, export_compact):
            return "export_compact_only"
        if reid_merges > 0 and export_rows > 0:
            return "both"
        if export_rows > 0:
            return "export_compact_only"
        if reid_merges > 0:
            return "minor_reid_activity_no_measurable_gain"
    return "none"


def recommendation_for_variant(row: Dict[str, Any]) -> str:
    """Produce a concise recommendation for a variant row."""
    source = str(row.get("improvement_source", "none"))
    if row.get("source_type") == "v1" and row.get("track1_valid") is not False:
        return "keep_v1_for_submission"
    if row.get("source_type") == "v2_current" and row.get("track1_valid") is not False:
        return "keep_v2_current_as_3d_mvp"
    if row.get("source_type") == "export_compact" and row.get("is_safe"):
        return "keep_v2_export_compact_as_safe_variant"
    if source == "reid":
        return "keep_as_reid_ablation_candidate"
    if source == "export_compact_only":
        return "attribute_gain_to_export_compact_not_reid"
    if source == "minor_reid_activity_no_measurable_gain":
        return "reid_infrastructure_valid_but_no_tracking_gain"
    if row.get("is_noop"):
        return "noop_do_not_select"
    if not row.get("is_safe"):
        return "not_safe_for_final"
    return "diagnostic_only"


def _deltas_vs_v2(row: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        ("track1_rows", "track1_rows_delta_vs_v2"),
        ("person_rows", "person_rows_delta_vs_v2"),
        ("non_person_rows", "non_person_rows_delta_vs_v2"),
        ("global_purity", "global_purity_delta_vs_v2"),
        ("false_merge_rate", "false_merge_delta_vs_v2"),
        ("fragmentation_approx", "fragmentation_delta_vs_v2"),
        ("person_fragmentation", "person_fragmentation_delta_vs_v2"),
    ]
    output = {}
    for source, dest in fields:
        left = _num(row.get(source))
        right = _num(v2.get(source))
        output[dest] = NOT_AVAILABLE if left is None or right is None else left - right
        output["v2_%s" % source] = v2.get(source, NOT_AVAILABLE)
    return output


def _same_as_export_compact(row: Dict[str, Any], export_compact: Dict[str, Any]) -> bool:
    if not export_compact:
        return False
    keys = ["track1_rows", "person_rows", "person_fragmentation", "fragmentation_approx"]
    for key in keys:
        left = _num(row.get(key))
        right = _num(export_compact.get(key))
        if left is None or right is None:
            continue
        if abs(left - right) > 1e-9:
            return False
    return True


def _find_variant(rows: List[Dict[str, Any]], source_type: str) -> Dict[str, Any]:
    for row in rows:
        if row.get("source_type") == source_type:
            return row
    return {}


def _num(value: Any) -> Optional[float]:
    return safe_float(value, None)


def _source_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields = [
        "variant_name",
        "source_type",
        "improvement_source",
        "real_upgrade",
        "is_noop",
        "is_safe",
        "num_reid_merges",
        "num_geometry_merges",
        "num_export_dropped_rows",
        "track1_rows_delta_vs_v2",
        "person_fragmentation_delta_vs_v2",
        "recommendation",
    ]
    return [{field: row.get(field, "") for field in fields} for row in rows]


def _filter_type(rows: List[Dict[str, Any]], source_type: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("source_type") == source_type]


def _filter_source(rows: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("improvement_source") == source or row.get("is_noop")]
