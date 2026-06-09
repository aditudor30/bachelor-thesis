"""Load and normalize selected fine-tuned ReID merge events."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.reid_visual_decision.visual_decision_config import output_root_from_config, source_root_from_config
from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import (
    bool_from_any,
    parse_track_key_or_empty,
    read_csv_dicts,
    read_json,
    safe_float,
    safe_int,
    track_key_text,
    write_csv_dicts,
    write_json,
)


MERGE_EVENT_FIELDS = [
    "variant",
    "merge_event_id",
    "source",
    "is_reid_merge",
    "is_export_compact_effect",
    "subset",
    "scene_name",
    "class_id",
    "fragment_a_id",
    "fragment_b_id",
    "global_track_a_before",
    "global_track_b_before",
    "global_track_after",
    "camera_a",
    "camera_b",
    "frame_start_a",
    "frame_end_a",
    "frame_start_b",
    "frame_end_b",
    "temporal_gap",
    "spatial_distance",
    "motion_score",
    "geometry_score",
    "reid_similarity",
    "combined_score",
    "min_mean_confidence",
    "same_gt_diagnostic",
    "reid_gt_pair_label",
    "track1_rows_delta",
    "person_fragmentation_delta_contribution",
    "available_visual_evidence",
    "audit_path",
]


def load_variant_merge_events(source_root: Path, variant: str) -> List[Dict[str, Any]]:
    """Load selected ReID merge events for one variant."""
    run_root = Path(source_root) / "sweep_runs" / variant
    audit_path = run_root / "diagnostics" / "reid_merge_audit.csv"
    mapping_path = run_root / "diagnostics" / "reid_merge_mapping.csv"
    audit_rows = read_csv_dicts(audit_path)
    mapping = load_mapping(mapping_path)
    selected = [
        row
        for row in audit_rows
        if bool_from_any(row.get("merge_selected")) and str(row.get("merge_reject_reason", "ok")) == "ok"
    ]
    events = []
    for index, row in enumerate(selected):
        if int(safe_int(row.get("class_id"), 0) or 0) != 0:
            continue
        event = normalize_merge_event(row, mapping, variant, index, audit_path)
        events.append(event)
    return events


def load_mapping(mapping_path: Path) -> Dict[str, str]:
    """Load old track key -> new global id mapping."""
    rows = read_csv_dicts(mapping_path)
    mapping: Dict[str, str] = {}
    for row in rows:
        key = track_key_text(row.get("old_track_key"))
        if key:
            mapping[key] = str(row.get("new_global_track_id", ""))
    return mapping


def normalize_merge_event(
    row: Dict[str, Any],
    mapping: Dict[str, str],
    variant: str,
    index: int,
    audit_path: Path,
) -> Dict[str, Any]:
    """Normalize one selected merge-audit row into review format."""
    track_a = track_key_text(row.get("track_a") or row.get("fragment_a"))
    track_b = track_key_text(row.get("track_b") or row.get("fragment_b"))
    key_a = parse_track_key_or_empty(track_a)
    key_b = parse_track_key_or_empty(track_b)
    subset = row.get("subset") or key_a[0] or key_b[0]
    scene_name = row.get("scene_name") or key_a[1] or key_b[1]
    class_id = row.get("class_id") or key_a[2] or key_b[2]
    after = mapping.get(track_a) or mapping.get(track_b) or canonical_global_id([key_a[3], key_b[3]])
    event_id = "%s__%s__%s__%04d" % (variant, str(scene_name), str(after), index)
    return {
        "variant": variant,
        "merge_event_id": event_id,
        "source": "reid_merge" if str(row.get("reid_status", "ok")) == "ok" else "geometry_merge",
        "is_reid_merge": "1" if str(row.get("reid_status", "ok")) == "ok" else "0",
        "is_export_compact_effect": "0",
        "subset": subset,
        "scene_name": scene_name,
        "class_id": class_id,
        "fragment_a_id": track_a,
        "fragment_b_id": track_b,
        "global_track_a_before": row.get("global_track_a") or key_a[3],
        "global_track_b_before": row.get("global_track_b") or key_b[3],
        "global_track_after": after,
        "camera_a": row.get("camera_a") or row.get("cameras_a", ""),
        "camera_b": row.get("camera_b") or row.get("cameras_b", ""),
        "frame_start_a": first_present(row, ["frame_start_a", "start_a", "first_frame_a"]),
        "frame_end_a": first_present(row, ["frame_end_a", "end_a", "last_frame_a"]),
        "frame_start_b": first_present(row, ["frame_start_b", "start_b", "first_frame_b"]),
        "frame_end_b": first_present(row, ["frame_end_b", "end_b", "last_frame_b"]),
        "temporal_gap": first_present(row, ["temporal_gap", "frame_gap"]),
        "spatial_distance": first_present(row, ["spatial_distance", "entry_exit_distance_3d", "entry_exit_distance"]),
        "motion_score": first_present(row, ["motion_score", "velocity_score", "velocity_angle"]),
        "geometry_score": first_present(row, ["geometry_score", "pair_score", "geometry_pair_score"]),
        "reid_similarity": row.get("reid_similarity", ""),
        "combined_score": first_present(row, ["combined_score", "combined_pair_score"]),
        "min_mean_confidence": row.get("min_mean_confidence", ""),
        "same_gt_diagnostic": row.get("same_gt_diagnostic", ""),
        "reid_gt_pair_label": row.get("reid_gt_pair_label", ""),
        "track1_rows_delta": "",
        "person_fragmentation_delta_contribution": "",
        "available_visual_evidence": "unknown",
        "audit_path": str(audit_path),
    }


def first_present(row: Dict[str, Any], names: List[str]) -> Any:
    """Return first non-empty row value among names."""
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def canonical_global_id(values: List[str]) -> str:
    """Choose a stable fallback global id."""
    valid = [str(value) for value in values if str(value) not in ("", "None")]
    if not valid:
        return ""

    def _key(value: str) -> Any:
        try:
            return (0, int(float(value)))
        except (TypeError, ValueError):
            return (1, value)

    return sorted(valid, key=_key)[0]


def load_and_write_merge_events(config: Dict[str, Any], variants: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load selected merge events for configured variants and write audit CSVs."""
    source_root = source_root_from_config(config)
    output_root = output_root_from_config(config)
    if variants is None:
        variants = [str(item) for item in config.get("person_reid_visual_decision", {}).get("variants", [])]
    if not variants:
        variants = ["threshold_080", "combined_safe_080"]
    all_events: List[Dict[str, Any]] = []
    per_variant: Dict[str, Any] = {}
    sweep_rows = read_csv_dicts(source_root / "comparison" / "sweep_summary.csv")
    sweep_by_name = {str(row.get("run_name", "")): row for row in sweep_rows}
    for variant in variants:
        events = load_variant_merge_events(source_root, variant)
        attach_variant_deltas(events, sweep_by_name.get(variant, {}))
        out_path = output_root / "merge_audit" / ("%s_merge_events.csv" % variant)
        write_csv_dicts(events, out_path, MERGE_EVENT_FIELDS)
        all_events.extend(events)
        per_variant[variant] = {"events": len(events), "path": str(out_path)}
    before_after = build_before_after_rows(all_events)
    write_csv_dicts(before_after, output_root / "merge_audit" / "merged_tracks_before_after.csv")
    summary = {
        "variants": per_variant,
        "total_events": len(all_events),
        "reid_events": len([row for row in all_events if str(row.get("is_reid_merge")) == "1"]),
        "export_compact_effect_events": len([row for row in all_events if str(row.get("is_export_compact_effect")) == "1"]),
    }
    write_json(summary, output_root / "merge_audit" / "merge_event_summary.json")
    return {"events": all_events, "summary": summary}


def attach_variant_deltas(events: List[Dict[str, Any]], sweep_row: Dict[str, Any]) -> None:
    """Attach variant-level deltas to each event for context."""
    for event in events:
        event["track1_rows_delta"] = sweep_row.get("track1_rows_delta", "")
        event["person_fragmentation_delta_contribution"] = sweep_row.get("person_fragmentation_delta", "")


def build_before_after_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a compact before/after track mapping table."""
    rows = []
    for event in events:
        rows.append(
            {
                "variant": event.get("variant"),
                "merge_event_id": event.get("merge_event_id"),
                "fragment_a_id": event.get("fragment_a_id"),
                "fragment_b_id": event.get("fragment_b_id"),
                "global_track_a_before": event.get("global_track_a_before"),
                "global_track_b_before": event.get("global_track_b_before"),
                "global_track_after": event.get("global_track_after"),
                "source": event.get("source"),
            }
        )
    return rows

