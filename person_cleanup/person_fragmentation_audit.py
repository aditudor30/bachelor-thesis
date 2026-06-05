"""Person-specific fragmentation audit."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from deep_oc_sort_3d.person_cleanup.person_cleanup_io import (
    count_by,
    frame_record_csv_files,
    generic_csv_files,
    load_yaml,
    mean,
    percentile,
    progress_iter,
    read_csv_rows,
    safe_float,
    safe_int,
    track_key,
    write_csv_rows,
    write_json,
)
from deep_oc_sort_3d.person_cleanup.person_track_classifier import build_track_stats_from_rows, classify_person_track


def run_person_fragmentation_audit(config_path: Path, progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Run Person fragmentation audit from a YAML config."""
    data = load_yaml(config_path)
    config = data.get("person_cleanup", data)
    paths = data.get("paths", config.get("paths", {}))
    output_root = Path(config.get("output_root", paths.get("output_root", "output/person_cleanup/baseline_v2_pseudo3d_fullcam")))
    return audit_person_fragmentation(
        final_export_root=Path(str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam"))),
        output_root=output_root / "audit",
        config=config,
        progress=progress,
    )


def audit_person_fragmentation(
    final_export_root: Path,
    output_root: Path,
    config: Dict[str, Any],
    progress: bool = True,
) -> Dict[str, Any]:
    """Audit Person fragmentation in final export artifacts."""
    person_class_id = int(config.get("class_id", 0))
    generic_rows = _load_generic_rows(final_export_root / "generic_tracking_export", person_class_id, progress)
    frame_rows = _load_frame_rows(final_export_root / "frame_global_records", person_class_id, progress)
    stats = [classify_person_track(item, config.get("classification", {})) for item in build_track_stats_from_rows(generic_rows, person_class_id)]
    stats_rows = [item.to_dict() for item in stats]
    singleton_rows = [row for row in stats_rows if int(row.get("rows", 0)) <= 1]
    short_rows = [row for row in stats_rows if int(row.get("rows", 0)) <= int(config.get("short_rows_threshold", 3))]
    low_conf_rows = [
        row
        for row in stats_rows
        if safe_float(row.get("mean_confidence"), 1.0) < float(config.get("low_mean_confidence_threshold", 0.03))
    ]
    duplicate_rows = duplicate_like_person_rows(generic_rows, config)
    hotspots = scene_camera_hotspots(generic_rows)
    gt_diagnostic = person_gt_diagnostic(frame_rows, stats)
    summary = _summary(generic_rows, frame_rows, stats, duplicate_rows, hotspots, gt_diagnostic)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, output_root / "person_fragmentation_summary.json")
    write_csv_rows(_summary_rows(summary), output_root / "person_fragmentation_summary.csv", ["metric", "value"])
    write_csv_rows(stats_rows, output_root / "person_track_length_distribution.csv")
    write_csv_rows(rows_per_track_distribution(stats), output_root / "person_rows_per_track_distribution.csv")
    write_csv_rows(singleton_rows, output_root / "person_singleton_tracks.csv")
    write_csv_rows(short_rows, output_root / "person_short_tracks.csv")
    write_csv_rows(low_conf_rows, output_root / "person_low_conf_tracks.csv")
    write_csv_rows(duplicate_rows, output_root / "person_duplicate_frame_analysis.csv")
    write_csv_rows(hotspots, output_root / "person_scene_camera_hotspots.csv")
    write_csv_rows(gt_diagnostic.get("rows", []), output_root / "person_val_holdout_gt_diagnostic.csv")
    return summary


def duplicate_like_person_rows(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find duplicate-like Person rows in the same scene/camera/frame."""
    max_rows = int(config.get("max_duplicate_rows", 50000))
    iou_threshold = float(config.get("duplicate_iou_threshold", 0.70))
    center_threshold = float(config.get("duplicate_center_distance_threshold", 1.0))
    groups = {}
    for row in rows:
        key = (row.get("subset", ""), row.get("scene_name", ""), row.get("camera_id", ""), row.get("frame_id", ""))
        groups.setdefault(key, []).append(row)
    output = []
    for key, group_rows in sorted(groups.items()):
        if len(group_rows) <= 1:
            continue
        for left_index in range(len(group_rows)):
            for right_index in range(left_index + 1, len(group_rows)):
                left = group_rows[left_index]
                right = group_rows[right_index]
                if str(left.get("global_track_id", "")) == str(right.get("global_track_id", "")):
                    continue
                iou = bbox_iou(left, right)
                distance = center_distance(left, right)
                if iou >= iou_threshold or (distance is not None and distance <= center_threshold):
                    output.append(
                        {
                            "subset": key[0],
                            "scene_name": key[1],
                            "camera_id": key[2],
                            "frame_id": key[3],
                            "global_track_id_a": left.get("global_track_id"),
                            "global_track_id_b": right.get("global_track_id"),
                            "iou": iou,
                            "center_distance": distance,
                        }
                    )
                    if len(output) >= max_rows:
                        return output
    return output


def scene_camera_hotspots(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize Person rows by subset/scene/camera."""
    groups = {}
    tracks = {}
    for row in rows:
        key = (str(row.get("subset", "")), str(row.get("scene_name", "")), str(row.get("camera_id", "")))
        groups[key] = groups.get(key, 0) + 1
        tracks.setdefault(key, set()).add(str(row.get("global_track_id", "")))
    output = []
    for key, count in sorted(groups.items(), key=lambda item: item[1], reverse=True):
        output.append(
            {
                "subset": key[0],
                "scene_name": key[1],
                "camera_id": key[2],
                "person_rows": count,
                "person_tracks": len(tracks.get(key, set())),
            }
        )
    return output


def person_gt_diagnostic(frame_rows: List[Dict[str, Any]], stats: List[Any]) -> Dict[str, Any]:
    """Build GT diagnostic for val/holdout Person rows only."""
    track_to_gt = {}
    gt_to_tracks = {}
    for row in frame_rows:
        if str(row.get("subset", "")) == "test":
            continue
        gt_id = row.get("matched_gt_object_id")
        if gt_id in (None, ""):
            continue
        key = track_key(row)
        gt_number = safe_int(gt_id, None)
        if gt_number is None:
            continue
        gt_key = str(gt_number)
        track_to_gt.setdefault(key, {})
        track_to_gt[key][gt_key] = track_to_gt[key].get(gt_key, 0) + 1
        gt_to_tracks.setdefault(gt_key, set()).add(key)
    rows = []
    false_merge_tracks = 0
    purity_values = []
    for key, gt_counts in sorted(track_to_gt.items(), key=lambda item: item[0]):
        total = sum(gt_counts.values())
        majority = max(gt_counts.values()) if gt_counts else 0
        purity = float(majority) / float(total) if total > 0 else None
        if purity is not None:
            purity_values.append(purity)
        if len(gt_counts) > 1:
            false_merge_tracks += 1
        rows.append(
            {
                "subset": key[0],
                "scene_name": key[1],
                "class_id": key[2],
                "global_track_id": key[3],
                "gt_ids": len(gt_counts),
                "gt_counts": str(gt_counts),
                "purity": purity,
            }
        )
    fragmentation = sum([max(0, len(track_keys) - 1) for track_keys in gt_to_tracks.values()])
    return {
        "rows": rows,
        "person_purity": mean(purity_values),
        "person_false_merge_tracks": false_merge_tracks,
        "person_false_merge_rate": float(false_merge_tracks) / float(len(track_to_gt)) if track_to_gt else None,
        "person_fragmentation_approx": fragmentation,
        "person_gt_object_coverage": len(gt_to_tracks),
    }


def rows_per_track_distribution(stats: List[Any]) -> List[Dict[str, Any]]:
    """Return histogram of rows per Person track."""
    counts = {}
    for item in stats:
        value = int(item.rows)
        counts[value] = counts.get(value, 0) + 1
    return [{"rows_per_track": key, "num_tracks": counts[key]} for key in sorted(counts.keys())]


def bbox_iou(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    """Compute 2D bbox IoU from generic rows."""
    box_a = _bbox(left)
    box_b = _bbox(right)
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter) / float(denom)


def center_distance(left: Dict[str, Any], right: Dict[str, Any]) -> Any:
    """Compute 3D center distance if coordinates exist."""
    a = [safe_float(left.get("center_x"), None), safe_float(left.get("center_y"), None), safe_float(left.get("center_z"), None)]
    b = [safe_float(right.get("center_x"), None), safe_float(right.get("center_y"), None), safe_float(right.get("center_z"), None)]
    if any(value is None for value in a + b):
        return None
    return sum([(float(a[index]) - float(b[index])) ** 2 for index in range(3)]) ** 0.5


def _load_generic_rows(root: Path, person_class_id: int, progress: bool) -> List[Dict[str, Any]]:
    rows = []
    files = generic_csv_files(root)
    for path in progress_iter(files, progress, "audit generic Person files", "file"):
        subset = path.parent.name
        file_rows, _fields = read_csv_rows(path)
        for row in file_rows:
            if safe_int(row.get("class_id"), -1) == person_class_id:
                copied = dict(row)
                copied["subset"] = subset
                rows.append(copied)
    return rows


def _load_frame_rows(root: Path, person_class_id: int, progress: bool) -> List[Dict[str, Any]]:
    rows = []
    files = frame_record_csv_files(root)
    for path in progress_iter(files, progress, "audit frame Person files", "file"):
        file_rows, _fields = read_csv_rows(path)
        for row in file_rows:
            if safe_int(row.get("class_id"), -1) == person_class_id:
                rows.append(row)
    return rows


def _summary(
    generic_rows: List[Dict[str, Any]],
    frame_rows: List[Dict[str, Any]],
    stats: List[Any],
    duplicate_rows: List[Dict[str, Any]],
    hotspots: List[Dict[str, Any]],
    gt_diagnostic: Dict[str, Any],
) -> Dict[str, Any]:
    lengths = [item.rows for item in stats]
    mean_conf = [item.mean_confidence for item in stats]
    return {
        "person_generic_rows": len(generic_rows),
        "person_frame_rows": len(frame_rows),
        "person_global_tracks": len(stats),
        "rows_per_track_mean": mean(lengths),
        "rows_per_track_median": percentile(lengths, 50),
        "rows_per_track_p95": percentile(lengths, 95),
        "mean_confidence_per_track": mean(mean_conf),
        "singleton_tracks": len([item for item in stats if item.rows <= 1]),
        "short_tracks_lte_2": len([item for item in stats if item.rows <= 2]),
        "short_tracks_lte_3": len([item for item in stats if item.rows <= 3]),
        "short_tracks_lte_5": len([item for item in stats if item.rows <= 5]),
        "duplicate_like_pairs": len(duplicate_rows),
        "hotspot_rows": len(hotspots),
        "per_subset_rows": count_by(generic_rows, "subset"),
        "per_scene_rows": count_by(generic_rows, "scene_name"),
        "gt_diagnostic": {key: value for key, value in gt_diagnostic.items() if key != "rows"},
    }


def _summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key, value in summary.items():
        rows.append({"metric": key, "value": value})
    return rows


def _bbox(row: Dict[str, Any]) -> Any:
    values = [safe_float(row.get("x1"), None), safe_float(row.get("y1"), None), safe_float(row.get("x2"), None), safe_float(row.get("y2"), None)]
    if any(value is None for value in values):
        return None
    return values
