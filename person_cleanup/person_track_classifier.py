"""Track-level statistics and labels for Person cleanup."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.person_cleanup.person_cleanup_io import mean, safe_float, safe_int, track_key


@dataclass
class PersonTrackStats:
    """Per-global-track stats used by audit and cleanup policies."""

    key: Tuple[str, str, str, str]
    subset: str
    scene_name: str
    class_id: int
    class_name: str
    global_track_id: str
    rows: int
    unique_frames: int
    cameras: List[str]
    min_frame: Optional[int]
    max_frame: Optional[int]
    mean_confidence: float
    max_confidence: float
    min_confidence: float
    matched_gt_rows: int = 0
    gt_id_counts: Dict[str, int] = field(default_factory=dict)
    category: str = "unknown"
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stats for CSV/JSON."""
        return {
            "subset": self.subset,
            "scene_name": self.scene_name,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "global_track_id": self.global_track_id,
            "rows": self.rows,
            "unique_frames": self.unique_frames,
            "num_cameras": len(self.cameras),
            "cameras": ",".join(self.cameras),
            "min_frame": self.min_frame,
            "max_frame": self.max_frame,
            "mean_confidence": self.mean_confidence,
            "max_confidence": self.max_confidence,
            "min_confidence": self.min_confidence,
            "matched_gt_rows": self.matched_gt_rows,
            "num_gt_ids": len(self.gt_id_counts),
            "gt_id_counts": str(self.gt_id_counts),
            "category": self.category,
            "reasons": ";".join(self.reasons),
        }


def build_track_stats_from_rows(rows: List[Dict[str, Any]], person_class_id: int = 0) -> List[PersonTrackStats]:
    """Build stats for Person tracks from generic or frame rows."""
    groups = {}
    for row in rows:
        class_id = safe_int(row.get("class_id"), -1)
        if class_id != int(person_class_id):
            continue
        key = track_key(row)
        if key[3] in ("", "None"):
            continue
        groups.setdefault(key, []).append(row)
    output = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        output.append(track_stats_from_group(key, group_rows))
    return output


def track_stats_from_group(key: Tuple[str, str, str, str], rows: List[Dict[str, Any]]) -> PersonTrackStats:
    """Compute stats for one grouped track."""
    frames = [safe_int(row.get("frame_id"), None) for row in rows]
    frames = [frame for frame in frames if frame is not None]
    confidences = [safe_float(row.get("confidence"), 0.0) or 0.0 for row in rows]
    cameras = sorted(set([str(row.get("camera_id", "")) for row in rows if str(row.get("camera_id", "")) != ""]))
    gt_counts = {}
    matched_gt_rows = 0
    for row in rows:
        gt_id = row.get("matched_gt_object_id")
        matched = str(row.get("matched_gt", "")).lower() in ("true", "1", "yes")
        if gt_id not in (None, ""):
            gt_number = safe_int(gt_id, None)
            if gt_number is not None:
                gt_counts[str(gt_number)] = gt_counts.get(str(gt_number), 0) + 1
        if matched:
            matched_gt_rows += 1
    class_id = safe_int(rows[0].get("class_id"), -1) if rows else -1
    stats = PersonTrackStats(
        key=key,
        subset=str(key[0]),
        scene_name=str(key[1]),
        class_id=int(class_id if class_id is not None else -1),
        class_name=str(rows[0].get("class_name", "Person")) if rows else "Person",
        global_track_id=str(key[3]),
        rows=len(rows),
        unique_frames=len(set(frames)),
        cameras=cameras,
        min_frame=min(frames) if frames else None,
        max_frame=max(frames) if frames else None,
        mean_confidence=float(mean(confidences) or 0.0),
        max_confidence=float(max(confidences) if confidences else 0.0),
        min_confidence=float(min(confidences) if confidences else 0.0),
        matched_gt_rows=matched_gt_rows,
        gt_id_counts=gt_counts,
    )
    return stats


def classify_person_track(stats: PersonTrackStats, config: Optional[Dict[str, Any]] = None) -> PersonTrackStats:
    """Assign a diagnostic category to a Person track."""
    cfg = config or {}
    short_rows = int(cfg.get("short_rows_threshold", 3))
    low_mean = float(cfg.get("low_mean_confidence_threshold", 0.03))
    low_max = float(cfg.get("low_max_confidence_threshold", 0.08))
    high_mean = float(cfg.get("high_mean_confidence_threshold", 0.30))
    reasons = []
    if stats.rows <= 1:
        reasons.append("singleton")
    if stats.rows <= short_rows:
        reasons.append("short")
    if stats.mean_confidence < low_mean:
        reasons.append("low_mean_confidence")
    if stats.max_confidence < low_max:
        reasons.append("low_max_confidence")
    if stats.mean_confidence >= high_mean:
        reasons.append("high_confidence")
    if len(stats.gt_id_counts) > 1:
        reasons.append("multi_gt_id")
    if stats.matched_gt_rows > 0:
        reasons.append("gt_matched")

    if "short" in reasons and "low_mean_confidence" in reasons and "low_max_confidence" in reasons:
        category = "likely_false_positive_short_fragment"
    elif "singleton" in reasons and "low_mean_confidence" in reasons:
        category = "low_confidence_singleton"
    elif "short" in reasons and "high_confidence" in reasons:
        category = "high_confidence_short_fragment"
    elif stats.matched_gt_rows > 0 and stats.rows <= short_rows:
        category = "likely_real_but_fragmented"
    elif len(stats.gt_id_counts) > 1:
        category = "possible_false_merge"
    else:
        category = "unknown"
    stats.category = category
    stats.reasons = reasons
    return stats
