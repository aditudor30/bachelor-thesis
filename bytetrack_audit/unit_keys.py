"""Canonical units and keys for coverage-retention diagnostics."""

from typing import Dict, Tuple


STAGE_UNITS = {
    "observations": "observation_record",
    "local_records": "local_track_record",
    "tracklets": "tracklet",
    "candidates": "candidate",
    "motion_clean_candidates": "motion_clean_candidate",
    "global_tracks": "global_track",
    "final_export_rows": "final_export_row",
    "track1_rows": "track1_row",
    "gt_object_frames": "gt_object_frame",
}


TRANSITIONS = [
    ("observations", "local_records"),
    ("local_records", "tracklets"),
    ("tracklets", "candidates"),
    ("candidates", "motion_clean_candidates"),
    ("motion_clean_candidates", "global_tracks"),
    ("global_tracks", "final_export_rows"),
    ("final_export_rows", "track1_rows"),
]


CONSISTENT_TRANSITIONS = {
    ("observations", "local_records"),
    ("tracklets", "candidates"),
    ("candidates", "motion_clean_candidates"),
}


def transition_units(stage_from: str, stage_to: str) -> Tuple[str, str, str]:
    """Return units and whether a ratio is a strict retention."""
    unit_from = STAGE_UNITS.get(stage_from, stage_from)
    unit_to = STAGE_UNITS.get(stage_to, stage_to)
    comparison = "consistent" if (stage_from, stage_to) in CONSISTENT_TRANSITIONS else "diagnostic_only"
    return unit_from, unit_to, comparison


def observation_key(row: Dict[str, object]) -> Tuple[str, str, int, int]:
    """Return a stable camera-frame-detection key."""
    return (
        str(row.get("scene_name", "")),
        str(row.get("camera_id", "")),
        int(row.get("frame_id", -1)),
        int(row.get("detection_id", -1)),
    )


def local_track_key(row: Dict[str, object]) -> Tuple[str, str, int, int]:
    """Return a stable scene-camera-class-local-track key."""
    return (
        str(row.get("scene_name", "")),
        str(row.get("camera_id", "")),
        int(row.get("class_id", -1)),
        int(row.get("local_track_id", row.get("track_id", -1))),
    )

