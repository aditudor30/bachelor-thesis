"""GT-only diagnostics for train/val local tracker benchmark outputs."""

from typing import Any, Dict, List, Sequence

import numpy as np

from deep_oc_sort_3d.local_tracker_benchmark.local_track_metrics import group_tracks


def compute_gt_diagnostics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute local purity, switches and fragmentation without affecting decisions."""
    gt_rows = [row for row in rows if row.get("matched_gt_object_id") not in (None, "")]
    purities = []
    false_merge_tracks = 0
    for values in group_tracks(gt_rows).values():
        counts = {}
        for row in values:
            identity = str(row.get("matched_gt_object_id"))
            counts[identity] = counts.get(identity, 0) + 1
        if counts:
            purities.append(float(max(counts.values())) / float(sum(counts.values())))
        if len(counts) > 1:
            false_merge_tracks += 1
    by_gt = {}
    for row in gt_rows:
        identity = (
            str(row.get("subset", "")),
            str(row.get("scene_name", "")),
            str(row.get("camera_id", "")),
            str(row.get("class_id", "")),
            str(row.get("matched_gt_object_id")),
        )
        by_gt.setdefault(identity, []).append(row)
    switches = 0
    fragmentation = 0
    for values in by_gt.values():
        ordered = sorted(values, key=lambda row: (int(float(row.get("frame_id", 0))), str(row.get("track_id", ""))))
        track_ids = set([str(row.get("track_id", "")) for row in ordered])
        fragmentation += max(0, len(track_ids) - 1)
        previous = None
        for row in ordered:
            current = str(row.get("track_id", ""))
            if previous is not None and current != previous:
                switches += 1
            previous = current
    num_tracks = len(group_tracks(gt_rows))
    return {
        "gt_match_rate": float(len(gt_rows)) / float(len(rows)) if rows else None,
        "local_purity_mean": float(np.mean(purities)) if purities else None,
        "local_purity_median": float(np.median(purities)) if purities else None,
        "approx_id_switches": switches,
        "approx_fragmentation": fragmentation,
        "false_merge_suspicion_rate": float(false_merge_tracks) / float(num_tracks) if num_tracks else None,
        "gt_matched_records": len(gt_rows),
    }
