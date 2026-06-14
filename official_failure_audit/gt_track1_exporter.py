"""Export validation GT in Track1-like form for local diagnosis only."""

from pathlib import Path
from typing import Any, Dict, Sequence

from deep_oc_sort_3d.official_failure_audit.failure_io import write_json
from deep_oc_sort_3d.official_failure_audit.track1_parser import AuditTrack1Row, write_track1_like


def export_gt_track1_like(
    rows: Sequence[AuditTrack1Row], output_root: Path, config: Dict[str, Any],
) -> Dict[str, Any]:
    directory = output_root / "gt_track1_like"
    path = directory / "val_gt_track1_like.txt"
    decimals = int(config.get("official_track1", {}).get("round_float_decimals", 2))
    written = write_track1_like(path, rows, decimals=decimals)
    summary = {
        "status": "ok" if written else "empty", "diagnostic_only": True,
        "not_for_submission": True, "path": str(path), "rows": written,
        "tracks": len(set((row.scene_id, row.class_id, row.object_id) for row in rows)),
        "scenes": sorted(set(row.scene_id for row in rows)),
        "classes": sorted(set(row.class_id for row in rows)),
        "coordinate_convention": "dataset GT 3d_location and 3d_bounding_box_scale as stored",
        "yaw_convention": "last component of dataset GT 3d_bounding_box_rotation",
    }
    write_json(directory / "val_gt_track1_like_summary.json", summary)
    return summary
