"""Load current local tracker outputs as the immutable benchmark baseline."""

from pathlib import Path
from typing import Any, Dict, List, Sequence

from deep_oc_sort_3d.local_tracker_benchmark.local_track_io import read_track_rows, write_csv_rows


def load_current_tracker_subset(
    current_root: Path,
    inventory_rows: Sequence[Dict[str, Any]],
    output_root: Path,
) -> Dict[str, Any]:
    """Copy normalized current tracker rows only for benchmark cameras."""
    files = 0
    records = 0
    missing = []
    for item in inventory_rows:
        source = Path(current_root) / str(item["subset"]) / str(item["scene_name"]) / (str(item["camera_id"]) + ".csv")
        target = output_root / "local_tracks" / str(item["subset"]) / str(item["scene_name"]) / (str(item["camera_id"]) + ".csv")
        rows = read_track_rows(source)
        if not rows:
            missing.append(str(source))
            continue
        for row in rows:
            row.setdefault("subset", item["subset"])
        write_csv_rows(target, rows)
        files += 1
        records += len(rows)
    return {
        "status": "ok" if files else "skipped",
        "reason": "no current local tracker files found" if not files else "",
        "files": files,
        "records": records,
        "missing_files": missing,
    }
