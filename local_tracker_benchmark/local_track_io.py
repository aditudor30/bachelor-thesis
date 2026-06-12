"""CSV I/O for benchmark-local tracks and generic tables."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from deep_oc_sort_3d.local_tracker_benchmark.tracker_input_types import BenchmarkTrackRecord


TRACK_FIELDS = (
    "scene_id", "scene_name", "subset", "split", "camera_id", "frame_id", "track_id",
    "class_id", "class_name", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence",
    "source_detection_id", "matched_gt_object_id", "track_age", "track_hits", "track_misses", "track_state",
)


def write_track_records(path: Path, records: Sequence[BenchmarkTrackRecord]) -> None:
    """Write benchmark records in the requested compatible schema."""
    rows = []
    for record in records:
        x1, y1, x2, y2 = record.bbox_xyxy
        rows.append(
            {
                "scene_id": record.scene_id, "scene_name": record.scene_name, "subset": record.subset,
                "split": record.split, "camera_id": record.camera_id, "frame_id": record.frame_id,
                "track_id": record.track_id, "class_id": record.class_id, "class_name": record.class_name,
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "confidence": record.confidence, "source_detection_id": record.source_detection_id,
                "matched_gt_object_id": "" if record.matched_gt_object_id is None else record.matched_gt_object_id,
                "track_age": record.track_age, "track_hits": record.track_hits,
                "track_misses": record.track_misses, "track_state": record.track_state,
            }
        )
    write_csv_rows(path, rows, TRACK_FIELDS)


def read_track_rows(path: Path) -> List[Dict[str, Any]]:
    """Read benchmark or current tracker CSV into a normalized row schema."""
    rows = read_csv_rows(path)
    output = []
    for row in rows:
        output.append(
            {
                **row,
                "track_id": row.get("track_id", row.get("local_track_id", "")),
                "bbox_x1": row.get("bbox_x1", row.get("x1", "")),
                "bbox_y1": row.get("bbox_y1", row.get("y1", "")),
                "bbox_x2": row.get("bbox_x2", row.get("x2", "")),
                "bbox_y2": row.get("bbox_y2", row.get("y2", "")),
            }
        )
    return output


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or _field_union(rows))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def progress_iter(values: Sequence[Any], enabled: bool, description: str) -> Any:
    if enabled:
        try:
            from tqdm import tqdm

            return tqdm(values, desc=description)
        except ImportError:
            pass
    return _print_progress(values, description, enabled)


def _print_progress(values: Sequence[Any], description: str, enabled: bool) -> Any:
    total = len(values)
    for index, value in enumerate(values):
        if enabled and (index == 0 or (index + 1) % max(1, total // 20) == 0 or index + 1 == total):
            print("%s: %d/%d" % (description, index + 1, total))
        yield value


def _field_union(rows: Sequence[Dict[str, Any]]) -> List[str]:
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    return fields
