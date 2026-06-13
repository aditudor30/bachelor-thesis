"""Official class remapping, deduplication and decimal formatting."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from deep_oc_sort_3d.official_023_027.official_track1_io import OfficialTrack1Row


def remap_rows_to_official(
    rows: Sequence[OfficialTrack1Row],
    internal_to_official: Dict[int, int],
) -> Tuple[List[OfficialTrack1Row], Dict[str, Any]]:
    """Apply the official class mapping without changing other fields."""
    output = []
    before = defaultdict(int)
    after = defaultdict(int)
    unmapped = defaultdict(int)
    for row in rows:
        before[str(row.class_id)] += 1
        official_class = internal_to_official.get(int(row.class_id))
        if official_class is None:
            unmapped[str(row.class_id)] += 1
            continue
        output.append(
            OfficialTrack1Row(
                scene_id=row.scene_id,
                class_id=int(official_class),
                object_id=row.object_id,
                frame_id=row.frame_id,
                x=row.x,
                y=row.y,
                z=row.z,
                width=row.width,
                length=row.length,
                height=row.height,
                yaw=row.yaw,
                source_line=row.source_line,
                confidence=row.confidence,
            )
        )
        after[str(official_class)] += 1
    return output, {
        "input_rows": len(rows),
        "output_rows": len(output),
        "per_internal_class": dict(sorted(before.items())),
        "per_official_class": dict(sorted(after.items())),
        "unmapped_classes": dict(sorted(unmapped.items())),
        "status": "ok" if not unmapped else "error",
    }


def stable_deduplicate_rows(rows: Sequence[OfficialTrack1Row]) -> Tuple[List[OfficialTrack1Row], Dict[str, Any]]:
    """Keep highest-confidence duplicates, otherwise the first stable row."""
    grouped = {}
    duplicate_count = 0
    confidence_resolved = 0
    first_row_resolved = 0
    for row in sorted(rows, key=lambda item: (item.key(), item.source_line)):
        key = row.key()
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = row
            continue
        duplicate_count += 1
        if row.confidence is not None and previous.confidence is not None:
            confidence_resolved += 1
            if float(row.confidence) > float(previous.confidence):
                grouped[key] = row
        else:
            first_row_resolved += 1
    output = sorted(grouped.values(), key=lambda item: item.key())
    return output, {
        "input_rows": len(rows),
        "output_rows": len(output),
        "duplicate_rows_removed": duplicate_count,
        "duplicates_resolved_by_confidence": confidence_resolved,
        "duplicates_resolved_by_first_stable_row": first_row_resolved,
    }
