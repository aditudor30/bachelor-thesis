"""Select informative drop traces for later visual review."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.bytetrack_audit.audit_config import output_root
from deep_oc_sort_3d.bytetrack_audit.audit_io import iter_csv, safe_float, write_csv


def build_drop_trace_samples(
    config: Dict[str, Any],
    gt_result: Dict[str, Any],
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Write bounded, high-information samples from GT and motion audits."""
    limit = int(max_samples or config.get("samples", {}).get("max_samples_per_category", 50))
    root = output_root(config)
    gt_rows = list(gt_result.get("drop_samples", []))
    object_rows = _balanced_take(gt_rows, "drop_stage", limit)
    local_rows = [row for row in object_rows if row.get("drop_stage") == "local_export"][:limit]
    rejected_path = root / "motion_filter_audit" / "rejected_candidates.csv"
    rejected = list(iter_csv(rejected_path))
    high_step = sorted(rejected, key=lambda row: safe_float(row.get("step_max"), -1.0) or -1.0, reverse=True)[:limit]
    high_bbox = sorted(
        rejected,
        key=lambda row: safe_float(row.get("bbox_height_delta"), -1.0) or -1.0,
        reverse=True,
    )[:limit]
    candidate_rows = _deduplicate(high_step + high_bbox, "candidate_id")
    output = root / "drop_trace_samples"
    write_csv(output / "dropped_object_frame_samples.csv", object_rows)
    write_csv(output / "dropped_local_record_samples.csv", local_rows)
    write_csv(output / "dropped_candidate_samples.csv", candidate_rows)
    _write_report(output / "sample_trace_report.md", object_rows, local_rows, candidate_rows)
    return {
        "object_frame_samples": len(object_rows),
        "local_record_samples": len(local_rows),
        "candidate_samples": len(candidate_rows),
    }


def _balanced_take(rows: List[Dict[str, Any]], key: str, limit: int) -> List[Dict[str, Any]]:
    groups = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(row)
    output = []
    for _name, values in sorted(groups.items()):
        output.extend(values[:limit])
    return output


def _deduplicate(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        value = (row.get("variant_name"), row.get(key))
        if value in seen:
            continue
        seen.add(value)
        output.append(row)
    return output


def _write_report(
    path: Path,
    object_rows: List[Dict[str, Any]],
    local_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ByteTrack Drop Trace Samples",
        "",
        "These tables are diagnostic selections, not training data.",
        "",
        "- GT object-frame samples: %d" % len(object_rows),
        "- Local export drops: %d" % len(local_rows),
        "- Motion-rejected candidate samples: %d" % len(candidate_rows),
        "",
        "Use scene, camera, frame, bbox, local track and candidate ids to build a later visual reviewer.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

