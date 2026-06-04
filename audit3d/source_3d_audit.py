"""Audit available provenance metadata for 3D fields."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from deep_oc_sort_3d.audit3d.audit3d_io import (
    iter_data_files,
    progress_iter,
    read_csv_dicts,
    read_jsonl_dicts,
    write_markdown,
)


SOURCE_3D_VALUES = ["gt_matched", "depth_sampled", "class_default", "baseline_estimate", "propagated", "unknown"]
RECOMMENDED_METADATA_FIELDS = [
    "source_3d",
    "depth_source",
    "dimensions_source",
    "yaw_source",
    "is_gt_derived",
    "is_estimated_for_test",
]


def infer_3d_source_from_record(record: Dict[str, Any]) -> str:
    """Infer source_3d only from explicit metadata already present."""
    explicit = _first_present(record, ["source_3d", "3d_source", "source3d"])
    if explicit in SOURCE_3D_VALUES:
        return str(explicit)

    matched_gt = record.get("matched_gt")
    if _truthy(matched_gt):
        return "gt_matched"

    depth_source = _first_present(record, ["depth_source", "depth_sampling_method", "depth_sample_method"])
    if depth_source not in (None, "", "none", "None"):
        return "depth_sampled"

    dimensions_source = _first_present(record, ["dimensions_source", "dimension_source"])
    if str(dimensions_source).lower() in ("class_default", "default_class_dimensions", "default"):
        return "class_default"

    source = str(record.get("source", "")).lower()
    if source in ("propagated", "global_record_propagated"):
        return "propagated"
    if source in ("baseline_estimate", "baseline", "estimated"):
        return "baseline_estimate"
    return "unknown"


def audit_3d_sources(
    frame_records_root: Union[str, Path],
    observations_root: Optional[Union[str, Path]] = None,
    candidates_root: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Audit source metadata in frame records and related pipeline outputs."""
    frame_records = _read_record_tree(frame_records_root, show_progress, "read frame source records")
    source_counts = {}
    per_subset = {}
    explicit_count = 0
    for record in progress_iter(frame_records, show_progress, "audit 3D sources", "record"):
        source = infer_3d_source_from_record(record)
        source_counts[source] = source_counts.get(source, 0) + 1
        subset = str(record.get("subset", record.get("split", "")))
        per_subset.setdefault(subset, {"subset": subset, "record_count": 0})
        per_subset[subset]["record_count"] += 1
        per_subset[subset][source] = per_subset[subset].get(source, 0) + 1
        if _has_explicit_source_metadata(record):
            explicit_count += 1

    observation_field_summary = _scan_available_fields(observations_root, show_progress, "scan observations")
    candidate_field_summary = _scan_available_fields(candidates_root, show_progress, "scan candidates")
    missing_metadata = [field for field in RECOMMENDED_METADATA_FIELDS if not _field_seen(frame_records, field)]
    report = {
        "frame_records_root": str(frame_records_root),
        "observations_root": str(observations_root) if observations_root is not None else "",
        "candidates_root": str(candidates_root) if candidates_root is not None else "",
        "record_count": len(frame_records),
        "records_with_explicit_source_metadata": explicit_count,
        "records_with_unknown_source": int(source_counts.get("unknown", 0)),
        "source_counts": {key: int(source_counts.get(key, 0)) for key in SOURCE_3D_VALUES},
        "per_subset": list(per_subset.values()),
        "missing_recommended_metadata_fields": missing_metadata,
        "recommended_metadata_fields": RECOMMENDED_METADATA_FIELDS,
        "observation_field_summary": observation_field_summary,
        "candidate_field_summary": candidate_field_summary,
        "interpretation": _interpretation(source_counts, len(frame_records), missing_metadata),
    }
    return report


def write_missing_source_metadata_report(report: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write a Markdown report for missing 3D source metadata."""
    lines = [
        "# Missing 3D Source Metadata Report",
        "",
        "This audit does not invent provenance. Rows without explicit source fields are marked `unknown`.",
        "",
        "## Summary",
        "",
        "- Records audited: %s" % report.get("record_count", 0),
        "- Records with explicit source metadata: %s" % report.get("records_with_explicit_source_metadata", 0),
        "- Records with `unknown` source: %s" % report.get("records_with_unknown_source", 0),
        "",
        "## Source Counts",
        "",
    ]
    for key, value in sorted(report.get("source_counts", {}).items()):
        lines.append("- `%s`: %s" % (key, value))
    lines.extend(["", "## Missing Recommended Fields", ""])
    missing = report.get("missing_recommended_metadata_fields", [])
    if missing:
        for field in missing:
            lines.append("- `%s`" % field)
    else:
        lines.append("No recommended metadata fields are missing from the sampled frame records.")
    lines.extend(
        [
            "",
            "## Recommended Future Metadata",
            "",
            "- `source_3d`: one of gt_matched, depth_sampled, class_default, baseline_estimate, propagated, unknown",
            "- `depth_source`: depth map, pseudo-depth, no-depth, or unavailable",
            "- `dimensions_source`: GT, class prior, detector estimate, propagated, or default",
            "- `yaw_source`: GT, motion estimate, camera/default prior, or unavailable",
            "- `is_gt_derived`: boolean guard for train/val diagnostics",
            "- `is_estimated_for_test`: boolean guard for test-time fields",
            "",
            "## Interpretation",
            "",
            str(report.get("interpretation", "")),
            "",
        ]
    )
    write_markdown("\n".join(lines), path)


def _read_record_tree(root_or_file: Union[str, Path], show_progress: bool, desc: str) -> List[Dict[str, Any]]:
    rows = []
    files = iter_data_files(root_or_file, [".csv", ".jsonl"])
    for path in progress_iter(files, show_progress, desc, "file"):
        if path.suffix.lower() == ".jsonl":
            rows.extend(read_jsonl_dicts(path))
        else:
            rows.extend(read_csv_dicts(path))
    return rows


def _scan_available_fields(
    root_or_file: Optional[Union[str, Path]],
    show_progress: bool,
    desc: str,
) -> Dict[str, Any]:
    if root_or_file is None:
        return {"available": False, "file_count": 0, "fields": []}
    files = iter_data_files(root_or_file, [".csv", ".jsonl", ".json"])
    fields = set()
    file_count = 0
    for path in progress_iter(files, show_progress, desc, "file"):
        file_count += 1
        fields.update(_fields_from_file(path))
    return {"available": bool(files), "file_count": file_count, "fields": sorted(fields)}


def _fields_from_file(path: Path) -> List[str]:
    if path.suffix.lower() == ".csv":
        rows = read_csv_dicts(path)
        return list(rows[0].keys()) if rows else []
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl_dicts(path)
        return list(rows[0].keys()) if rows else []
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if isinstance(data, dict):
            return list(data.keys())
    return []


def _first_present(record: Dict[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if field in record:
            return record.get(field)
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _has_explicit_source_metadata(record: Dict[str, Any]) -> bool:
    return any(field in record and record.get(field) not in (None, "") for field in RECOMMENDED_METADATA_FIELDS + ["source"])


def _field_seen(records: List[Dict[str, Any]], field: str) -> bool:
    for record in records:
        if field in record:
            return True
    return False


def _interpretation(source_counts: Dict[str, int], record_count: int, missing_metadata: List[str]) -> str:
    if record_count == 0:
        return "No frame records were found, so 3D source provenance could not be audited."
    unknown = int(source_counts.get("unknown", 0))
    ratio = float(unknown) / float(record_count)
    if ratio > 0.5:
        return (
            "Most rows have unknown 3D provenance. Treat the current Track1 3D fields as structurally valid "
            "but not provenance-certified until explicit metadata is written by the pipeline."
        )
    if missing_metadata:
        return "Some provenance is available, but the audit is incomplete until recommended metadata fields are added."
    return "3D source metadata appears available for the audited frame records."

