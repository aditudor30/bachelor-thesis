"""Build final submission summary artifacts."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union


def build_submission_summary(
    track1_report: Dict[str, Any],
    dedup_report: Dict[str, Any],
    config_paths: List[Path],
    baseline_name: str,
) -> Dict[str, Any]:
    """Build a final submission summary dictionary."""
    distribution = track1_report.get("distribution", {})
    return {
        "baseline_name": baseline_name,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "track1_path": track1_report.get("track1_path", ""),
        "total_rows": track1_report.get("total_rows", distribution.get("total_rows", 0)),
        "scenes_present": sorted(distribution.get("per_scene_rows", {}).keys()),
        "classes_present": sorted(distribution.get("per_class_rows", {}).keys()),
        "validation_errors": int(track1_report.get("num_errors", 0)),
        "validation_warnings": int(track1_report.get("num_warnings", 0)),
        "dedup": {
            "generic_rows_total": dedup_report.get("generic_rows_total"),
            "official_rows_total": dedup_report.get("official_rows_total"),
            "duplicate_rows_removed_estimated": dedup_report.get("duplicate_rows_removed_estimated"),
            "dedup_ratio": dedup_report.get("dedup_ratio"),
            "dedup_rule": dedup_report.get("dedup_rule"),
        },
        "configs_used": [str(path) for path in config_paths],
        "known_limitations": [
            "No ReID.",
            "Geometry-only global association.",
            "No neural 3D bbox head.",
            "Detector remains weaker for some rare classes.",
        ],
    }


def write_submission_summary_json(summary: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write submission summary JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def write_submission_summary_markdown(summary: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write submission summary Markdown."""
    output_path = Path(path)
    lines = []
    lines.append("# Submission Summary")
    lines.append("")
    lines.append("- baseline_name: `%s`" % summary.get("baseline_name"))
    lines.append("- timestamp_utc: `%s`" % summary.get("timestamp_utc"))
    lines.append("- track1_path: `%s`" % summary.get("track1_path"))
    lines.append("- total_rows: `%s`" % summary.get("total_rows"))
    lines.append("- validation_errors: `%s`" % summary.get("validation_errors"))
    lines.append("- validation_warnings: `%s`" % summary.get("validation_warnings"))
    lines.append("")
    lines.append("## Scenes")
    lines.append("")
    for scene in summary.get("scenes_present", []):
        lines.append("- `%s`" % scene)
    lines.append("")
    lines.append("## Classes")
    lines.append("")
    for class_id in summary.get("classes_present", []):
        lines.append("- `%s`" % class_id)
    lines.append("")
    lines.append("## Deduplication")
    lines.append("")
    dedup = summary.get("dedup", {})
    for key in sorted(dedup.keys()):
        lines.append("- %s: `%s`" % (key, dedup.get(key)))
    lines.append("")
    lines.append("## Known Limitations")
    lines.append("")
    for item in summary.get("known_limitations", []):
        lines.append("- %s" % item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
