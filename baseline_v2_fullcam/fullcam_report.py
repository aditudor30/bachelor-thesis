"""Markdown report generation for baseline_v2_pseudo3d_fullcam."""

from pathlib import Path
from typing import Any, Dict, List


def build_fullcam_report(summary: Dict[str, Any]) -> str:
    """Build the Step 15H Markdown report."""
    v1 = summary.get("baseline_v1", {})
    v2 = summary.get("baseline_v2_fullcam", {})
    deltas = summary.get("deltas", {})
    verdict = summary.get("verdict", {})
    lines = [
        "# Baseline V1 vs V2 Pseudo3D Fullcam Report",
        "",
        "## Executive Summary",
        "",
        "- V1 baseline: `baseline_v1_geometry_only`.",
        "- V2 baseline: `baseline_v2_pseudo3d_fullcam`.",
        "- Verdict: `%s`." % verdict.get("label"),
        "- Reasons: %s." % ", ".join([str(item) for item in verdict.get("reasons", [])]),
        "",
        "## Pseudo3D Usage",
        "",
        "- V2 observations: %s" % _nested(v2, ["observations", "output_observations"]),
        "- V2 pseudo3D used rate: %s" % _nested(v2, ["observations", "pseudo3d_used_rate"]),
        "- V2 fallback original used rate: %s" % _nested(v2, ["observations", "fallback_original_used_rate"]),
        "- V2 source metadata completeness: %s" % _nested(v2, ["observations", "metadata_complete_rate"]),
        "",
        "## Track1 Validation",
        "",
        "- V1 Track1 rows: %s" % _nested(v1, ["track1", "rows"]),
        "- V2 Track1 rows: %s" % _nested(v2, ["track1", "rows"]),
        "- V2 validation errors: %s" % _nested(v2, ["track1", "validation_errors"]),
        "",
        "## Local Tracking",
        "",
        "- V2 local track records: %s" % _nested(v2, ["local_tracking", "local_track_records"]),
        "- V2 active tracks: %s" % _nested(v2, ["local_tracking", "active_tracks"]),
        "",
        "## Tracklets And Candidates",
        "",
        "- V2 total tracklets: %s" % _nested(v2, ["tracklets", "total_tracklets"]),
        "- V2 kept candidates: %s" % _nested(v2, ["candidates", "kept_candidates"]),
        "",
        "## Motion Filtering",
        "",
        "- V2 motion good: %s" % _nested(v2, ["motion_clean", "motion_good"]),
        "- V2 motion invalid: %s" % _nested(v2, ["motion_clean", "motion_invalid"]),
        "",
        "## Global Association",
        "",
        "- V1 global tracks: %s" % _nested(v1, ["global_association", "global_tracks"]),
        "- V2 global tracks: %s" % _nested(v2, ["global_association", "global_tracks"]),
        "- V1 multi-camera tracks: %s" % _nested(v1, ["global_association", "multi_camera_tracks"]),
        "- V2 multi-camera tracks: %s" % _nested(v2, ["global_association", "multi_camera_tracks"]),
        "- V2 global purity mean: %s" % _nested(v2, ["global_association", "global_purity_mean"]),
        "- V2 false merge rate: %s" % _nested(v2, ["global_association", "false_merge_rate"]),
        "- V2 fragmentation approx: %s" % _nested(v2, ["global_association", "fragmentation_approx"]),
        "",
        "## Metric Deltas",
        "",
    ]
    for key in sorted(deltas.keys()):
        lines.append("- %s: %s" % (key, deltas.get(key)))
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            _recommendation(verdict),
            "",
        ]
    )
    return "\n".join(lines)


def write_fullcam_report(summary: Dict[str, Any], path: Path) -> None:
    """Write the fullcam Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_fullcam_report(summary), encoding="utf-8")


def _recommendation(verdict: Dict[str, Any]) -> str:
    label = str(verdict.get("label", ""))
    if label == "baseline_v2_fullcam_ready_for_submission":
        return "V2 is Track1-valid and can be considered as a submission candidate after visual sanity checks."
    if label == "baseline_v2_fullcam_needs_tracking_tuning":
        return "Keep V1 for submission and use V2 as the provenance-backed 3D MVP until tracking is tuned."
    if label == "baseline_v2_fullcam_valid_but_not_submission_candidate":
        return "Keep V1 for submission; V2 is useful for diagnostics and provenance."
    return "V2 is not ready for submission; fix validation or pseudo3D coverage first."


def _nested(data: Dict[str, Any], keys: List[str]) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
