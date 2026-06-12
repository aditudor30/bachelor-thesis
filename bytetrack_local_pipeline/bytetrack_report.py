"""Markdown reporting for Step 21B."""

from pathlib import Path
from typing import Any, Dict


def write_bytetrack_report(summary: Dict[str, Any], path: Path) -> None:
    """Write an honest local-to-global propagation report."""
    variants = summary.get("variants", {})
    current = variants.get("baseline_v2_pseudo3d_fullcam", {})
    candidate = variants.get("baseline_v2_pseudo3d_fullcam_bytetrack_local", {})
    precheck = summary.get("precheck", {})
    lines = [
        "# Baseline V2 ByteTrack Local Report",
        "",
        "## Context",
        "",
        "Steps 20C and 21A showed that the main bottleneck starts in single-camera tracking. Step 21B replaces only the local tracker while preserving YOLO11m detections and V2 pseudo3D observations.",
        "",
        "## Precheck",
        "",
        "- Verdict: `%s`" % precheck.get("label", "not_available"),
        "",
        "## Local tracking",
        "",
        "| Metric | V2 current | V2 ByteTrack local |",
        "|---|---:|---:|",
    ]
    for metric in (
        "num_records", "num_tracks", "mean_track_length", "median_track_length",
        "short_track_ratio_le3", "approx_id_switches", "approx_fragmentation",
        "local_purity_mean", "false_merge_suspicion_rate",
    ):
        lines.append(
            "| %s | %s | %s |" % (
                metric,
                current.get("local_tracking", {}).get(metric),
                candidate.get("local_tracking", {}).get(metric),
            )
        )
    lines.extend(
        [
            "",
            "## Global and Track1",
            "",
            "- Candidate global tracks: %s" % candidate.get("global_association", {}).get("global_tracks"),
            "- Candidate multi-camera tracks: %s" % candidate.get("global_association", {}).get("multi_camera_tracks"),
            "- Candidate global purity: %s" % candidate.get("global_association", {}).get("global_purity_mean"),
            "- Candidate false merge rate: %s" % candidate.get("global_association", {}).get("false_merge_rate"),
            "- Candidate global fragmentation: %s" % candidate.get("global_association", {}).get("fragmentation_approx"),
            "- Track1 rows: %s" % candidate.get("track1", {}).get("rows"),
            "- Track1 validation errors: %s" % candidate.get("track1", {}).get("validation_errors"),
            "",
            "## Verdict",
            "",
            "`%s`" % summary.get("verdict", {}).get("label"),
            "",
            "Reasons: %s" % ", ".join(summary.get("verdict", {}).get("reasons", [])),
            "",
            "## Risks",
            "",
            "- Fewer local records can improve fragmentation metrics while reducing detector coverage.",
            "- Longer local tracks may increase false merges; global purity and false-merge rate remain hard safeguards.",
            "- ReID is intentionally disabled in this baseline so the effect is attributable to local tracking.",
            "",
            "## Recommendation",
            "",
            "Proceed to Step 21C only if Track1 is valid and the local gain survives global association without an unacceptable purity or false-merge regression.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
