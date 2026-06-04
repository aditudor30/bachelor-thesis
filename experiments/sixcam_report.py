"""Markdown reporting for the pseudo-3D 6-camera experiment."""

from typing import Any, Dict, List

from deep_oc_sort_3d.experiments.sixcam_subset import SixCamItem


def build_sixcam_report(summary: Dict[str, Any], subset_items: List[SixCamItem]) -> str:
    """Build the final 6cam comparison Markdown report."""
    verdict = summary.get("verdict", {})
    deltas = summary.get("deltas", {})
    return "\n".join(
        [
            "# Baseline V1 vs V2 Pseudo3D 6Cam Report",
            "",
            "## Executive Summary",
            "",
            "This experiment compares geometry-only V1 and pseudo3D V2 on the six cameras where stabilized pseudo3D exists.",
            "It is intended as an apples-to-apples local/3D quality test, not a complete MTMC stress test.",
            "",
            "## Subset Definition",
            "",
            _subset_lines(subset_items),
            "",
            "## Pseudo3D Coverage",
            "",
            "- V2 pseudo3d_used_rate: %s" % deltas.get("v2_pseudo3d_used_rate"),
            "",
            "## Metric Deltas",
            "",
            "- Observations delta: %s" % deltas.get("observations_delta"),
            "- Local records delta: %s" % deltas.get("local_records_delta"),
            "- Active tracks delta: %s" % deltas.get("active_tracks_delta"),
            "- Tracklets delta: %s" % deltas.get("tracklets_delta"),
            "- Valid tracklets delta: %s" % deltas.get("valid_tracklets_delta"),
            "- Candidates delta: %s" % deltas.get("candidates_delta"),
            "- Motion invalid delta: %s" % deltas.get("motion_invalid_delta"),
            "- Final rows delta: %s" % deltas.get("final_rows_delta"),
            "",
            "## Limitations",
            "",
            "The subset contains one camera per scene, so multi-camera MTMC conclusions are limited. Use this mainly for local tracking, 3D provenance, and smoothness diagnostics.",
            "",
            "## Verdict",
            "",
            "- Label: %s" % verdict.get("label"),
            "- Reason: %s" % verdict.get("reason"),
            "",
        ]
    )


def _subset_lines(items: List[SixCamItem]) -> str:
    lines = []
    for item in items:
        lines.append("- %s/%s/%s" % (item.subset, item.scene_name, item.camera_id))
    return "\n".join(lines)

