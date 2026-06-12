"""Markdown report for Step 21A."""

from pathlib import Path
from typing import Any, Dict, List, Optional


def write_benchmark_report(
    path: Path,
    rows: List[Dict[str, Any]],
    selected: Dict[str, Any],
    warnings: List[str],
    probes: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Write a direct benchmark interpretation."""
    lines = [
        "# Local Tracker Benchmark Report",
        "",
        "## Context",
        "",
        "Step 20C showed that global ReID/MLP association cannot reliably repair the large number of short local fragments. Step 21A therefore compares local trackers over the same YOLO11m SmartSpaces detections.",
        "",
        "## Tracker variants",
        "",
        "`bytetrack_style_yolo11m` and the BoT-SORT variants are internal style-compatible implementations, not claims of bit-exact reproduction of the external repositories.",
        "",
        "| Tracker | Status | Tracks | Mean length | Median length | <=3 ratio | Person median | Non-Person <=3 | Purity | False merge suspicion | Runtime s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("tracker_name"), row.get("status"), row.get("num_tracks"),
                row.get("mean_track_length"), row.get("median_track_length"),
                row.get("short_track_ratio_le3"), row.get("person_median_track_length"),
                row.get("nonperson_short_track_ratio_le3"),
                row.get("local_purity_mean"), row.get("false_merge_suspicion_rate"),
                row.get("runtime_seconds"),
            )
        )
    lines.extend(
        [
            "",
            "## Downstream probe",
            "",
            "| Tracker | Tracklets | Short ratio | Candidate pressure | Motion good | Suspicious | Invalid |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for probe in probes or []:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                probe.get("tracker_name"),
                probe.get("tracklet_count_probe"),
                probe.get("short_tracklet_ratio"),
                probe.get("candidate_count_probe"),
                probe.get("motion_good"),
                probe.get("motion_suspicious"),
                probe.get("motion_invalid"),
            )
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            "- Selected tracker: %s" % selected.get("selected_tracker"),
            "- Verdict: `%s`" % selected.get("verdict"),
            "",
            "Selection prioritizes longer tracks and fewer short fragments while protecting local purity, non-Person behavior and runtime. A smaller number of tracks is not sufficient by itself.",
            "",
            "## Risks",
            "",
            "- Internal ByteTrack/BoT-SORT-style variants are controlled baselines, not bit-exact external repository reproductions.",
            "- The OSNet variant performs lazy video seeks and Person crop inference, so it is expected to be substantially slower.",
            "- Current tracker runtime is not directly comparable because existing outputs are loaded rather than regenerated.",
            "- GT metrics are diagnostics only and are never used by tracker association or on the test split.",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend(["- %s" % warning for warning in warnings] or ["- None"])
    lines.extend(
        [
            "",
            "## Recommended next step",
            "",
            "Run the selected tracker through a controlled Step 21B subset rerun, then rebuild tracklets and compare downstream fragmentation before touching global association.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
