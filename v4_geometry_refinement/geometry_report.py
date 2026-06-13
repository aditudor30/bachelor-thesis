"""Human-readable Step 22C geometry report."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import output_root
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import read_json


def write_geometry_report(config: Dict[str, Any]) -> Path:
    """Write an honest Markdown summary of local proxy results."""
    root = output_root(config)
    comparison = read_json(root / "comparison" / "v4_geometry_refinement_summary.json")
    readiness = read_json(root / "frozen_candidate" / "comparison" / "upload_readiness.json")
    package = readiness.get("v4_geometry_refined_official", {})
    selected = comparison.get("selected_variant")
    lines: List[str] = [
        "# V4 Geometry Refinement Report",
        "",
        "## Executive Summary",
        "",
        "Step 22C starts from the frozen V3.1 coverage candidate and refines geometry only. No row key, object identity, class, scene or frame may change.",
        "The chosen result is based on strict official validation and conservative local geometry proxies.",
        "",
        "## Decision",
        "",
        "- Selected variant: `%s`" % (selected or "none"),
        "- Verdict: `%s`" % comparison.get("verdict", "unknown"),
        "- Frozen package ready: `%s`" % package.get("ready", False),
        "- Frozen Track1: `%s`" % package.get("track1_path", "not_available"),
        "- Upload ZIP: `%s`" % package.get("zip_path", "not_available"),
        "- Rows / unique tracks: `%s` / `%s`" % (package.get("rows"), package.get("unique_tracks")),
        "",
        "## V3.1 Input Audit",
        "",
        "- Rows: `%s`" % comparison.get("baseline_metrics", {}).get("rows"),
        "- Unique tracks: `%s`" % comparison.get("baseline_metrics", {}).get("unique_tracks"),
        "- Step p95: `%s`" % _fmt(comparison.get("baseline_metrics", {}).get("step_p95")),
        "- Suspect tracks: `%s`" % comparison.get("baseline_metrics", {}).get("suspect_track_count"),
        "- Dimension variance mean: `%s`" % _fmt(comparison.get("baseline_metrics", {}).get("dimension_variance_mean")),
        "- Yaw jumps: `%s`" % comparison.get("baseline_metrics", {}).get("yaw_jump_count"),
        "",
        "## Variants",
        "",
        "- `v4_smooth_only`: robust gap-aware moving median for x/y/z.",
        "- `v4_outlier_repair`: conservative interpolation of isolated jump-and-return or z outliers.",
        "- `v4_dimension_consistency`: robust per-track dimensions with optional mapped class priors.",
        "- `v4_yaw_refinement`: circular smoothing, with cautious heading blending only for vehicle-like classes.",
        "- `v4_geometry_refined_balanced`: sequential conservative combination of all four stages.",
        "",
        "## Safety",
        "",
        "V4 preserves scene, class, object and frame identity. It may change only x, y, z, dimensions and yaw.",
        "Every selected package is revalidated against the exact V3.1 row-key set before zipping.",
        "Optional dimension priors use the explicit official-to-internal class mapping from the resolved YAML; Track1 class IDs are never rewritten.",
        "",
        "## Variant Comparison",
        "",
        "| Variant | Valid | Aggressive | Step p95 | Suspect tracks | Dim variance | Yaw jumps | Mean position change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.get("variants", []):
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row.get("variant"), row.get("hard_valid"), row.get("too_aggressive"),
                _fmt(row.get("step_p95")), row.get("suspect_track_count"),
                _fmt(row.get("dimension_variance_mean")), row.get("yaw_jump_count"),
                _fmt(row.get("mean_position_change_m")),
            )
        )
    lines.extend([
        "",
        "## V2 / V3 Reference",
        "",
    ])
    for item in comparison.get("v2_v3_reference", []):
        lines.append("- `%s`: available=%s, rows=%s, unique_tracks=%s" % (
            item.get("variant"), item.get("available"), item.get("rows"), item.get("unique_tracks")))
    lines.extend([
        "",
        "These older candidates are comparison-only. Their geometry proxy deltas are marked `not_available` because their identity sets differ from V3.1.",
        "",
        "## Interpretation Limits",
        "",
        "Scenes 023-027 do not contain ground truth. Therefore Step 22C cannot prove improved 3D accuracy, purity, false-merge rate or fragmentation.",
        "Selection uses only local consistency proxies and conservative change limits. A lower proxy value is evidence of smoother geometry, not proof of a better official score.",
        "",
        "## Upload Position",
        "",
        "V3.1 remains the coverage-oriented and safer baseline until the official server confirms a gain. The selected V4 package is an optional fourth upload candidate intended to test geometry quality while preserving identity coverage.",
    ])
    path = root / "comparison" / "V4_GEOMETRY_REFINEMENT_OFFICIAL_023_027_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    try:
        return "%.6f" % float(value)
    except (TypeError, ValueError):
        return "n/a"
