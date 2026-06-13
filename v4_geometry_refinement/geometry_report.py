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
        "## Decision",
        "",
        "- Selected variant: `%s`" % (selected or "none"),
        "- Verdict: `%s`" % comparison.get("verdict", "unknown"),
        "- Frozen package ready: `%s`" % package.get("ready", False),
        "",
        "## Safety",
        "",
        "V4 preserves scene, class, object and frame identity. It may change only x, y, z, dimensions and yaw.",
        "Every selected package is revalidated against the exact V3.1 row-key set before zipping.",
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
        "## Interpretation Limits",
        "",
        "Scenes 023-027 do not contain ground truth. Therefore Step 22C cannot prove improved 3D accuracy, purity, false-merge rate or fragmentation.",
        "Selection uses only local consistency proxies and conservative change limits. A lower proxy value is evidence of smoother geometry, not proof of a better official score.",
        "",
        "## Upload Position",
        "",
        "V3.1 remains the coverage-oriented baseline. The selected V4 package is an optional fourth upload candidate intended to test geometry quality while preserving identity coverage.",
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
