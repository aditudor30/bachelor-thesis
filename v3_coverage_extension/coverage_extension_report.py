"""Generate the final Step 22B Markdown report."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import output_root
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import read_json


def write_final_report(config: Dict[str, Any]) -> Path:
    """Write a concise, evidence-based V2/V3/V3.1 report."""
    root = output_root(config)
    audit = root / "audit"
    v2 = read_json(audit / "v2_official_reference_summary.json")
    v3 = read_json(audit / "v3_official_baseline_summary.json")
    comparison = read_json(root / "comparison" / "v3_coverage_extension_summary.json")
    selected = read_json(root / "comparison" / "selected_variant.json")
    verdict = read_json(root / "comparison" / "verdict.json")
    readiness = read_json(root / "frozen_candidate" / "comparison" / "upload_readiness.json")
    lines = [
        "# V3 Coverage Extension Official 023-027",
        "",
        "## Executive summary",
        "",
        "Step 22B extends V3 only from its own ByteTrack/local-track/pseudo3D artifacts. V2 is used solely as a coverage reference; no V2 rows are copied.",
        "",
        "- Verdict: `%s`" % verdict.get("label", "not_available"),
        "- Selected variant: `%s`" % selected.get("selected_variant", "none"),
        "- Upload ready: `%s`" % readiness.get("v3_coverage_extended_official", {}).get("ready", False),
        "",
        "## Official baselines",
        "",
        "| Candidate | Rows | Unique tracks | Rows/track mean |",
        "|---|---:|---:|---:|",
        "| V2 official | %s | %s | %s |" % (v2.get("rows"), v2.get("unique_tracks"), v2.get("rows_per_track_mean")),
        "| V3 official | %s | %s | %s |" % (v3.get("rows"), v3.get("unique_tracks"), v3.get("rows_per_track_mean")),
        "",
        "V3 has substantially longer identities but lower row coverage, especially in scenes 024, 026 and 023 and in Person/Forklift.",
        "",
        "## Coverage gaps by scene",
        "",
        "| Scene | V2 rows | V3 rows | V3-V2 |",
        "|---:|---:|---:|---:|",
    ]
    for scene_id in sorted(set(v2.get("scene_distribution", {}).keys()).union(v3.get("scene_distribution", {}).keys()), key=int):
        v2_count = int(v2.get("scene_distribution", {}).get(scene_id, 0))
        v3_count = int(v3.get("scene_distribution", {}).get(scene_id, 0))
        lines.append("| %s | %d | %d | %d |" % (scene_id, v2_count, v3_count, v3_count - v2_count))
    lines.extend(["", "## Coverage gaps by class", "", "| Official class | V2 rows | V3 rows | V3-V2 |", "|---:|---:|---:|---:|"])
    for class_id in sorted(set(v2.get("class_distribution", {}).keys()).union(v3.get("class_distribution", {}).keys()), key=int):
        v2_count = int(v2.get("class_distribution", {}).get(class_id, 0))
        v3_count = int(v3.get("class_distribution", {}).get(class_id, 0))
        lines.append("| %s | %d | %d | %d |" % (class_id, v2_count, v3_count, v3_count - v2_count))
    lines.extend([
        "", "## Variants", "",
        "| Variant | Rows | Gain vs V3 | Tracks | Errors | Duplicates | NaN/inf | Bad dims | Rounding |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison.get("variants", []):
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row.get("variant"), row.get("track1_rows"), row.get("row_gain_vs_v3"), row.get("unique_tracks"),
            row.get("validation_errors"), row.get("duplicate_keys"), row.get("nan_inf"),
            row.get("non_positive_dimensions"), row.get("rounding_issues"),
        ))
    lines.extend(["", "### Recovery distribution", ""])
    for row in comparison.get("variants", []):
        lines.append("")
        lines.append("`%s` additions by scene: `%s`; by class: `%s`." % (row.get("variant"), row.get("added_rows_by_scene", {}), row.get("added_rows_by_class", {})))
    lines.extend([
        "", "## Validation and safety", "",
        "Every selectable candidate must contain scenes 23-27, official classes only, two-decimal floats, positive dimensions, finite values and no duplicate keys.",
        "Optional purity, false-merge and fragmentation metrics are reported as `not_available` when test-only artifacts cannot support them.",
        "", "## Decision", "",
        "Selection failures: `%s`" % selected.get("selection_failures", []),
        "Recommendation: %s" % verdict.get("recommendation", "Keep V3 official unless V3.1 passes every gate."),
    ])
    frozen = readiness.get("v3_coverage_extended_official", {})
    if frozen.get("ready"):
        lines.extend(["", "## Upload file", "", "Upload `%s` if a third official candidate is allowed." % frozen.get("zip_path")])
    else:
        lines.extend(["", "## Upload file", "", "No V3.1 package is recommended. V3 official remains the identity-quality candidate."])
    path = root / "comparison" / "V3_COVERAGE_EXTENSION_OFFICIAL_023_027_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
