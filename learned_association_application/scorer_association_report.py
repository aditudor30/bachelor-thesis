"""Markdown report generation for Step 20C."""

from pathlib import Path
from typing import Any, Dict, List


def write_report(
    path: Path,
    score_summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    baselines: Dict[str, Dict[str, Any]],
    selected: Dict[str, Any],
) -> None:
    """Write an honest, compact association sweep report."""
    lines = [
        "# Person Scorer Association Report",
        "",
        "## Context",
        "",
        "Step 20C applies the Step 20B MLP as an additional conservative gate over ReID, geometry, time and graph conflicts. It never replaces the hard safety constraints.",
        "",
        "## Scorer coverage",
        "",
        "- Candidate pairs: %s" % score_summary.get("candidate_pairs"),
        "- Pairs with ReID: %s" % score_summary.get("pairs_with_reid"),
        "- Pairs with geometry: %s" % score_summary.get("pairs_with_geometry"),
        "- MLP score median: %s" % score_summary.get("score_median"),
        "- MLP score p95: %s" % score_summary.get("score_p95"),
        "",
        "## Sweep metrics",
        "",
        "| Variant | Track1 valid | Person tracks | Person fragmentation | Person purity | Person false merge rate | Accepted edges |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("run_name"), row.get("track1_valid"), row.get("person_global_tracks"),
                row.get("person_fragmentation"), row.get("person_purity_mean"),
                row.get("person_false_merge_rate"), row.get("accepted_edges"),
            )
        )
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "Available baselines: %s." % ", ".join([name for name, value in baselines.items() if value.get("status") != "not_available"]),
            "",
            "## Selection",
            "",
            "- Selected variant: %s" % selected.get("selected_variant"),
            "- Verdict: `%s`" % selected.get("verdict"),
            "",
            "The learned scorer must remain conservative. A lower row count alone is not evidence of better association, and any gain attributed to export_compact must be separated from the MLP merge effect.",
            "",
            "## Step 20D",
            "",
            "Visually inspect accepted merges from the selected variant and a score-stratified sample of rejected edges before promoting the variant.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
