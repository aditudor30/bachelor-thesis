"""Markdown/JSON report for Step 18D visual decision."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import write_json, write_text


def write_visual_decision_report(
    output_root: Path,
    summary: Dict[str, Any],
    final_decision: Dict[str, Any],
    review_paths: List[Path],
    figure_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Write the final Step 18D report artifacts."""
    root = Path(output_root)
    write_json(summary, root / "comparison" / "visual_decision_summary.json")
    write_json(final_decision, root / "comparison" / "final_variant_decision.json")
    lines = [
        "# Person ReID Visual Decision Report",
        "",
        "## Verdict",
        "",
        "- final verdict: `%s`" % final_decision.get("final_verdict"),
        "- reason: `%s`" % final_decision.get("reason"),
        "- selected Step 18C variant: `%s`" % final_decision.get("selected_variant_from_step18c"),
        "",
        "## Visual Review Summary",
        "",
        "- total review events: `%s`" % summary.get("total_review_events"),
        "- likely good: `%s`" % summary.get("likely_good_count"),
        "- suspicious or bad: `%s`" % summary.get("suspicious_or_bad_count"),
        "- not enough visual evidence: `%s`" % summary.get("not_enough_visual_evidence_count"),
        "- mean risk score: `%s`" % summary.get("mean_risk_score"),
        "",
        "## Review Sheets",
    ]
    for path in review_paths:
        lines.append("- `%s`" % path)
    lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )
    for key, value in sorted(figure_summary.items()):
        lines.append("- `%s`: `%s`" % (key, value))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This step does not rerun association or Track1 export. It audits already-produced ReID merge outputs and creates visual evidence for a conservative final decision.",
        ]
    )
    report_path = root / "comparison" / "PERSON_REID_VISUAL_DECISION_REPORT.md"
    write_text(lines, report_path)
    readme_path = root / "reports" / "README_PERSON_REID_VISUAL_DECISION.md"
    write_text(
        [
            "# Step 18D Outputs",
            "",
            "- `merge_audit/`: normalized selected merge events.",
            "- `visual_panels/`: PNG panels grouped by variant and auto label.",
            "- `manual_review/`: CSV sheets for human review.",
            "- `comparison/`: JSON/Markdown decision summaries.",
            "- `figures/`: aggregate diagnostics.",
        ],
        readme_path,
    )
    return {"report_path": str(report_path), "readme_path": str(readme_path)}

