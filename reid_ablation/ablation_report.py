"""Markdown report for Step 16C ReID ablation decision."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.reid_ablation.ablation_io import write_json


def write_reid_ablation_report(result: Dict[str, Any], output_root: Path) -> None:
    """Write the final Step 16C report and compact JSON summary."""
    comparison = result.get("comparison", {})
    decision = result.get("decision", {})
    rows = comparison.get("variants", [])
    report_path = output_root / "report" / "REID_ABLATION_DECISION_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# ReID Ablation Decision Report")
    lines.append("")
    lines.append("## Final Verdicts")
    lines.append("")
    for verdict in decision.get("verdicts", []):
        lines.append("- `%s`" % verdict)
    lines.append("")
    lines.append("## Kept Variants")
    lines.append("")
    for key, value in sorted(decision.get("kept_variants", {}).items()):
        lines.append("- `%s`: `%s`" % (key, value))
    lines.append("")
    lines.append("## Variant Table")
    lines.append("")
    lines.append("| variant | type | valid | safe | source | Track1 rows | Person frag | ReID merges | export drops | recommendation |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("variant_name"),
                row.get("source_type"),
                row.get("track1_valid"),
                row.get("is_safe"),
                row.get("improvement_source"),
                row.get("track1_rows"),
                row.get("person_fragmentation"),
                row.get("num_reid_merges"),
                row.get("num_export_dropped_rows"),
                row.get("recommendation"),
            )
        )
    lines.append("")
    lines.append("## ReID-only vs Export Compact")
    lines.append("")
    lines.append(
        "ReID-only runs are useful only if they reduce fragmentation or rows without degrading purity/false-merge metrics. "
        "If `reid_with_export_compact` improves while ReID-only does not, the gain should be attributed to `export_compact`, not to ReID."
    )
    lines.append("")
    lines.append("## Honest Interpretation")
    lines.append("")
    lines.append(
        "OSNet ReID is connected and diagnostically useful, but off-the-shelf embeddings should not be treated as a final automatic merge signal unless the comparison marks a safe real ReID-only upgrade."
    )
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append(decision.get("final_recommendation", "not_available"))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(_summary(result), output_root / "report" / "REID_ABLATION_DECISION_SUMMARY.json")


def _summary(result: Dict[str, Any]) -> Dict[str, Any]:
    decision = result.get("decision", {})
    rows = result.get("comparison", {}).get("variants", [])
    return {
        "verdicts": decision.get("verdicts", []),
        "kept_variants": decision.get("kept_variants", {}),
        "final_recommendation": decision.get("final_recommendation"),
        "variants": [
            {
                "variant_name": row.get("variant_name"),
                "source_type": row.get("source_type"),
                "improvement_source": row.get("improvement_source"),
                "is_safe": row.get("is_safe"),
                "is_noop": row.get("is_noop"),
                "real_upgrade": row.get("real_upgrade"),
                "recommendation": row.get("recommendation"),
            }
            for row in rows
        ],
    }
