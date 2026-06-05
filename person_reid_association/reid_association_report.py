"""Markdown reporting for ReID-guided Person association."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.person_reid_association.reid_association_io import write_json


def write_reid_association_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Write a concise Step 16B markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation = summary.get("best_reid_person_association_recommendation", {})
    lines = []
    lines.append("# Person ReID-guided Association Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- verdict: `%s`" % recommendation.get("verdict"))
    lines.append("- best_run: `%s`" % recommendation.get("best_run"))
    lines.append("")
    lines.append("## Baselines")
    lines.extend(_baseline_lines("v2_current", summary.get("v2_current", {})))
    lines.extend(_baseline_lines("v2_export_compact", summary.get("v2_export_compact", {})))
    lines.extend(_baseline_lines("v1_baseline", summary.get("v1_baseline", {})))
    lines.append("")
    lines.append("## ReID Runs")
    lines.append("")
    for run in summary.get("runs", []):
        lines.append(
            "- `%s`: status=%s, merges=%s, reid_pairs=%s, passing=%s, person_frag=%s, purity=%s, false_merge=%s, non_person_delta=%s"
            % (
                run.get("run_name"),
                run.get("run_status", "ok"),
                run.get("merges_applied"),
                run.get("pairs_with_both_reid"),
                run.get("pairs_passing_reid_threshold"),
                run.get("person_fragmentation_approx"),
                run.get("global_purity_mean"),
                run.get("false_merge_rate"),
                run.get("vs_v2_non_person_rows_delta"),
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "ReID is used only as a conservative additional gate over geometry/time/motion-compatible Person fragments. "
        "A run is useful only when Track1 validation stays clean, non-Person rows stay unchanged, and purity/false-merge deltas remain within the configured limits."
    )
    lines.append("")
    lines.append("## Suggested Decision")
    lines.append("")
    lines.append(
        "If no ReID run is selected, keep the V2/export-compact MVP and treat the result as evidence that the current OSNet embeddings need domain tuning before being used for automatic global ID merges."
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(_compact_summary(summary), output_path.parent / "PERSON_REID_GUIDED_ASSOCIATION_SUMMARY.json")


def _baseline_lines(name: str, metrics: Dict[str, Any]) -> List[str]:
    if not metrics:
        return ["", "- `%s`: unavailable" % name]
    return [
        "",
        "- `%s`: track1_rows=%s, person_rows=%s, non_person_rows=%s, person_frag=%s, purity=%s, false_merge=%s"
        % (
            name,
            metrics.get("track1_rows"),
            metrics.get("person_rows"),
            metrics.get("non_person_rows"),
            metrics.get("person_fragmentation_approx"),
            metrics.get("global_purity_mean"),
            metrics.get("false_merge_rate"),
        ),
    ]


def _compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    recommendation = summary.get("best_reid_person_association_recommendation", {})
    return {
        "verdict": recommendation.get("verdict"),
        "best_run": recommendation.get("best_run"),
        "num_runs": len(summary.get("runs", [])),
        "runs": [
            {
                "run_name": run.get("run_name"),
                "status": run.get("run_status", "ok"),
                "merges_applied": run.get("merges_applied"),
                "person_fragmentation_approx": run.get("person_fragmentation_approx"),
                "global_purity_mean": run.get("global_purity_mean"),
                "false_merge_rate": run.get("false_merge_rate"),
                "non_person_delta": run.get("vs_v2_non_person_rows_delta"),
            }
            for run in summary.get("runs", [])
        ],
    }
