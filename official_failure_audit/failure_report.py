"""Final JSON verdict and Markdown report for Step 23A."""

from pathlib import Path
from typing import Any, Dict, Sequence

from deep_oc_sort_3d.official_failure_audit.failure_io import write_json


def write_failure_report(
    config: Dict[str, Any], gt_summary: Dict[str, Any], pred_summary: Dict[str, Any],
    original: Dict[str, Any], sweep: Dict[str, Any], diagnosis: Dict[str, Any],
    figures: Dict[str, Any], output_root: Path,
) -> Dict[str, Any]:
    directory = output_root / "comparison"
    summary = {
        "status": "ok" if diagnosis.get("verdict") != "val_prediction_source_missing_fix_required" else "incomplete",
        "official_scores": config.get("official_scores", {}), "gt_audit": gt_summary,
        "prediction_source": pred_summary, "original_matching": original,
        "best_hypothesis": sweep.get("best", {}), "top_hypotheses": sweep.get("top", [])[:10],
        "diagnosis": diagnosis, "figures": figures,
    }
    verdict = {
        "verdict": diagnosis.get("verdict", "no_clear_convention_fix_found"),
        "likely_causes": diagnosis.get("likely_causes", []),
        "best_hypothesis": sweep.get("best", {}),
        "recommended_v6_fix": diagnosis.get("recommended_v6_fix"),
        "upload_recommendation": diagnosis.get("upload_recommendation"),
    }
    write_json(directory / "official_score_failure_audit_summary.json", summary)
    write_json(directory / "verdict.json", verdict)
    report_path = directory / "OFFICIAL_SCORE_FAILURE_AUDIT_23A_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(config, gt_summary, pred_summary, original, sweep, diagnosis), encoding="utf-8")
    return summary


def summarize_existing_root(root: Path) -> Dict[str, Any]:
    from deep_oc_sort_3d.official_failure_audit.failure_io import read_json

    summary = read_json(root / "comparison" / "official_score_failure_audit_summary.json")
    verdict = read_json(root / "comparison" / "verdict.json")
    return {"summary": summary, "verdict": verdict}


def _markdown(
    config: Dict[str, Any], gt: Dict[str, Any], pred: Dict[str, Any], original: Dict[str, Any],
    sweep: Dict[str, Any], diagnosis: Dict[str, Any],
) -> str:
    scores = config.get("official_scores", {})
    lines = [
        "# Official Score Failure Audit 23A", "", "## Executive Summary", "",
        "Verdict: `%s`." % diagnosis.get("verdict", "not_available"),
        "", diagnosis.get("recommended_v6_fix", "No recommendation available."), "",
        "## Official Scores", "",
        "| Variant | 3D HOTA | DetA | AssA | LocA |", "|---|---:|---:|---:|---:|",
        _score_row("V2", scores.get("v2", {})), _score_row("V5", scores.get("v5", {})), "",
        "Near-zero DetA indicates a systemic spatial or semantic mismatch. Very low AssA cannot be interpreted as a pure ReID failure while detection matching is nearly absent. LocA of 8-18 percent indicates weak localization even for accepted matches.",
        "", "## GT Structure Audit", "",
        "GT status: `%s`; rows: `%s`; tracks: `%s`; frame indexing: `%s`." % (
            gt.get("status"), gt.get("rows"), gt.get("tracks"), gt.get("frame_indexing_inference")),
        "", "Raw GT fields: `%s`." % sorted(gt.get("raw_object_fields", {}).keys()),
        "", "## Prediction Structure Audit", "",
        "Prediction status: `%s`; selected variant: `%s`; rows: `%s`." % (
            pred.get("status"), pred.get("selected_variant"), pred.get("selected_rows")),
        "", "The source finder rejected GT-derived rows and records provenance in `pred_audit/pred_source_summary.json`.",
        "", "## Original Matching", "", _metric_table(original),
        "", "## Top 10 Hypotheses", "", _top_table(sweep.get("top", [])[:10]),
        "", "## Individual and Combined Sweep", "",
        "Full individual and combined results are stored under `hypothesis_sweep/`. All hypotheses use identical prediction and GT rows; only the declared prediction convention transform changes.",
        "", "## Likely Failure Cause", "",
        "Likely causes: `%s`." % diagnosis.get("likely_causes", []),
        "", "Best hypothesis: `%s`." % sweep.get("best", {}).get("hypothesis", "not_available"),
        "", "## Recommended V6 Fix", "", diagnosis.get("recommended_v6_fix", "Not available."),
        "", "## What Not To Optimize Yet", "",
    ]
    lines.extend("- %s" % value for value in diagnosis.get("do_not_optimize_before_fix", []))
    lines.extend([
        "", "## Upload Recommendation", "", diagnosis.get("upload_recommendation", "Not available."),
        "", "## Honesty and Limitations", "",
        "- This is a diagnostic matcher, not a reimplementation of the official evaluator.",
        "- Axis-aligned 3D IoU ignores yaw when constructing overlap volumes.",
        "- If no hypothesis improves matching, the likely issue is projection/localization quality rather than a simple convention.",
        "- If comparable validation predictions are missing, no geometric conclusion should be used for an upload.",
    ])
    return "\n".join(lines) + "\n"


def _score_row(name: str, values: Dict[str, Any]) -> str:
    return "| %s | %s%% | %s%% | %s%% | %s%% |" % (
        name, values.get("hota_3d_percent", ""), values.get("deta_percent", ""),
        values.get("assa_percent", ""), values.get("loca_percent", ""),
    )


def _metric_table(values: Dict[str, Any]) -> str:
    rows = ["| Metric | Value |", "|---|---:|"]
    for key in [
        "num_predictions", "num_gt", "num_matches", "match_rate_at_0_5m", "match_rate_at_1m",
        "match_rate_at_2m", "match_rate_at_5m", "match_rate_at_10m",
        "center_error_median", "center_error_p95", "yaw_error_median", "iou3d_proxy_median",
    ]:
        rows.append("| %s | %s |" % (key, values.get(key)))
    return "\n".join(rows)


def _top_table(values: Sequence[Dict[str, Any]]) -> str:
    rows = ["| Rank | Hypothesis | Match@2m | Center median | IoU median |", "|---:|---|---:|---:|---:|"]
    for index, row in enumerate(values, 1):
        rows.append("| %d | %s | %s | %s | %s |" % (
            index, row.get("hypothesis"), row.get("match_rate_at_2m"),
            row.get("center_error_median"), row.get("iou3d_proxy_median"),
        ))
    return "\n".join(rows)
