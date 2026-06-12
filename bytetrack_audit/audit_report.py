"""Orchestration, comparisons and final report for Step 21D."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.bytetrack_audit.artifact_inventory import build_artifact_inventories
from deep_oc_sort_3d.bytetrack_audit.audit_config import output_root, write_resolved_config
from deep_oc_sort_3d.bytetrack_audit.audit_figures import write_audit_figures
from deep_oc_sort_3d.bytetrack_audit.audit_io import write_csv, write_json
from deep_oc_sort_3d.bytetrack_audit.bottleneck_ranker import rank_bottlenecks
from deep_oc_sort_3d.bytetrack_audit.drop_trace_sampler import build_drop_trace_samples
from deep_oc_sort_3d.bytetrack_audit.fix_recommender import recommend_fix
from deep_oc_sort_3d.bytetrack_audit.gt_object_frame_audit import run_gt_object_frame_audit
from deep_oc_sort_3d.bytetrack_audit.lifecycle_audit import run_lifecycle_audit
from deep_oc_sort_3d.bytetrack_audit.motion_filter_audit import run_motion_filter_audit
from deep_oc_sort_3d.bytetrack_audit.stage_retention_audit import run_stage_retention_audit


def run_complete_audit(
    config: Dict[str, Any],
    progress: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
    artifact_only: bool = False,
    instrumented_mini_rerun: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all Step 21D diagnostic stages."""
    del overwrite, skip_existing
    write_resolved_config(config)
    inventories = build_artifact_inventories(config, progress=progress)
    lifecycle = run_lifecycle_audit(
        config,
        progress=progress,
        artifact_only=artifact_only or not instrumented_mini_rerun,
        instrumented_mini_rerun=instrumented_mini_rerun,
    )
    stage = run_stage_retention_audit(config, progress=progress, inventories=inventories)
    gt_result = run_gt_object_frame_audit(config, progress=progress, max_samples=max_samples)
    motion = run_motion_filter_audit(config, progress=progress)
    samples = build_drop_trace_samples(config, gt_result, max_samples=max_samples)
    return finalize_audit(config, inventories, lifecycle, stage, gt_result, motion, samples)


def finalize_audit(
    config: Dict[str, Any],
    inventories: Dict[str, Any],
    lifecycle: Dict[str, Any],
    stage: Dict[str, Any],
    gt_result: Dict[str, Any],
    motion: Dict[str, Any],
    samples: Dict[str, Any],
) -> Dict[str, Any]:
    """Write comparisons, recommendation, verdict, report and figures."""
    root = output_root(config)
    stage_rows = stage.get("rows", [])
    bottlenecks = rank_bottlenecks(stage_rows, gt_result)
    recommendation = recommend_fix(lifecycle, stage_rows, gt_result, motion)
    verdict = {
        "label": recommendation.get("verdict"),
        "reasons": recommendation.get("reasons", []),
        "recommended_step_21e": recommendation.get("recommended_step_21e"),
    }
    comparison = root / "comparison"
    write_csv(comparison / "v2_current_vs_bytetrack_21b_audit.csv", _comparison_rows(inventories, "bytetrack_21b"))
    write_csv(
        comparison / "v2_current_vs_bytetrack_21c_best_audit.csv",
        _comparison_rows(inventories, "bytetrack_21c_best"),
    )
    write_csv(comparison / "bottleneck_ranking.csv", bottlenecks)
    write_json(comparison / "recommended_fix_plan.json", recommendation)
    write_json(comparison / "verdict.json", verdict)
    result = {
        "inventories": inventories,
        "lifecycle": lifecycle,
        "stage_retention": stage,
        "gt_audit": gt_result,
        "motion_audit": motion,
        "samples": samples,
        "bottlenecks": bottlenecks,
        "recommendation": recommendation,
        "verdict": verdict,
    }
    write_audit_report(result, comparison / "BYTETRACK_LIFECYCLE_AND_FILTERING_AUDIT_REPORT.md")
    if bool(config.get("figures", {}).get("enabled", True)):
        write_audit_figures(stage_rows, lifecycle, gt_result, motion, root / "figures")
    write_json(comparison / "audit_summary.json", _compact_result(result))
    return result


def write_audit_report(result: Dict[str, Any], path: Path) -> None:
    """Write the final honest Markdown report."""
    recommendation = result.get("recommendation", {})
    lifecycle = result.get("lifecycle", {})
    bottlenecks = result.get("bottlenecks", [])
    gt_rows = result.get("gt_audit", {}).get("summary_rows", [])
    motion_rows = result.get("motion_audit", {}).get("summary_rows", [])
    lines = [
        "# ByteTrack Lifecycle and Filtering Audit",
        "",
        "## Executive summary",
        "",
        "Verdict: `%s`" % recommendation.get("verdict"),
        "",
        "Recommended Step 21E: %s" % recommendation.get("recommended_step_21e"),
        "",
        "Reasons: %s" % ", ".join(recommendation.get("reasons", [])),
        "",
        "## Context",
        "",
        "Step 21B and 21C produced longer, cleaner local tracks but retained too little final Track1 coverage. This audit separates strict retention from ratios between different semantic units.",
        "",
        "## Measurement units",
        "",
        "Strict comparisons are limited to observation records -> local records, tracklets -> candidates, candidates -> motion-clean candidates, and GT object-frames across frame-level stages. Other transitions are explicitly diagnostic-only.",
        "",
        "## ByteTrack lifecycle and export policy",
        "",
        "Artifact-only rows can measure input-versus-exported detection records and exported tentative/confirmed states. They cannot prove lost/removed counts or associated-but-not-exported detections.",
    ]
    if lifecycle.get("instrumented", {}).get("camera_rows"):
        lines.append("The optional mini-rerun was available, so lifecycle association/export counters are directly measured on selected cameras.")
    else:
        lines.append("No instrumented mini-rerun evidence is present; export-policy conclusions remain limited.")
    lines.extend(["", "## Stage retention", ""])
    for row in bottlenecks[:8]:
        lines.append(
            "- %s: retention=%s, type=%s, confidence=%s"
            % (row.get("bottleneck"), _fmt(row.get("retention")), row.get("unit_comparison_type"), _fmt(row.get("evidence_confidence")))
        )
    lines.extend(["", "## GT object-frame retention", ""])
    for row in gt_rows:
        lines.append("- %s: %s" % (row.get("stage"), _fmt(row.get("gt_object_frame_retention"))))
    lines.extend(["", "## Motion filtering", ""])
    for row in motion_rows:
        lines.append(
            "- %s: rejected %s/%s (%s)"
            % (row.get("variant_name"), row.get("rejected_candidates"), row.get("total_candidates"), _fmt(row.get("rejection_rate")))
        )
    lines.extend(
        [
            "",
            "## Bottleneck ranking",
            "",
            "The ranking weights consistent-unit evidence more strongly than diagnostic-only ratios.",
            "",
            "## Recommended fix plan",
            "",
            "`%s`" % recommendation.get("verdict"),
            "",
            "Evidence: `%s`" % recommendation.get("evidence"),
            "",
            "## Limitations",
            "",
            "- No GT or depth is used on test.",
            "- Artifact-only lifecycle outputs do not contain every internal ByteTrack state transition.",
            "- Global-track to frame-row ratios are not strict retention metrics.",
            "- Pseudo-3D jump and bbox-height analyses are diagnostic proxies, not causal proof.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_rows(inventories: Dict[str, Any], candidate_name: str) -> List[Dict[str, Any]]:
    baseline = inventories.get("v2_current", {}).get("stage_counts", {})
    candidate = inventories.get(candidate_name, {}).get("stage_counts", {})
    rows = []
    for stage in sorted(set(baseline.keys()) | set(candidate.keys())):
        left = int(baseline.get(stage, 0) or 0)
        right = int(candidate.get(stage, 0) or 0)
        rows.append(
            {
                "variant_name": candidate_name,
                "stage_name": stage,
                "v2_current_count": left,
                "candidate_count": right,
                "delta": right - left,
                "retention_vs_v2_current": None if left <= 0 else float(right) / float(left),
            }
        )
    return rows


def _compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verdict": result.get("verdict"),
        "recommended_fix_plan": result.get("recommendation"),
        "top_bottlenecks": result.get("bottlenecks", [])[:10],
        "sample_counts": result.get("samples"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return "%.4f" % float(value)
    except (TypeError, ValueError):
        return str(value)

