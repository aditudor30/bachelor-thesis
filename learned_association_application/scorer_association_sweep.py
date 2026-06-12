"""End-to-end Step 20C learned Person association application."""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.learned_association_application.candidate_pair_feature_adapter import (
    load_and_adapt_candidate_pairs,
)
from deep_oc_sort_3d.learned_association_application.candidate_pair_scorer import (
    SCORE_FIELDS,
    score_candidate_pairs,
)
from deep_oc_sort_3d.learned_association_application.conservative_merge_graph import (
    MERGE_DECISION_FIELDS,
    build_conservative_merge_mapping,
)
from deep_oc_sort_3d.learned_association_application.mlp_scorer_loader import load_selected_mlp
from deep_oc_sort_3d.learned_association_application.scorer_application_config import (
    load_application_config,
    output_root_from_config,
    save_resolved_config,
    variant_names,
)
from deep_oc_sort_3d.learned_association_application.scorer_association_export import export_variant
from deep_oc_sort_3d.learned_association_application.scorer_association_figures import create_figures
from deep_oc_sort_3d.learned_association_application.scorer_association_io import (
    prepare_output_root,
    progress_iter,
    read_csv_rows,
    read_json,
    write_csv_rows,
    write_json,
)
from deep_oc_sort_3d.learned_association_application.scorer_association_metrics import (
    build_comparison_rows,
    collect_baseline_metrics,
    collect_variant_metrics,
)
from deep_oc_sort_3d.learned_association_application.scorer_association_report import write_report
from deep_oc_sort_3d.learned_association_application.scorer_association_selector import select_variant


def apply_person_association_scorer(
    config_path: Path,
    progress: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Adapt candidate pairs, apply Step 20B preprocessing and score with the MLP."""
    config = load_application_config(config_path)
    output_root = prepare_output_root(output_root_from_config(config), overwrite=overwrite)
    save_resolved_config(config, output_root)
    feature_rows, adapter_summary = load_and_adapt_candidate_pairs(config, progress=progress)
    scorer = load_selected_mlp(config)
    known_camera_pairs = set(getattr(scorer.preprocessor, "category_values", {}).get("camera_pair", []))
    for row in feature_rows:
        row["camera_pair_seen_in_train"] = int(str(row.get("camera_pair", "")) in known_camera_pairs)
    write_csv_rows(output_root / "features" / "candidate_pair_features.csv", feature_rows)
    scored_rows, score_summary, missing_report = score_candidate_pairs(
        feature_rows,
        scorer,
        batch_size=int(config.get("candidate_scoring", {}).get("batch_size", 4096)),
    )
    write_csv_rows(output_root / "features" / "candidate_pair_scores.csv", scored_rows, SCORE_FIELDS)
    write_csv_rows(output_root / "features" / "missing_feature_report.csv", missing_report)
    coverage = dict(adapter_summary)
    coverage.update(score_summary)
    write_json(output_root / "features" / "score_summary.json", score_summary)
    write_json(output_root / "diagnostics" / "scorer_coverage_summary.json", coverage)
    write_json(output_root / "diagnostics" / "candidate_filtering_summary.json", _filtering_summary(scored_rows))
    write_json(output_root / "diagnostics" / "score_distribution.json", score_summary)
    write_csv_rows(output_root / "diagnostics" / "threshold_diagnostics.csv", _threshold_diagnostics(scored_rows))
    warnings = _warnings(coverage, missing_report)
    write_json(output_root / "diagnostics" / "warnings.json", {"warnings": warnings})
    status = {
        "status": "ok",
        "output_root": str(output_root),
        "candidate_pairs": len(scored_rows),
        "warnings": warnings,
    }
    write_json(output_root / "diagnostics" / "scoring_status.json", status)
    return status


def run_person_scorer_association_sweep(
    config_path: Path,
    progress: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run scoring, all conservative variants, export, validation and comparison."""
    apply_status = apply_person_association_scorer(config_path, progress=progress, overwrite=overwrite)
    config = load_application_config(config_path)
    output_root = output_root_from_config(config)
    scored_rows = read_csv_rows(output_root / "features" / "candidate_pair_scores.csv")
    statuses = []
    metrics_rows = []
    for variant_name in progress_iter(variant_names(config), progress, "learned association sweep"):
        run_root = output_root / "sweep_runs" / variant_name
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            mapping, accepted, rejected, merge_summary = build_conservative_merge_mapping(
                scored_rows, variant_name, config
            )
            write_csv_rows(run_root / "merge_decisions.csv", accepted, MERGE_DECISION_FIELDS)
            write_csv_rows(run_root / "rejected_merge_decisions.csv", rejected, MERGE_DECISION_FIELDS)
            write_json(run_root / "merge_summary.json", merge_summary)
            export_summary = export_variant(variant_name, mapping, config, run_root, progress)
            metrics = collect_variant_metrics(variant_name, run_root)
            write_json(run_root / "summary.json", metrics)
            metrics_rows.append(metrics)
            status = {
                "variant_name": variant_name,
                "status": "ok",
                "accepted_edges": len(accepted),
                "track1_errors": metrics.get("track1_errors"),
                "export": export_summary,
            }
        except Exception as exc:
            status = {"variant_name": variant_name, "status": "error", "error_message": str(exc)}
        statuses.append(status)
        write_json(output_root / "comparison" / "incremental_sweep_status.json", {"runs": statuses})
        gc.collect()
    comparison = compare_person_scorer_association(config_path, progress=progress)
    result = {"apply_status": apply_status, "runs": statuses, "comparison": comparison}
    write_json(output_root / "comparison" / "step20c_status.json", result)
    return result


def compare_person_scorer_association(
    config_path: Path,
    progress: bool = True,
) -> Dict[str, Any]:
    """Compare completed variants with V2 and Person cleanup/ReID baselines."""
    config = load_application_config(config_path)
    output_root = output_root_from_config(config)
    metrics_rows = []
    for name in progress_iter(variant_names(config), progress, "association comparison"):
        metrics_path = output_root / "sweep_runs" / name / "summary.json"
        metrics = read_json(metrics_path)
        if metrics is None and (output_root / "sweep_runs" / name / "final_export").exists():
            metrics = collect_variant_metrics(name, output_root / "sweep_runs" / name)
            write_json(metrics_path, metrics)
        if metrics is not None:
            metrics_rows.append(metrics)
    baselines = collect_baseline_metrics(config)
    comparison_rows = build_comparison_rows(metrics_rows, baselines)
    write_csv_rows(output_root / "comparison" / "sweep_summary.csv", comparison_rows)
    write_json(output_root / "comparison" / "sweep_summary.json", {"variants": comparison_rows, "baselines": baselines})
    write_csv_rows(output_root / "comparison" / "metric_deltas_vs_v2_current.csv", _delta_rows(comparison_rows, "vs_v2_current_"))
    write_csv_rows(output_root / "comparison" / "metric_deltas_vs_combined_safe_080.csv", _delta_rows(comparison_rows, "vs_combined_safe_080_"))
    baseline_name = str(config.get("selection", {}).get("compare_against", "combined_safe_080"))
    baseline = baselines.get(baseline_name, {})
    if baseline.get("status") == "not_available":
        baseline = baselines.get("v2_current", {})
    selected = select_variant(comparison_rows, baseline, config.get("selection", {}))
    selected["ready_for_step_20d"] = bool(
        selected.get("selected_variant")
        and selected.get("verdict") not in (
            "mlp_association_invalid_fix_required",
            "mlp_association_increases_false_merges",
        )
    )
    write_json(output_root / "comparison" / "selected_variant.json", selected)
    score_summary = read_json(output_root / "features" / "score_summary.json") or {}
    write_report(
        output_root / "comparison" / "PERSON_SCORER_ASSOCIATION_REPORT.md",
        score_summary,
        comparison_rows,
        baselines,
        selected,
    )
    scored_rows = read_csv_rows(output_root / "features" / "candidate_pair_scores.csv")
    figures = []
    if bool(config.get("figures", {}).get("enabled", True)):
        figures = create_figures(scored_rows, comparison_rows, output_root / "figures")
    return {"variants": comparison_rows, "baselines": baselines, "selected": selected, "figures": figures}


def summarize_person_scorer_association(root: Path) -> Dict[str, Any]:
    """Return and print the compact Step 20C outcome."""
    selected = read_json(Path(root) / "comparison" / "selected_variant.json") or {}
    score = read_json(Path(root) / "features" / "score_summary.json") or {}
    rows = read_csv_rows(Path(root) / "comparison" / "sweep_summary.csv")
    summary = {
        "root": str(root),
        "verdict": selected.get("verdict"),
        "selected_variant": selected.get("selected_variant"),
        "candidate_pairs": score.get("candidate_pairs"),
        "pairs_with_reid": score.get("pairs_with_reid"),
        "variants": len(rows),
    }
    return summary


def _filtering_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reasons = {}
    for row in rows:
        reason = str(row.get("rejection_reason", "ok"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {"candidate_pairs": len(rows), "rejection_reasons": reasons}


def _threshold_diagnostics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for threshold in (0.50, 0.60, 0.70, 0.77, 0.80, 0.85, 0.90):
        passing = [row for row in rows if float(row.get("mlp_score") or 0.0) >= threshold]
        output.append(
            {
                "mlp_threshold": threshold,
                "candidate_pairs": len(rows),
                "passing_pairs": len(passing),
                "passing_with_reid_080": len([row for row in passing if float(row.get("reid_similarity") or -1.0) >= 0.80]),
                "passing_with_reid_085": len([row for row in passing if float(row.get("reid_similarity") or -1.0) >= 0.85]),
            }
        )
    return output


def _warnings(coverage: Dict[str, Any], missing_report: List[Dict[str, Any]]) -> List[str]:
    warnings = []
    if int(coverage.get("candidate_pairs") or 0) == 0:
        warnings.append("no_candidate_pairs")
    if int(coverage.get("pairs_with_reid") or 0) < int(coverage.get("candidate_pairs") or 0):
        warnings.append("some_candidate_pairs_missing_reid")
    severe = [row.get("feature") for row in missing_report if float(row.get("missing_ratio") or 0.0) >= 0.95]
    if severe:
        warnings.append("features_missing_at_least_95_percent:%s" % ",".join([str(value) for value in severe]))
    return warnings


def _delta_rows(rows: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        selected = {"run_name": row.get("run_name")}
        for key, value in row.items():
            if key.startswith(prefix):
                selected[key[len(prefix) :]] = value
        output.append(selected)
    return output
