"""Metrics for ReID-guided Person association runs."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.person_association.person_association_metrics import (
    collect_person_association_metrics,
    compute_association_deltas,
)
from deep_oc_sort_3d.person_reid_association.reid_association_io import read_json, write_json


def collect_reid_association_metrics(
    run_name: str,
    final_export_root: Path,
    track1_root: Path,
    diagnostics_root: Path,
) -> Dict[str, Any]:
    """Collect ReID association metrics."""
    metrics = collect_person_association_metrics(
        run_name,
        final_export_root,
        track1_root,
        diagnostics_root / "reid_merge_summary.json",
    )
    merge = read_json(diagnostics_root / "reid_merge_summary.json") or {}
    candidates = read_json(diagnostics_root.parent / "candidate_pairs" / "reid_person_candidate_pairs_summary.json") or {}
    scores = read_json(diagnostics_root.parent / "scores" / "reid_person_pair_scores_summary.json") or {}
    metrics.update(
        {
            "candidate_pairs_generated": candidates.get("candidate_rows"),
            "pairs_with_both_reid": candidates.get("pairs_with_both_reid"),
            "pairs_missing_reid": candidates.get("pairs_missing_reid"),
            "pairs_passing_reid_threshold": scores.get("pairs_passing_reid_threshold"),
            "merges_applied": merge.get("mapping_size"),
            "merged_components": merge.get("merged_components"),
            "selected_same_gt_diagnostic": merge.get("selected_same_gt_diagnostic"),
            "selected_different_gt_diagnostic": merge.get("selected_different_gt_diagnostic"),
            "selected_unknown_gt_diagnostic": merge.get("selected_unknown_gt_diagnostic"),
            "selected_reid_similarity_mean": merge.get("selected_reid_similarity_mean"),
        }
    )
    return metrics


def write_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Write metrics JSON."""
    write_json(metrics, path)


def compute_reid_association_deltas(run: Dict[str, Any], baseline: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Compute comparable deltas."""
    return compute_association_deltas(run, baseline, prefix)

