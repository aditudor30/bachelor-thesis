"""CLI for printing a compact ReID-guided Person association summary."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_reid_association.reid_association_io import read_csv_rows, read_json, safe_float


def main() -> None:
    """Print compact summary."""
    parser = argparse.ArgumentParser(description="Summarize ReID-guided Person association outputs.")
    parser.add_argument("--root", type=Path, default=Path("output/person_reid_association/baseline_v2_pseudo3d_fullcam"))
    parser.add_argument("--top-reasons", type=int, default=5)
    args = parser.parse_args()
    summary = read_json(args.root / "comparison" / "reid_person_association_summary.json") or {}
    recommendation = summary.get("best_reid_person_association_recommendation", {})
    print("verdict: %s" % recommendation.get("verdict"))
    print("best_run: %s" % recommendation.get("best_run"))
    for run in summary.get("runs", []):
        run_name = str(run.get("run_name"))
        run_root = args.root / "runs" / run_name
        candidate_summary = read_json(run_root / "candidate_pairs" / "reid_person_candidate_pairs_summary.json") or {}
        score_summary = _score_summary(run_root)
        merge_summary = read_json(run_root / "diagnostics" / "reid_merge_summary.json") or {}
        print("")
        print(
            "%s status=%s merges=%s person_frag=%s purity=%s false_merge=%s non_person_delta=%s reid_passing=%s"
            % (
                run_name,
                run.get("run_status", "ok"),
                run.get("merges_applied"),
                run.get("person_fragmentation_approx"),
                run.get("global_purity_mean"),
                run.get("false_merge_rate"),
                run.get("vs_v2_non_person_rows_delta"),
                run.get("pairs_passing_reid_threshold"),
            )
        )
        print(
            "  pairs: candidates=%s with_reid=%s missing_reid=%s threshold=%s"
            % (
                _first_not_none(candidate_summary.get("candidate_rows"), candidate_summary.get("stored_rows")),
                _first_not_none(candidate_summary.get("pairs_with_both_reid"), score_summary.get("pairs_with_reid")),
                _first_not_none(candidate_summary.get("pairs_missing_reid"), score_summary.get("pairs_missing_reid")),
                _first_not_none(score_summary.get("reid_similarity_threshold"), _threshold_from_run_name(run_name)),
            )
        )
        print(
            "  similarity: min=%s med=%s p90=%s p95=%s p99=%s max=%s mean=%s"
            % (
                _fmt(score_summary.get("reid_similarity_min")),
                _fmt(score_summary.get("reid_similarity_median")),
                _fmt(score_summary.get("reid_similarity_p90")),
                _fmt(score_summary.get("reid_similarity_p95")),
                _fmt(score_summary.get("reid_similarity_p99")),
                _fmt(score_summary.get("reid_similarity_max")),
                _fmt(score_summary.get("reid_similarity_mean")),
            )
        )
        print(
            "  threshold counts: >=0.70:%s >=0.75:%s >=0.80:%s >=0.82:%s >=0.85:%s"
            % (
                score_summary.get("pairs_passing_reid_070"),
                score_summary.get("pairs_passing_reid_075"),
                score_summary.get("pairs_passing_reid_080"),
                score_summary.get("pairs_passing_reid_082"),
                score_summary.get("pairs_passing_reid_085"),
            )
        )
        print(
            "  selected_gt_diag: same=%s different=%s unknown=%s selected_sim_mean=%s"
            % (
                merge_summary.get("selected_same_gt_diagnostic"),
                merge_summary.get("selected_different_gt_diagnostic"),
                merge_summary.get("selected_unknown_gt_diagnostic"),
                _fmt(merge_summary.get("selected_reid_similarity_mean")),
            )
        )
        print("  merge reject reasons: %s" % _top_reason_text(merge_summary.get("reject_reasons", {}), args.top_reasons))
        print("  quick_diagnosis: %s" % _diagnosis(run_name, candidate_summary, score_summary, merge_summary))

def _score_summary(run_root: Path) -> Dict[str, Any]:
    stored = read_json(run_root / "scores" / "reid_person_pair_scores_summary.json") or {}
    rows, _fields = read_csv_rows(run_root / "scores" / "reid_person_pair_scores.csv")
    if not rows:
        return stored
    similarities = [row.get("reid_similarity") for row in rows if row.get("reid_score_status") == "ok" or row.get("reid_status") == "ok"]
    dynamic = {
        "pairs_with_reid": len(similarities),
        "pairs_missing_reid": len(rows) - len(similarities),
        "reid_similarity_min": _percentile(similarities, 0),
        "reid_similarity_mean": _mean(similarities),
        "reid_similarity_median": _percentile(similarities, 50),
        "reid_similarity_p90": _percentile(similarities, 90),
        "reid_similarity_p95": _percentile(similarities, 95),
        "reid_similarity_p99": _percentile(similarities, 99),
        "reid_similarity_max": _percentile(similarities, 100),
        "pairs_passing_reid_070": _count_at_least(similarities, 0.70),
        "pairs_passing_reid_075": _count_at_least(similarities, 0.75),
        "pairs_passing_reid_080": _count_at_least(similarities, 0.80),
        "pairs_passing_reid_082": _count_at_least(similarities, 0.82),
        "pairs_passing_reid_085": _count_at_least(similarities, 0.85),
    }
    output = dict(stored)
    output.update(dynamic)
    return output


def _diagnosis(run_name: str, candidate_summary: Dict[str, Any], score_summary: Dict[str, Any], merge_summary: Dict[str, Any]) -> str:
    with_reid = _to_float(_first_not_none(candidate_summary.get("pairs_with_both_reid"), score_summary.get("pairs_with_reid")))
    passing = _to_float(score_summary.get("pairs_passing_reid_threshold"))
    max_similarity = _to_float(score_summary.get("reid_similarity_max"))
    threshold = _to_float(_first_not_none(score_summary.get("reid_similarity_threshold"), _threshold_from_run_name(run_name)))
    selected = _to_float(merge_summary.get("selected_edges_before_conflict_filter"))
    mapping = _to_float(merge_summary.get("mapping_size"))
    if with_reid is None or with_reid <= 0:
        return "no candidate pairs have both ReID embeddings; check embedding keys/root."
    if threshold is not None and max_similarity is not None and max_similarity < threshold:
        return "all ReID similarities are below the configured threshold."
    if passing is not None and passing <= 0:
        return "no pair passes the configured ReID threshold."
    if selected is not None and selected > 0 and (mapping is None or mapping <= 0):
        return "pairs passed ReID, but duplicate/conflict filtering removed the merge mapping."
    if mapping is not None and mapping > 0:
        return "ReID produced Person merge mappings; inspect purity/false-merge trade-off."
    return "pairs exist, but another merge gate rejected them; inspect merge reject reasons."


def _top_reason_text(reasons: Any, limit: int) -> str:
    if not isinstance(reasons, dict) or not reasons:
        return "{}"
    items = sorted(reasons.items(), key=lambda item: int(float(item[1])), reverse=True)
    return ", ".join(["%s=%s" % (key, value) for key, value in items[:limit]])


def _threshold_from_run_name(run_name: str) -> Optional[float]:
    if run_name == "reid_strict":
        return 0.85
    if run_name == "reid_medium" or run_name == "reid_with_export_compact":
        return 0.82
    if run_name == "reid_balanced":
        return 0.80
    return None


def _first_not_none(left: Any, right: Any) -> Any:
    return left if left not in (None, "") else right


def _fmt(value: Any) -> str:
    number = safe_float(value, None)
    if number is None:
        return "None"
    return "%.4f" % float(number)


def _to_float(value: Any) -> Optional[float]:
    return safe_float(value, None)


def _count_at_least(values: List[Any], threshold: float) -> int:
    numeric = [safe_float(value, None) for value in values]
    return len([value for value in numeric if value is not None and float(value) >= float(threshold)])


def _mean(values: List[Any]) -> Optional[float]:
    numeric = [safe_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return float(sum(numeric)) / float(len(numeric))


def _percentile(values: List[Any], p: float) -> Optional[float]:
    parsed = [safe_float(value, None) for value in values]
    numeric = sorted([value for value in parsed if value is not None])
    if not numeric:
        return None
    if len(numeric) == 1:
        return float(numeric[0])
    index = int(round((float(p) / 100.0) * float(len(numeric) - 1)))
    index = max(0, min(len(numeric) - 1, index))
    return float(numeric[index])


if __name__ == "__main__":
    main()
