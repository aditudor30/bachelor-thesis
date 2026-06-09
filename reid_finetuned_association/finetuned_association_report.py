"""Report generation for fine-tuned Person ReID association sweep."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_reid.reid_utils import cosine_similarity
from deep_oc_sort_3d.person_reid_association.reid_pair_mining import load_reid_global_embeddings
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import (
    output_root_from_config,
    read_json,
    safe_float,
    write_json,
    write_text,
)


def write_finetuned_association_report(config: Dict[str, Any], comparison: Dict[str, Any], selected: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    """Write Markdown report and pretrained-vs-fine-tuned diagnostics."""
    pretrained_diag = compare_pretrained_and_finetuned_fragments(config, output_root)
    paths = config.get("paths", {})
    extraction = read_json(output_root / "embeddings" / "embedding_extraction_summary.json") or {}
    coverage = read_json(output_root / "diagnostics" / "embedding_coverage_summary.json") or {}
    distribution = read_json(output_root / "diagnostics" / "reid_score_distribution.json") or {}
    lines = [
        "# Person Fine-Tuned ReID Association Report",
        "",
        "## Context",
        "",
        "- Step 18A built the SmartSpaces Person ReID dataset.",
        "- Step 18B fine-tuned OSNet and produced `best_retrieval_top1.pth`.",
        "- Step 18C re-extracts fragment embeddings and runs association sweeps in a separate output root.",
        "",
        "checkpoint: `%s`" % paths.get("finetuned_checkpoint", ""),
        "output_root: `%s`" % output_root,
        "",
        "## Embedding Coverage",
        "",
        "- crop embeddings: `%s`" % extraction.get("crop_embeddings"),
        "- fragment embeddings: `%s`" % ((extraction.get("fragment_summary") or {}).get("num_valid_fragments")),
        "- sampled crop rate vs Person records: `%s`" % coverage.get("sampled_crop_rate_vs_person_records"),
        "",
        "## Pretrained vs Fine-Tuned",
        "",
        "- available: `%s`" % pretrained_diag.get("pretrained_comparison_available"),
        "- overlap fragments: `%s`" % pretrained_diag.get("overlap_fragments"),
        "- mean cosine between pretrained and fine-tuned fragment embeddings: `%s`" % pretrained_diag.get("same_fragment_cosine_mean"),
        "",
        "## ReID Score Distribution",
        "",
        "- pairs with ReID: `%s`" % distribution.get("num_pairs_with_reid"),
        "- median: `%s`" % distribution.get("median"),
        "- p90: `%s`" % distribution.get("p90"),
        "- p95: `%s`" % distribution.get("p95"),
        "",
        "## Sweep Verdict",
        "",
        "- verdict: `%s`" % selected.get("verdict"),
        "- best_run: `%s`" % selected.get("best_run"),
        "",
        "## Sweep Table",
        "",
        "| run | status | track1 errors | person fragmentation delta | false merge delta | purity delta | non-person delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.get("runs", []):
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.get("run_name"),
                row.get("run_status"),
                row.get("track1_validation_errors"),
                row.get("vs_v2_person_fragmentation_approx_delta"),
                row.get("vs_v2_false_merge_rate_delta"),
                row.get("vs_v2_global_purity_mean_delta"),
                row.get("vs_v2_non_person_rows_delta"),
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            recommendation_text(selected.get("verdict")),
        ]
    )
    report_path = output_root / "comparison" / "PERSON_REID_FINETUNED_ASSOCIATION_REPORT.md"
    write_text(lines, report_path)
    return {"report_path": str(report_path), "pretrained_diagnostics": pretrained_diag}


def compare_pretrained_and_finetuned_fragments(config: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    """Compare old pretrained fragment embeddings with fine-tuned fragment embeddings if present."""
    paths = config.get("paths", {})
    pretrained_root = Path(str(paths.get("pretrained_reid_root", "output/reid_person/baseline_v2_pseudo3d_fullcam"))) / "embeddings_global_fragment"
    finetuned_root = Path(output_root) / "embeddings" / "fragment_embeddings"
    if not pretrained_root.exists():
        summary = {
            "pretrained_comparison_available": False,
            "reason": "pretrained_embedding_root_missing",
            "pretrained_root": str(pretrained_root),
        }
        write_json(summary, Path(output_root) / "diagnostics" / "finetuned_vs_pretrained_fragment_similarity.json")
        return summary
    try:
        pretrained = load_reid_global_embeddings(pretrained_root, person_class_id=0)
        finetuned = load_reid_global_embeddings(finetuned_root, person_class_id=0)
    except Exception as exc:
        summary = {"pretrained_comparison_available": False, "reason": str(exc), "pretrained_root": str(pretrained_root)}
        write_json(summary, Path(output_root) / "diagnostics" / "finetuned_vs_pretrained_fragment_similarity.json")
        return summary
    keys = sorted(set(pretrained.keys()).intersection(set(finetuned.keys())), key=lambda item: str(item))
    similarities = [cosine_similarity(pretrained[key].embedding, finetuned[key].embedding) for key in keys]
    summary = {
        "pretrained_comparison_available": True,
        "pretrained_root": str(pretrained_root),
        "finetuned_root": str(finetuned_root),
        "pretrained_fragments": len(pretrained),
        "finetuned_fragments": len(finetuned),
        "overlap_fragments": len(keys),
        "same_fragment_cosine_mean": _mean(similarities),
        "same_fragment_cosine_median": _percentile(similarities, 50),
        "same_fragment_cosine_p05": _percentile(similarities, 5),
        "same_fragment_cosine_p95": _percentile(similarities, 95),
    }
    write_json(summary, Path(output_root) / "diagnostics" / "finetuned_vs_pretrained_fragment_similarity.json")
    return summary


def recommendation_text(verdict: Any) -> str:
    """Return recommendation text for a selector verdict."""
    if verdict == "finetuned_reid_association_improves_v2":
        return "Use the selected fine-tuned ReID variant as a candidate upgrade after visual sanity checks."
    if verdict == "finetuned_reid_valid_small_gain":
        return "Keep the selected fine-tuned ReID variant as a safe small-gain candidate, not as a forced replacement."
    if verdict == "finetuned_reid_increases_false_merges":
        return "Do not promote this ReID association variant; use it only as diagnostic until thresholds/domain tuning improve."
    if verdict == "finetuned_reid_invalid_fix_required":
        return "Stop here and fix validation/export errors before considering Step 18D."
    return "Keep V2 current and V2 export_compact as baselines; use fine-tuned ReID only as diagnostic unless a later sweep shows clear gain."


def _mean(values: List[Any]) -> Optional[float]:
    numeric = [safe_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return float(sum(numeric)) / float(len(numeric))


def _percentile(values: List[Any], percentile: float) -> Optional[float]:
    numeric = sorted([safe_float(value, None) for value in values if safe_float(value, None) is not None])
    if not numeric:
        return None
    if len(numeric) == 1:
        return float(numeric[0])
    index = int(round(float(percentile) / 100.0 * float(len(numeric) - 1)))
    index = max(0, min(len(numeric) - 1, index))
    return float(numeric[index])
