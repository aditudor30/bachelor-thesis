"""Markdown report generation for Step 16A Person ReID."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.person_reid.reid_utils import read_json


def write_person_reid_report(output_root: Path) -> Dict[str, Any]:
    """Write the Step 16A report and return the combined summary."""
    crop = read_json(output_root / "summaries" / "crop_extraction_summary.json") or {}
    embedding = read_json(output_root / "summaries" / "crop_embedding_summary.json") or {}
    aggregation = read_json(output_root / "summaries" / "aggregation_summary.json") or {}
    diagnostics = read_json(output_root / "summaries" / "reid_diagnostics_summary.json") or {}
    summary = {
        "crop_extraction": crop,
        "crop_embedding": embedding,
        "aggregation": aggregation,
        "diagnostics": diagnostics,
        "verdict": diagnostics.get("verdict", _fallback_verdict(embedding)),
    }
    report_path = output_root / "report" / "PERSON_REID_DIAGNOSTIC_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: Dict[str, Any]) -> str:
    crop = summary.get("crop_extraction", {})
    embedding = summary.get("crop_embedding", {})
    aggregation = summary.get("aggregation", {})
    diagnostics = summary.get("diagnostics", {})
    sim = diagnostics.get("similarity", {}) if isinstance(diagnostics.get("similarity", {}), dict) else {}
    retrieval = diagnostics.get("retrieval", {}) if isinstance(diagnostics.get("retrieval", {}), dict) else {}
    lines = []
    lines.append("# Person ReID Diagnostic Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- verdict: `%s`" % summary.get("verdict"))
    lines.append("")
    lines.append("## Backend")
    lines.append("")
    lines.append("- backend: `%s`" % embedding.get("backend"))
    lines.append("- status: `%s`" % embedding.get("status"))
    lines.append("- weights_loaded: `%s`" % embedding.get("weights_loaded"))
    lines.append("- embedding_dim: `%s`" % embedding.get("embedding_dim"))
    if embedding.get("message"):
        lines.append("- message: %s" % embedding.get("message"))
    lines.append("")
    lines.append("## Crop Extraction")
    lines.append("")
    lines.append("- total Person records: `%s`" % crop.get("total_person_records"))
    lines.append("- crop records: `%s`" % crop.get("crop_records"))
    lines.append("- crop success rate: `%s`" % crop.get("crop_success_rate"))
    lines.append("- invalid bbox count: `%s`" % crop.get("invalid_bbox_count"))
    lines.append("- missing frame count: `%s`" % crop.get("missing_frame_count"))
    lines.append("")
    lines.append("## Embeddings")
    lines.append("")
    lines.append("- embeddings generated: `%s`" % embedding.get("embeddings_generated"))
    lines.append("- norm mean/std: `%s / %s`" % (embedding.get("embedding_norm_mean"), embedding.get("embedding_norm_std")))
    lines.append("- local track embeddings: `%s`" % aggregation.get("local_track_embeddings"))
    lines.append("- global fragment embeddings: `%s`" % aggregation.get("global_fragment_embeddings"))
    lines.append("")
    lines.append("## Similarity Diagnostics")
    lines.append("")
    lines.append("- same-GT mean/median: `%s / %s`" % (sim.get("same_gt_similarity_mean"), sim.get("same_gt_similarity_median")))
    lines.append("- different-GT mean/median: `%s / %s`" % (sim.get("different_gt_similarity_mean"), sim.get("different_gt_similarity_median")))
    lines.append("- separation margin: `%s`" % sim.get("separation_margin"))
    lines.append("- sampled pairs: `%s`" % sim.get("num_pairs"))
    lines.append("")
    lines.append("## Retrieval")
    lines.append("")
    lines.append("- status: `%s`" % retrieval.get("status"))
    lines.append("- top1: `%s`" % retrieval.get("top1_accuracy"))
    lines.append("- top5: `%s`" % retrieval.get("top5_accuracy"))
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(_recommendation(summary.get("verdict")))
    return "\n".join(lines) + "\n"


def _fallback_verdict(embedding: Dict[str, Any]) -> str:
    if embedding.get("status") == "backend_unavailable":
        return "reid_backend_unavailable"
    return "use_v2_current_without_reid"


def _recommendation(verdict: Any) -> str:
    if verdict == "reid_ready_for_person_association":
        return "Continue to Step 16B with ReID-guided Person association and conservative thresholding."
    if verdict == "reid_promising_but_needs_threshold_tuning":
        return "Run a threshold/weight sweep before allowing ReID to create Person merges."
    if verdict == "reid_backend_unavailable":
        return "Provide local pretrained ReID weights or switch to a locally available backend."
    if verdict == "reid_not_discriminative_enough":
        return "Try a stronger pretrained ReID backend before integrating appearance into association."
    return "Keep V2 current as the safe baseline."

