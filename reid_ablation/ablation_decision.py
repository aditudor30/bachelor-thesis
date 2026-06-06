"""Final decision logic for ReID ablation cleanup."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.reid_ablation.ablation_comparison import build_reid_ablation_comparison_from_config
from deep_oc_sort_3d.reid_ablation.ablation_io import load_yaml, write_json


def run_reid_ablation_decision(config_path: Path, progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Run the complete Step 16C ablation decision workflow."""
    _unused_overwrite = overwrite
    config = load_yaml(config_path)
    output_root = Path(str(config.get("reid_ablation_decision", {}).get("output_root", "output/reid_ablation_decision/baseline_v2_pseudo3d_fullcam")))
    comparison = build_reid_ablation_comparison_from_config(config_path, progress=progress)
    decision = make_final_decision(comparison)
    write_json(decision, output_root / "comparison" / "final_variant_decision.json")
    return {"comparison": comparison, "decision": decision, "output_root": str(output_root)}


def make_final_decision(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Create final keep/drop decisions and high-level verdicts."""
    rows = comparison.get("variants", [])
    v1 = _find(rows, "v1")
    v2 = _find(rows, "v2_current")
    export_compact = _find(rows, "export_compact")
    reid_only = [row for row in rows if row.get("source_type") == "reid_only"]
    reid_plus = [row for row in rows if row.get("source_type") == "reid_plus_compact"]
    verdicts = []
    keep = {}
    if v1 and v1.get("track1_valid") is True:
        verdicts.append("keep_v1_for_submission")
        keep["submission_safe_baseline"] = v1.get("variant_name")
    if v2 and v2.get("track1_valid") is True:
        verdicts.append("keep_v2_current_as_3d_mvp")
        keep["provenance_3d_mvp"] = v2.get("variant_name")
    if export_compact and export_compact.get("is_safe"):
        verdicts.append("keep_v2_export_compact_as_safe_variant")
        keep["safe_compact_variant"] = export_compact.get("variant_name")
    reid_ready = _best_reid_upgrade(reid_only)
    if reid_ready is not None:
        verdicts.append("reid_ready_for_next_association_round")
        keep["reid_ablation_candidate"] = reid_ready.get("variant_name")
    else:
        if _has_reid_activity(reid_only + reid_plus):
            verdicts.append("reid_infrastructure_valid_but_no_gain")
        verdicts.append("reid_needs_domain_tuning")
        keep["reid_ablation_candidate"] = "diagnostic_only"
    reid_plus_best = _best_safe(reid_plus)
    if reid_plus_best is not None and reid_plus_best.get("improvement_source") == "export_compact_only":
        keep["reid_plus_compact_note"] = "gain_attributed_to_export_compact_not_reid"
    return {
        "verdicts": verdicts,
        "kept_variants": keep,
        "reid_only_real_upgrade": reid_ready is not None,
        "best_reid_only": None if reid_ready is None else reid_ready.get("variant_name"),
        "best_reid_plus_compact": None if reid_plus_best is None else reid_plus_best.get("variant_name"),
        "final_recommendation": _final_recommendation(verdicts, keep),
    }


def _final_recommendation(verdicts: List[str], keep: Dict[str, Any]) -> str:
    if "reid_ready_for_next_association_round" in verdicts:
        return "Keep V2 as MVP and keep the best ReID run as an ablation candidate, but verify manually before submission."
    return (
        "Keep V1 as submission-safe baseline, keep V2 fullcam as the 3D provenance-backed MVP, "
        "keep export_compact as a safe compact variant, and treat ReID as infrastructure that needs domain tuning."
    )


def _find(rows: List[Dict[str, Any]], source_type: str) -> Dict[str, Any]:
    for row in rows:
        if row.get("source_type") == source_type:
            return row
    return {}


def _best_reid_upgrade(rows: List[Dict[str, Any]]) -> Any:
    candidates = [row for row in rows if row.get("real_upgrade") and row.get("is_safe")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: _number(row.get("person_fragmentation_delta_vs_v2")) or 0.0)[0]


def _best_safe(rows: List[Dict[str, Any]]) -> Any:
    candidates = [row for row in rows if row.get("is_safe")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: _number(row.get("track1_rows_delta_vs_v2")) or 0.0)[0]


def _has_reid_activity(rows: List[Dict[str, Any]]) -> bool:
    for row in rows:
        try:
            if float(row.get("num_reid_merges") or 0) > 0:
                return True
            if float(row.get("pairs_with_reid") or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _number(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
