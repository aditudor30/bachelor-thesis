"""Registry of final project variants frozen in Step 19."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import find_first_track1


def final_variant_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the frozen final variant registry."""
    paths = config.get("paths", {})
    reid_root = Path(str(paths.get("reid_finetuned_association_root", "")))
    compact_text = str(paths.get("v2_export_compact_root", ""))
    compact_root = Path(compact_text) if compact_text else Path("__missing_v2_export_compact_root__")
    threshold_root = reid_root / "sweep_runs" / "threshold_080"
    combined_root = reid_root / "sweep_runs" / "combined_safe_080"
    specs = [
        {
            "variant_name": "v1_geometry_only",
            "display_name": "V1 geometry-only",
            "role": "submission-safe baseline",
            "status": "keep",
            "source_paths": [
                paths.get("v1_global_root", ""),
                paths.get("v1_final_export_root", ""),
                paths.get("v1_track1_root", ""),
                paths.get("v1_package_root", ""),
            ],
            "final_export_root": paths.get("v1_final_export_root", ""),
            "track1_root": paths.get("v1_track1_root", ""),
            "global_root": paths.get("v1_global_root", ""),
            "reid_used": False,
            "reid_model": "",
            "reid_training": "",
            "visual_review_status": "",
            "final_recommendation": "keep as submission-safe baseline",
            "notes": "Compact validated geometry-only fallback; no full 3D provenance.",
        },
        {
            "variant_name": "v2_pseudo3d_fullcam_current",
            "display_name": "V2 pseudo3D fullcam",
            "role": "3D provenance MVP",
            "status": "keep",
            "source_paths": [
                paths.get("v2_pipeline_root", ""),
                paths.get("v2_global_root", ""),
                paths.get("v2_final_export_root", ""),
                paths.get("v2_track1_root", ""),
                paths.get("v2_comparison_root", ""),
            ],
            "final_export_root": paths.get("v2_final_export_root", ""),
            "track1_root": paths.get("v2_track1_root", ""),
            "global_root": paths.get("v2_global_root", ""),
            "pseudo3d_used_rate_default": 0.9807563276013348,
            "fallback_original_used_rate_default": 0.01924367239866525,
            "reid_used": False,
            "reid_model": "",
            "reid_training": "",
            "visual_review_status": "",
            "final_recommendation": "keep as main 3D MVP",
            "notes": "Main pseudo3D full-camera MVP with provenance; Person fragmentation remains the main limitation.",
        },
        {
            "variant_name": "v2_export_compact",
            "display_name": "V2 export_compact",
            "role": "safe compact variant",
            "status": "keep",
            "source_paths": [str(compact_root)],
            "final_export_root": str(_find_run_root(compact_root) / "final_export"),
            "track1_root": str(_find_run_root(compact_root) / "track1_submission"),
            "global_root": paths.get("v2_global_root", ""),
            "pseudo3d_used_rate_default": 0.9807563276013348,
            "fallback_original_used_rate_default": 0.01924367239866525,
            "reid_used": False,
            "reid_model": "",
            "reid_training": "",
            "visual_review_status": "",
            "final_recommendation": "keep as safe compact variant",
            "notes": "Conservative compact export; intended to reduce row inflation without changing non-Person classes.",
        },
        {
            "variant_name": "osnet_pretrained_diagnostic",
            "display_name": "OSNet pretrained diagnostic",
            "role": "ReID-only diagnostic",
            "status": "diagnostic",
            "source_paths": [
                paths.get("reid_pretrained_root", ""),
                paths.get("reid_pretrained_association_root", ""),
                paths.get("reid_ablation_root", ""),
            ],
            "final_export_root": "",
            "track1_root": "",
            "global_root": paths.get("v2_global_root", ""),
            "reid_used": True,
            "reid_model": "OSNet pretrained",
            "reid_training": "off-the-shelf",
            "visual_review_status": "diagnostic only",
            "final_recommendation": "keep as ReID diagnostic",
            "notes": "Pretrained OSNet infrastructure validated but no final tracking gain before SmartSpaces fine-tuning.",
        },
        {
            "variant_name": "osnet_finetuned_threshold_080",
            "display_name": "OSNet fine-tuned threshold_080",
            "role": "ReID-only diagnostic",
            "status": "diagnostic",
            "source_paths": [str(threshold_root)],
            "final_export_root": str(threshold_root / "final_export"),
            "track1_root": str(threshold_root / "track1_submission"),
            "global_root": paths.get("v2_global_root", ""),
            "reid_used": True,
            "reid_model": "OSNet fine-tuned SmartSpaces Person",
            "reid_training": "fine-tuned on SmartSpaces Person crops",
            "visual_review_status": "diagnostic",
            "final_recommendation": "keep as ReID-only diagnostic",
            "notes": "Pure ReID threshold 0.80 gives a controlled but small fragmentation gain.",
        },
        {
            "variant_name": "osnet_finetuned_combined_safe_080",
            "display_name": "OSNet fine-tuned combined_safe_080",
            "role": "experimental fine-tuned ReID variant",
            "status": "experimental",
            "source_paths": [str(combined_root), paths.get("reid_visual_decision_root", "")],
            "final_export_root": str(combined_root / "final_export"),
            "track1_root": str(combined_root / "track1_submission"),
            "global_root": paths.get("v2_global_root", ""),
            "reid_used": True,
            "reid_model": "OSNet fine-tuned SmartSpaces Person",
            "reid_training": "fine-tuned on SmartSpaces Person crops",
            "visual_review_status": "manual review promising; auto labels conservative",
            "final_recommendation": "keep as experimental fine-tuned ReID final",
            "notes": "ReID-enhanced extension; visually promising but not a full replacement for the submission-safe baseline.",
        },
    ]
    for spec in specs:
        spec["track1_path"] = _track1_path(spec)
    return specs


def _find_run_root(root: Path) -> Path:
    """Return root or an export_compact experiment subroot if present."""
    for child in [
        root / "experiments" / "export_compact",
        root / "runs" / "export_compact",
        root / "export_compact",
        root,
    ]:
        if child.exists():
            return child
    return root


def _track1_path(spec: Dict[str, Any]) -> str:
    root = Path(str(spec.get("track1_root", "")))
    found = find_first_track1(root)
    if found is not None:
        return str(found)
    if str(spec.get("track1_root", "")):
        return str(root / "track1.txt")
    return ""
