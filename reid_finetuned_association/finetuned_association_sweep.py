"""Run fine-tuned ReID-guided Person association sweep variants."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.person_reid_association.reid_merge_runner import run_reid_person_association_experiment
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import (
    load_finetuned_association_config,
    output_root_from_config,
    prepare_output_root,
    progress_iter,
    save_resolved_config,
    threshold_to_name,
    write_json,
    write_yaml,
)
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_metrics import compare_sweep_to_v2
from deep_oc_sort_3d.reid_finetuned_association.finetuned_embedding_extractor import extract_finetuned_person_embeddings_from_config
from deep_oc_sort_3d.reid_finetuned_association.finetuned_pair_scorer import score_finetuned_candidate_pairs_from_config
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_report import write_finetuned_association_report
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_selector import select_finetuned_reid_variant
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_figures import create_finetuned_association_figures


def run_finetuned_person_reid_association_sweep(
    config_path: Path,
    progress: bool = True,
    overwrite: bool = False,
    run_embeddings: bool = True,
    run_scoring: bool = True,
    run_sweep: bool = True,
) -> Dict[str, Any]:
    """Run the complete Step 18C pipeline into a separate output root."""
    config = load_finetuned_association_config(Path(config_path))
    output_root = prepare_output_root(config, overwrite=overwrite)
    save_resolved_config(config, Path(config_path), output_root)
    status: Dict[str, Any] = {"status": "ok", "output_root": str(output_root), "config_path": str(config_path)}
    if run_embeddings:
        status["embedding_extraction"] = extract_finetuned_person_embeddings_from_config(config, show_progress=progress, overwrite=overwrite)
    if run_scoring:
        status["candidate_scoring"] = score_finetuned_candidate_pairs_from_config(config, show_progress=progress)
    if run_sweep:
        statuses = run_sweep_variants(config, progress=progress, overwrite=overwrite)
        status["sweep_statuses"] = statuses
    comparison = compare_sweep_to_v2(config, progress=progress)
    selected = select_finetuned_reid_variant(comparison, config.get("selection", {}))
    write_json(selected, output_root / "comparison" / "selected_variant.json")
    write_finetuned_association_report(config, comparison, selected, output_root)
    create_finetuned_association_figures(output_root)
    status["comparison"] = comparison
    status["selected_variant"] = selected
    write_json(status, output_root / "comparison" / "step18c_status.json")
    return status


def run_sweep_variants(config: Dict[str, Any], progress: bool = True, overwrite: bool = False) -> List[Dict[str, Any]]:
    """Run each configured fine-tuned ReID association variant."""
    output_root = output_root_from_config(config)
    run_specs = build_run_specs(config)
    statuses = []
    for spec in progress_iter(run_specs, progress, "fine-tuned ReID association sweep", "run"):
        name = str(spec.get("name", ""))
        run_root = output_root / "sweep_runs" / name
        config_path = output_root / "configs" / ("%s.yaml" % name)
        write_yaml({"person_reid_association_run": spec["config"]}, config_path)
        status = run_reid_person_association_experiment(
            name,
            config_path,
            run_root,
            overwrite=overwrite,
            progress=progress,
        )
        statuses.append(status)
        write_json({"runs": statuses}, output_root / "comparison" / "incremental_sweep_status.json")
    return statuses


def build_run_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build existing-runner configs for all Step 18C sweep variants."""
    sweep_cfg = config.get("sweep", {})
    thresholds = [float(value) for value in sweep_cfg.get("thresholds", [0.65, 0.70, 0.75, 0.80, 0.85])]
    variants = [str(value) for value in sweep_cfg.get("variants", [])]
    specs: List[Dict[str, Any]] = []
    for threshold in thresholds:
        name = threshold_to_name(threshold)
        if variants and name not in variants:
            continue
        specs.append({"name": name, "config": build_single_run_config(config, threshold, use_export_compact=False)})
    for name, threshold in [("combined_safe_075", 0.75), ("combined_safe_080", 0.80)]:
        if variants and name not in variants:
            continue
        specs.append({"name": name, "config": build_single_run_config(config, threshold, use_export_compact=True)})
    return specs


def build_single_run_config(config: Dict[str, Any], threshold: float, use_export_compact: bool) -> Dict[str, Any]:
    """Create one config for the existing ReID association runner."""
    paths = config.get("paths", {})
    candidate_cfg = config.get("candidate_scoring", {})
    selection_cfg = config.get("selection", {})
    output_root = output_root_from_config(config)
    run_config = {
        "paths": {
            "v2_final_export_root": str(paths.get("v2_final_export_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam")),
            "schema_yaml": str(paths.get("schema_yaml", "deep_oc_sort_3d/configs/track1_schema_confirmed.yaml")),
            "reid_global_embeddings_root": str(output_root / "embeddings" / "fragment_embeddings"),
        },
        "class_id": int(candidate_cfg.get("class_id", 0)),
        "apply_to_subsets": candidate_cfg.get("subsets", ["internal_holdout", "official_val", "test"]),
        "apply_to_scenes": candidate_cfg.get("scenes"),
        "pair_mining": {
            "class_id": int(candidate_cfg.get("class_id", 0)),
            "max_temporal_gap": int(candidate_cfg.get("max_temporal_gap", 300)),
            "max_entry_exit_distance": float(candidate_cfg.get("max_spatial_distance", 12.0)),
            "max_expected_position_error": float(candidate_cfg.get("max_expected_position_error", candidate_cfg.get("max_spatial_distance", 12.0))),
            "max_velocity_angle": float(candidate_cfg.get("max_velocity_angle", 140.0)),
            "forbid_same_camera_temporal_overlap": bool(candidate_cfg.get("forbid_same_camera_temporal_overlap", True)),
            "include_rejected": False,
            "store_rejected_pairs": False,
        },
        "scoring": {
            "temporal_gap_weight": float(candidate_cfg.get("temporal_gap_weight", 0.15)),
            "distance_weight": float(candidate_cfg.get("distance_weight", 0.25)),
            "velocity_weight": float(candidate_cfg.get("velocity_weight", 0.15)),
            "confidence_weight": float(candidate_cfg.get("confidence_weight", 0.10)),
            "expected_position_weight": float(candidate_cfg.get("expected_position_weight", 0.15)),
            "reid_weight": float(candidate_cfg.get("reid_weight", 0.35)),
        },
        "merge_policy": {
            "apply_merges": True,
            "reid_similarity_threshold": float(threshold),
            "min_mean_confidence": float(candidate_cfg.get("min_mean_confidence", 0.03)),
            "max_combined_pair_score": candidate_cfg.get("max_combined_pair_score"),
            "prevent_duplicate_frame_keys": True,
            "reject_known_false_gt": False,
        },
        "export_compact": {
            "enabled": bool(use_export_compact),
            "classification": {
                "short_rows_threshold": int(selection_cfg.get("compact_short_rows_threshold", 5)),
                "low_mean_confidence_threshold": float(selection_cfg.get("compact_low_mean_confidence_threshold", 0.02)),
                "low_max_confidence_threshold": float(selection_cfg.get("compact_low_max_confidence_threshold", 0.06)),
            },
            "pruning": {
                "enabled": True,
                "class_id": int(candidate_cfg.get("class_id", 0)),
                "mode": "compact",
                "max_rows_per_track": int(selection_cfg.get("compact_max_rows_per_track", 5)),
                "mean_confidence_threshold": float(selection_cfg.get("compact_low_mean_confidence_threshold", 0.02)),
            },
        },
    }
    return run_config
