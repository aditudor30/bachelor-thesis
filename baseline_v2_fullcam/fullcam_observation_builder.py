"""Build Observation3D files for baseline_v2_pseudo3d_fullcam."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deep_oc_sort_3d.pseudo3d_integration.observation3d_pseudo3d_adapter import (
    merge_observation_with_pseudo3d,
)
from deep_oc_sort_3d.pseudo3d_integration.pseudo3d_observation_builder import (
    build_pseudo3d_observations_batch,
)


REQUIRED_METADATA_FIELDS = [
    "center_3d_source",
    "dimensions_3d_source",
    "yaw_source",
    "depth_source",
    "is_gt_derived",
    "is_estimated_for_test",
    "pseudo3d_method",
    "pseudo3d_version",
    "pseudo3d_confidence",
    "coordinate_frame",
    "source_notes",
]


def load_fullcam_config(path: Path) -> Dict[str, Any]:
    """Load a fullcam baseline config."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_fullcam_observations_from_config(
    config: Dict[str, Any],
    show_progress: bool = True,
    overwrite: bool = False,
    subsets: Optional[List[str]] = None,
    scenes: Optional[List[str]] = None,
    camera_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build full-camera stabilized pseudo3D Observation3D files."""
    builder_config = build_observation_builder_config(config)
    paths = builder_config.get("paths", {})
    selected_subsets = subsets if subsets is not None else list(builder_config.get("subsets", {}).keys())
    selected_scenes = scenes if scenes is not None else scenes_for_subsets(builder_config.get("subsets", {}), selected_subsets)
    summary = build_pseudo3d_observations_batch(
        input_observations_root=Path(paths["input_observations_root"]),
        pseudo3d_predictions_root=Path(paths["pseudo3d_predictions_root"]),
        output_observations_root=Path(paths["output_observations_root"]),
        config=builder_config,
        subsets=selected_subsets,
        scenes=selected_scenes,
        camera_ids=camera_ids,
        show_progress=show_progress,
        overwrite=overwrite,
    )
    enriched = enrich_fullcam_observation_summary(summary, builder_config)
    write_fullcam_observation_diagnostics(enriched, builder_config)
    validate_fullcam_observation_summary(enriched, builder_config)
    return enriched


def build_observation_builder_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a full pipeline config into the existing pseudo3D builder schema."""
    section = config.get("baseline_v2_fullcam", {})
    paths = config.get("paths", {})
    integration = config.get("pseudo3d_integration", {})
    builder = {
        "progress": section.get("progress", True),
        "paths": {
            "input_observations_root": paths.get(
                "input_observations_root",
                "output/pipeline_runs/yolo11m_medium_curriculum_conf001/observations3d",
            ),
            "pseudo3d_predictions_root": paths.get(
                "pseudo3d_predictions_root",
                "output/pseudo3d/baseline_v2_pseudo3d_fullcam/predictions_stabilized",
            ),
            "output_observations_root": paths.get(
                "output_observations_root",
                str(Path(paths.get("output_pipeline_root", "output/pipeline_runs/baseline_v2_pseudo3d_fullcam")) / "observations3d"),
            ),
            "output_metadata_root": paths.get(
                "output_metadata_root",
                str(Path(paths.get("output_pipeline_root", "output/pipeline_runs/baseline_v2_pseudo3d_fullcam")) / "observations3d_metadata"),
            ),
            "output_comparison_root": paths.get(
                "output_comparison_root",
                "output/baseline_v2_pseudo3d_fullcam_comparison",
            ),
            "class_priors": paths.get(
                "class_priors",
                "output/3d_priors/baseline_v1_geometry_only/priors/class_dimension_priors_final.json",
            ),
        },
        "pseudo3d_integration": dict(integration),
        "subsets": config.get("subsets", {}),
    }
    return builder


def enrich_fullcam_observation_summary(summary: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Add fullcam-specific rates and verdict fields to an observation summary."""
    total = int(summary.get("output_observations", 0) or 0)
    used = int(summary.get("pseudo3d_used", 0) or 0)
    fallback = int(summary.get("fallback_original_used", 0) or 0)
    metadata = aggregate_metadata_completeness(summary.get("files_detail", []))
    min_rate = float(config.get("pseudo3d_integration", {}).get("min_required_pseudo3d_used_rate", 0.95))
    fallback_rate = _rate(fallback, total)
    used_rate = _rate(used, total)
    metadata_rate = metadata.get("overall_required_metadata_complete_rate")
    valid = True
    reasons = []
    if used_rate is not None and used_rate < min_rate:
        valid = False
        reasons.append("pseudo3d_used_rate_below_threshold")
    if fallback_rate is not None and fallback_rate >= 0.05:
        valid = False
        reasons.append("fallback_original_used_rate_too_high")
    if metadata_rate is not None and metadata_rate < 0.95:
        valid = False
        reasons.append("metadata_completeness_below_threshold")
    summary.update(
        {
            "baseline_name": "baseline_v2_pseudo3d_fullcam",
            "pseudo3d_used_rate": used_rate,
            "fallback_original_used_rate": fallback_rate,
            "source_metadata_completeness_fullcam": metadata,
            "fullcam_valid_for_pipeline": bool(valid),
            "fullcam_invalid_reasons": reasons,
        }
    )
    return summary


def validate_fullcam_observation_summary(summary: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Raise when fullcam pseudo3D coverage is below configured thresholds."""
    policy = config.get("pseudo3d_integration", {})
    if not bool(policy.get("fail_if_below_threshold", True)):
        return
    if not bool(summary.get("fullcam_valid_for_pipeline", False)):
        raise ValueError("Fullcam pseudo3D observations failed quality thresholds: %s" % summary.get("fullcam_invalid_reasons", []))


def write_fullcam_observation_diagnostics(summary: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Write Step 15H observation diagnostics."""
    paths = config.get("paths", {})
    diagnostics_root = Path(paths.get("output_comparison_root", "output/baseline_v2_pseudo3d_fullcam_comparison")) / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    _write_json(summary, diagnostics_root / "pseudo3d_usage_fullcam.json")
    _write_json(summary.get("source_metadata_completeness_fullcam", {}), diagnostics_root / "source_metadata_completeness_fullcam.json")
    _write_fallback_csv(summary.get("files_detail", []), diagnostics_root / "fallback_usage_fullcam.csv")


def aggregate_metadata_completeness(file_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metadata completeness across per-file observation summaries."""
    totals = {}
    total_observations = 0
    for row in file_rows:
        metadata = row.get("source_metadata_completeness", {})
        total = int(metadata.get("total", row.get("output_observations", 0)) or 0)
        total_observations += total
        for key, value in metadata.items():
            if key.endswith("_complete"):
                totals[key] = totals.get(key, 0) + int(value or 0)
    output = {"total": total_observations}
    required_complete_counts = []
    for key, value in sorted(totals.items()):
        rate_key = "%s_rate" % key
        output[key] = value
        output[rate_key] = _rate(value, total_observations)
        if key.replace("_complete", "") in REQUIRED_METADATA_FIELDS:
            required_complete_counts.append(value)
    if required_complete_counts and total_observations > 0:
        output["overall_required_metadata_complete_rate"] = min(required_complete_counts) / float(total_observations)
    else:
        output["overall_required_metadata_complete_rate"] = None
    return output


def scenes_for_subsets(subsets_cfg: Dict[str, Any], subset_names: Optional[List[str]]) -> Optional[List[str]]:
    """Return configured scenes for selected subsets."""
    if subset_names is None:
        return None
    scenes = []
    for subset in subset_names:
        data = subsets_cfg.get(subset, {})
        if isinstance(data, dict):
            scenes.extend([str(item) for item in data.get("scenes", [])])
    return scenes or None


def merge_observation_for_test(observation: Any, prediction: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Small public wrapper used by tests and docs."""
    return merge_observation_with_pseudo3d(observation, prediction, config)


def _write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_fallback_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "input_path",
        "output_path",
        "status",
        "input_observations",
        "output_observations",
        "pseudo3d_used",
        "pseudo3d_used_rate",
        "fallback_original_used",
        "fallback_original_used_rate",
        "pseudo3d_missing",
        "no_3d_records",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output_observations = int(row.get("output_observations", 0) or 0)
            out = {field: row.get(field) for field in fields}
            out["fallback_original_used_rate"] = _rate(row.get("fallback_original_used", 0), output_observations)
            writer.writerow(out)


def _rate(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        denominator_value = float(denominator)
        if denominator_value <= 0.0:
            return None
        return float(numerator) / denominator_value
    except (TypeError, ValueError):
        return None
