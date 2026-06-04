"""Compare baseline_v1 and baseline_v2 pseudo-3D on the 6-camera subset."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import write_csv, write_json, write_markdown
from deep_oc_sort_3d.experiments.baseline_subset_metrics import collect_all_subset_metrics
from deep_oc_sort_3d.experiments.sixcam_comparison import compare_metric_dicts, metric_delta_rows
from deep_oc_sort_3d.experiments.sixcam_report import build_sixcam_report
from deep_oc_sort_3d.experiments.sixcam_subset import read_sixcam_subset_json, sixcam_items_from_config


def run(args: Any) -> Dict[str, Any]:
    cfg = _load_yaml(args.config)
    output_root = Path(cfg.get("experiment", {}).get("output_root", "output/baseline_v2_pseudo3d_6cam_comparison"))
    items = _items(cfg, output_root)
    v1 = collect_all_subset_metrics(_paths(cfg, "v1"), items, "baseline_v1_geometry_only_6cam")
    v2 = collect_all_subset_metrics(_paths(cfg, "v2_6cam"), items, "baseline_v2_pseudo3d_6cam")
    summary = compare_metric_dicts(v1, v2)
    comparison_root = output_root / "comparison"
    diagnostics_root = output_root / "diagnostics"
    _write_metric_bundle(v1, output_root / "v1_subset_metrics")
    _write_metric_bundle(v2, output_root / "v2_subset_metrics")
    write_json(summary, comparison_root / "v1_vs_v2_6cam_summary.json")
    write_csv(metric_delta_rows(summary.get("deltas", {})), comparison_root / "metric_deltas.csv")
    write_csv(_per_camera_rows(v1, v2), comparison_root / "per_camera_comparison.csv")
    write_csv(_per_scene_rows(v1, v2), comparison_root / "per_scene_comparison.csv")
    write_csv(_per_class_rows(v1, v2), comparison_root / "per_class_comparison.csv")
    write_json(summary.get("verdict", {}), comparison_root / "verdict.json")
    write_csv(_summary_rows(summary), comparison_root / "v1_vs_v2_6cam_summary.csv")
    write_markdown(build_sixcam_report(summary, items), comparison_root / "BASELINE_V1_VS_V2_PSEUDO3D_6CAM_REPORT.md")
    write_json(_pseudo3d_coverage(v2), diagnostics_root / "pseudo3d_coverage_6cam.json")
    write_json(_source_metadata(v2), diagnostics_root / "source_metadata_completeness_6cam.json")
    write_json({"status": "not_available", "reason": "Run smoothness audit separately on 6cam exports if needed."}, diagnostics_root / "smoothness_6cam_comparison.json")
    write_json({"status": "not_available", "reason": "Projection checks are not produced by the subset metric collector."}, diagnostics_root / "projection_6cam_comparison.json")
    write_csv(_fallback_usage_rows(v2), diagnostics_root / "fallback_usage_6cam.csv")
    print("sixcam verdict: %s" % summary.get("verdict", {}).get("label"))
    return summary


def _items(cfg: Dict[str, Any], output_root: Path) -> Any:
    items = read_sixcam_subset_json(output_root / "subset_definition" / "six_camera_subset.json")
    return items if items else sixcam_items_from_config(cfg)


def _paths(cfg: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    paths = cfg.get("paths", {})
    return {
        "pipeline_root": paths.get("%s_pipeline_root" % prefix),
        "local_tracks_root": paths.get("%s_local_tracks_root" % prefix),
        "tracklets_root": paths.get("%s_tracklets_root" % prefix),
        "candidates_root": paths.get("%s_candidates_root" % prefix),
        "motion_clean_root": paths.get("%s_motion_clean_root" % prefix),
        "global_root": paths.get("%s_global_root" % prefix),
        "final_export_root": paths.get("%s_final_export_root" % prefix),
    }


def _write_metric_bundle(metrics: Dict[str, Any], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    names = {
        "observations": "observations_summary",
        "local_tracking": "local_tracking_summary",
        "tracklets": "tracklets_summary",
        "candidates": "candidates_summary",
        "motion_clean": "motion_clean_summary",
        "global_association": "global_association_summary",
        "final_export": "final_export_summary",
    }
    for key, name in names.items():
        write_json(metrics.get(key, {}), root / ("%s.json" % name))
    write_json(metrics, root / "track1_like_summary.json")


def _per_camera_rows(v1: Dict[str, Any], v2: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    v1_rows = _rows_by_camera(v1.get("observations", {}).get("rows", []))
    v2_rows = _rows_by_camera(v2.get("observations", {}).get("rows", []))
    for key in sorted(set(list(v1_rows.keys()) + list(v2_rows.keys()))):
        a = v1_rows.get(key, {})
        b = v2_rows.get(key, {})
        out.append({"camera_key": key, "v1_observations": a.get("num_observations"), "v2_observations": b.get("num_observations"), "v2_pseudo3d_used_rate": b.get("pseudo3d_used_rate"), "v2_fallback_rate": b.get("fallback_original_used_rate")})
    return out


def _per_scene_rows(v1: Dict[str, Any], v2: Dict[str, Any]) -> List[Dict[str, Any]]:
    camera_rows = _per_camera_rows(v1, v2)
    scene_rows = {}
    for row in camera_rows:
        parts = str(row.get("camera_key", "")).split("/")
        scene = "/".join(parts[:2]) if len(parts) >= 2 else str(row.get("camera_key"))
        item = scene_rows.setdefault(scene, {"scene_key": scene, "v1_observations": 0, "v2_observations": 0, "v2_pseudo3d_used": 0})
        item["v1_observations"] += int(row.get("v1_observations") or 0)
        item["v2_observations"] += int(row.get("v2_observations") or 0)
    return list(scene_rows.values())


def _per_class_rows(v1: Dict[str, Any], v2: Dict[str, Any]) -> List[Dict[str, Any]]:
    v1_counts = _merge_class_counts(v1.get("final_export", {}).get("rows", []))
    v2_counts = _merge_class_counts(v2.get("final_export", {}).get("rows", []))
    rows = []
    for key in sorted(set(list(v1_counts.keys()) + list(v2_counts.keys()))):
        rows.append({"class_id": key, "v1_rows": v1_counts.get(key, 0), "v2_rows": v2_counts.get(key, 0), "delta": v2_counts.get(key, 0) - v1_counts.get(key, 0)})
    return rows


def _rows_by_camera(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in rows:
        key = "%s/%s/%s" % (row.get("subset"), row.get("scene_name"), row.get("camera_id"))
        out[key] = row
    return out


def _merge_class_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {}
    for row in rows:
        data = row.get("class_distribution", {})
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
    return counts


def _pseudo3d_coverage(v2: Dict[str, Any]) -> Dict[str, Any]:
    observations = v2.get("observations", {})
    return {
        "pseudo3d_used": observations.get("pseudo3d_used"),
        "pseudo3d_used_rate": observations.get("pseudo3d_used_rate"),
        "fallback_original_used": observations.get("fallback_original_used"),
        "no_3d_records": observations.get("no_3d_records"),
    }


def _source_metadata(v2: Dict[str, Any]) -> Dict[str, Any]:
    rows = v2.get("observations", {}).get("rows", [])
    return {"per_camera": [{"camera": "%s/%s/%s" % (row.get("subset"), row.get("scene_name"), row.get("camera_id")), "metadata_completeness": row.get("metadata_completeness")} for row in rows]}


def _fallback_usage_rows(v2: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in v2.get("observations", {}).get("rows", []):
        rows.append({"subset": row.get("subset"), "scene_name": row.get("scene_name"), "camera_id": row.get("camera_id"), "pseudo3d_used_rate": row.get("pseudo3d_used_rate"), "fallback_original_used_rate": row.get("fallback_original_used_rate")})
    return rows


def _summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"metric": key, "value": value} for key, value in sorted(summary.get("deltas", {}).items())]


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    if data.get("include_config"):
        return _load_yaml(Path(data["include_config"]))
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline_v1 and baseline_v2 6cam metrics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
