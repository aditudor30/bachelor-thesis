"""Print the manual runbook for reproducing the current MVP baseline."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.scripts.check_mvp_outputs import load_mvp_baseline_config


def build_mvp_runbook_commands(config: Dict[str, Any]) -> List[str]:
    """Return the manual commands for the current MVP baseline."""
    dataset_root = str(config.get("dataset_root", "/path/to/MTMC_Tracking_2026"))
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    pipeline_root = _path(paths, "pipeline_run_root", "output/pipeline_runs/yolo11m_medium_curriculum_conf001")
    local_tracks_root = _path(paths, "local_tracks_root", "output/local_tracks/yolo11m_medium_conf001")
    yolo_model_path = _path(
        paths,
        "yolo_model_path",
        "runs/detect/output/yolo_runs/yolo11m_medium_curriculum/weights/best.pt",
    )
    tracklets_root = _path(paths, "tracklets_root", "output/tracklets/yolo11m_medium_conf001_mid")
    candidates_root = _path(paths, "mtmc_candidates_root", "output/mtmc_candidates/yolo11m_medium_conf001_mid_dense")
    motion_clean_root = _path(
        paths,
        "motion_clean_candidates_root",
        "output/mtmc_candidates_motion_clean/yolo11m_medium_conf001_mid_dense",
    )
    global_root = _path(paths, "global_mtmc_root", "output/global_mtmc_transition/yolo11m_medium_conf001_transition")
    commands = [
        "DATA_ROOT=%s" % dataset_root,
        "python -m deep_oc_sort_3d.scripts.run_detection_to_observation_pipeline "
        "--config deep_oc_sort_3d/configs/pipeline_yolo11m_medium_conf001.yaml "
        "--root \"$DATA_ROOT\" --model %s --progress" % yolo_model_path,
        "python -m deep_oc_sort_3d.scripts.run_batch_local_tracking "
        "--config deep_oc_sort_3d/configs/local_tracking_medium_conf001.yaml "
        "--run-root %s --output-root %s "
        "--max-misses 45 --max-3d-distance 4.0 --cost-threshold 0.8 --min-iou 0.0 "
        "--progress" % (pipeline_root, local_tracks_root),
        "python -m deep_oc_sort_3d.scripts.build_batch_local_tracklets "
        "--config deep_oc_sort_3d/configs/tracklet_build_medium_conf001.yaml "
        "--tracking-root %s --output-root %s --progress" % (local_tracks_root, tracklets_root),
        "python -m deep_oc_sort_3d.scripts.export_batch_mtmc_candidates "
        "--config deep_oc_sort_3d/configs/mtmc_candidates_medium_conf001.yaml "
        "--tracklet-root %s --output-root %s "
        "--trajectory-sample-rate 1 --max-trajectory-points 200 --progress" % (tracklets_root, candidates_root),
        "python -m deep_oc_sort_3d.scripts.filter_mtmc_candidates_by_motion "
        "--candidate-root %s --output-root %s "
        "--config deep_oc_sort_3d/configs/mtmc_motion_quality_medium_conf001.yaml --progress"
        % (candidates_root, motion_clean_root),
        "python -m deep_oc_sort_3d.scripts.run_batch_global_mtmc_with_transitions "
        "--candidates-root %s --output-root %s "
        "--config deep_oc_sort_3d/configs/global_mtmc_transition_medium_conf001.yaml --progress"
        % (motion_clean_root, global_root),
        "python -m deep_oc_sort_3d.scripts.run_final_mvp_export "
        "--config deep_oc_sort_3d/configs/final_mvp_export_medium_transition.yaml --progress --overwrite",
        "python -m deep_oc_sort_3d.scripts.check_mvp_outputs "
        "--config deep_oc_sort_3d/configs/mvp_baseline.yaml --progress --fail-on-errors",
    ]
    final_root = paths.get("final_export_root")
    if final_root:
        commands.append(
            "python -m deep_oc_sort_3d.scripts.summarize_final_export "
            "--export-root %s" % final_root
        )
    return commands


def _path(paths: Dict[str, Any], key: str, default: str) -> str:
    value = paths.get(key)
    if value in (None, ""):
        return default
    return str(value)


def print_mvp_runbook(config: Dict[str, Any]) -> None:
    """Print the MVP runbook commands."""
    print("MVP baseline runbook")
    print("Run these manually on the Linux machine. Do not use depth at test time.")
    print("")
    for index, command in enumerate(build_mvp_runbook_commands(config), start=1):
        print("%02d. %s" % (index, command))
    print("")
    print("Official track1.txt export is still TODO until the exact official schema is confirmed.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Print the MVP baseline runbook.")
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_mvp_baseline_config(args.config)
    print_mvp_runbook(config)


if __name__ == "__main__":
    main()
