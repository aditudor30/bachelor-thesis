"""Run isolated pseudo-3D on one frame-record CSV."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.audit3d.audit3d_io import progress_iter, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_config import load_pseudo3d_config
from deep_oc_sort_3d.pseudo3d.pseudo3d_estimator import Pseudo3DEstimator
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import (
    load_scene_camera_calibration,
    prediction_summary,
    read_frame_record_inputs,
    write_pseudo3d_predictions_csv,
    write_pseudo3d_predictions_jsonl,
)
from deep_oc_sort_3d.pseudo3d.pseudo3d_priors import load_pseudo3d_priors


def run(args: Any) -> Dict[str, Any]:
    """Run pseudo-3D estimator on one records file."""
    config = load_pseudo3d_config(args.config)
    priors = load_pseudo3d_priors(args.priors)
    calibration = load_scene_camera_calibration(args.root, args.split, args.scene, args.camera_id)
    inputs = read_frame_record_inputs(
        args.records,
        subset=args.subset,
        split=args.split,
        scene_name=args.scene,
        camera_id=args.camera_id,
        calibration=calibration,
        show_progress=args.progress,
    )
    estimator = Pseudo3DEstimator(priors, config)
    outputs = []
    for item in progress_iter(inputs, args.progress, "estimate pseudo3D records", "record"):
        outputs.append(estimator.estimate(item))
    write_pseudo3d_predictions_jsonl(outputs, args.output_jsonl)
    write_pseudo3d_predictions_csv(outputs, args.output_csv)
    summary = prediction_summary(outputs)
    write_json(summary, Path(args.output_jsonl).with_suffix(".summary.json"))
    print("Pseudo3D predictions: %s" % summary.get("num_predictions"))
    print("Success rate: %s" % summary.get("success_rate"))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated pseudo-3D on one records CSV.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
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

