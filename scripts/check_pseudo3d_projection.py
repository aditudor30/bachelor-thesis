"""Check projection quality for pseudo-3D predictions."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.audit3d.audit3d_io import iter_data_files, progress_iter, write_csv, write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_io import load_scene_camera_calibration, read_pseudo3d_predictions_csv
from deep_oc_sort_3d.pseudo3d.pseudo3d_projection_check import check_prediction_projection, summarize_projection_checks


def run(args: Any) -> Dict[str, Any]:
    rows = []
    for path in progress_iter(iter_data_files(args.predictions_root, [".csv"]), args.progress, "projection prediction files", "file"):
        predictions = read_pseudo3d_predictions_csv(path)
        for prediction in predictions:
            split = str(prediction.get("split") or _split_for_subset(str(prediction.get("subset", ""))))
            calibration = load_scene_camera_calibration(args.root, split, str(prediction.get("scene_name", "")), str(prediction.get("camera_id", "")))
            check = check_prediction_projection(prediction, calibration)
            row = dict(prediction)
            row.update(check)
            rows.append(row)
    summary = summarize_projection_checks(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_root / "projection_summary.json")
    write_csv([row for row in rows if not row.get("projection_valid")], args.output_root / "projection_failures.csv")
    (args.output_root / "optional_examples").mkdir(parents=True, exist_ok=True)
    print("Projection success rate: %s" % summary.get("projection_success_rate"))
    return summary


def _split_for_subset(subset: str) -> str:
    if subset == "official_val":
        return "val"
    if subset == "internal_holdout":
        return "train"
    return "test"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check pseudo-3D projection quality.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
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

