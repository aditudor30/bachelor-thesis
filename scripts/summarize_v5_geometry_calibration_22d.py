"""Print compact Step 22D calibration and upload results."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v5_geometry_calibration.geometry_calibration_io import read_json


def main() -> None:
    args = _parser().parse_args()
    root = Path(args.root)
    dataset = read_json(root / "calibration_dataset" / "calibration_matches_summary.json")
    calibration = read_json(root / "validation_diagnostics" / "calibration_verdict.json")
    comparison = read_json(root / "comparison" / "v5_geometry_calibration_summary.json")
    readiness = read_json(root / "frozen_candidate" / "comparison" / "upload_readiness.json")
    package = readiness.get("v5_geometry_calibrated_official", {})
    print("verdict: %s" % comparison.get("verdict"))
    print("selected_variant: %s" % comparison.get("selected_variant"))
    print("fit_source: %s" % calibration.get("fit_source"))
    print("num_matches: %s" % dataset.get("num_matches"))
    print("match_rate: %s" % dataset.get("match_rate"))
    print("samples_per_class: %s" % dataset.get("samples_per_class"))
    print("selected_components: %s" % calibration.get("selected_components"))
    print("ready_for_upload: %s" % package.get("ready"))
    print("track1_rows: %s" % package.get("rows"))
    print("unique_tracks: %s" % package.get("unique_tracks"))
    print("validation_errors: %s" % package.get("validation_errors"))
    print("scene_distribution: %s" % package.get("per_scene_rows"))
    print("class_distribution: %s" % package.get("per_class_rows"))
    print("zip_size_mb: %s" % package.get("zip_size_mb"))
    print("zip_path: %s" % package.get("zip_path"))
    for row in comparison.get("variants", []):
        print("%s valid=%s risk=%s useful=%s position_p95=%s dimension_ratio_p95=%s" % (
            row.get("variant"), row.get("hard_valid"), row.get("quality_risk"),
            row.get("has_selected_correction"), row.get("p95_position_change_m"),
            row.get("p95_dimension_change_ratio"),
        ))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
