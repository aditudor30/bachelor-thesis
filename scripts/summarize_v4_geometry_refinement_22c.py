"""Print the compact Step 22C V4 geometry decision summary."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v4_geometry_refinement.geometry_refinement_config import load_geometry_refinement_config, output_root
from deep_oc_sort_3d.v4_geometry_refinement.track1_geometry_io import read_json


def main() -> None:
    args = _parser().parse_args()
    config = load_geometry_refinement_config(Path(args.config))
    root = output_root(config)
    comparison = read_json(root / "comparison" / "v4_geometry_comparison.json")
    package = read_json(root / "packages" / "upload_readiness_report.json")
    print("verdict: %s" % comparison.get("verdict"))
    print("selected_variant: %s" % comparison.get("selected_variant"))
    print("ready_for_upload: %s" % package.get("ready"))
    print("track1_rows: %s" % package.get("track1_rows"))
    print("scene_distribution: %s" % package.get("per_scene_rows"))
    print("class_distribution: %s" % package.get("per_class_rows"))
    print("validation_errors: %s" % package.get("validation_errors"))
    print("zip_size_mb: %s" % package.get("zip_size_mb"))
    print("zip_path: %s" % package.get("zip_path"))
    for row in comparison.get("variants", []):
        print(
            "%s valid=%s aggressive=%s improvements=%s step_p95=%s suspect_tracks=%s dim_var=%s yaw_jumps=%s" % (
                row.get("variant"), row.get("hard_valid"), row.get("too_aggressive"),
                row.get("proxy_improvement_count"), row.get("step_p95"), row.get("suspect_track_count"),
                row.get("dimension_variance_mean"), row.get("yaw_jump_count"),
            )
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None, help="Accepted for CLI consistency.")
    parser.add_argument("--all", action="store_true", help="Accepted for CLI consistency.")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
