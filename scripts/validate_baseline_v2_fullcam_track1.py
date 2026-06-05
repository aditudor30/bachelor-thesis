"""Validate baseline_v2_pseudo3d_fullcam Track1 export."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_validation import validate_fullcam_track1


def run(args: Any) -> None:
    """Run Track1 validation."""
    report = validate_fullcam_track1(
        args.track1,
        output_root=args.output_root,
        expected_scene_ids=args.expected_scene_ids,
        valid_class_ids=args.valid_class_ids,
        show_progress=args.progress,
    )
    print("status: %s" % report.get("status"))
    print("total_rows: %s" % report.get("total_rows"))
    print("num_errors: %s" % report.get("num_errors"))
    if args.fail_on_errors and int(report.get("num_errors", 0)) > 0:
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Validate baseline_v2_pseudo3d_fullcam Track1 export.")
    parser.add_argument("--track1", type=Path, default=Path("output/track1_submission/baseline_v2_pseudo3d_fullcam/track1.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("output/track1_submission/baseline_v2_pseudo3d_fullcam/validation"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--expected-scene-ids", nargs="+", type=int, default=[23, 24, 25])
    parser.add_argument("--valid-class-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
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
