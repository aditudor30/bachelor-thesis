"""Validate the Step 21B Track1 submission."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import load_bytetrack_pipeline_config
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_validation import validate_bytetrack_track1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate baseline V2 ByteTrack-local Track1")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--track1", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    track1, output_root = _resolve_paths(args.config, args.track1, args.output_root)
    report = validate_bytetrack_track1(track1, output_root=output_root, progress=args.progress)
    print("status: %s" % report.get("status"))
    print("total_rows: %s" % report.get("total_rows"))
    print("num_errors: %s" % report.get("num_errors"))


def _resolve_paths(
    config_path: Optional[Path],
    track1: Optional[Path],
    output_root: Optional[Path],
) -> Tuple[Path, Path]:
    if config_path is not None:
        config = load_bytetrack_pipeline_config(config_path)
        configured_root = Path(str(config.get("paths", {}).get("output_track1_root")))
        track1 = track1 or configured_root / "track1.txt"
        output_root = output_root or configured_root / "validation"
    if track1 is None:
        raise ValueError("Provide --track1 or --config")
    return track1, output_root or track1.parent / "validation"


if __name__ == "__main__":
    main()
