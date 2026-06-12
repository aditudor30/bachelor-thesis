"""Package the Step 21B Track1 submission."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_package import package_bytetrack_submission
from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import load_bytetrack_pipeline_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package baseline V2 ByteTrack-local submission")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_bytetrack_pipeline_config(args.config)
    summary = package_bytetrack_submission(config, overwrite=args.overwrite, progress=args.progress)
    print("package_root: %s" % summary.get("package_root"))
    print("manifest_path: %s" % summary.get("manifest_path"))


if __name__ == "__main__":
    main()
