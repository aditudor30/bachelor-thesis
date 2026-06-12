"""Run the complete isolated Step 21B pipeline."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_full_pipeline_runner import run_bytetrack_full_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline V2 ByteTrack-local full pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_bytetrack_full_pipeline(
        args.config,
        progress=args.progress,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print("stages: %d" % len(result.get("stages", [])))


if __name__ == "__main__":
    main()
