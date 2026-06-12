"""Run Step 21B extended local-tracker precheck."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_precheck import run_bytetrack_precheck


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ByteTrack-local extended precheck")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_bytetrack_precheck(args.config, progress=args.progress, overwrite=args.overwrite)
    print("verdict: %s" % summary.get("verdict", {}).get("label"))
    print("benchmark_root: %s" % summary.get("benchmark_root"))


if __name__ == "__main__":
    main()
