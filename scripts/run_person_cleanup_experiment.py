"""CLI for one Person cleanup experiment."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.person_cleanup.person_cleanup_runner import run_person_cleanup_experiment


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run one Person cleanup experiment.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    status = run_person_cleanup_experiment(
        args.run_name,
        args.config,
        args.output_root,
        overwrite=args.overwrite,
        progress=args.progress,
    )
    print("run_name: %s" % status.get("run_name"))
    print("status: %s" % status.get("status"))
    if status.get("error_message"):
        print("error: %s" % status.get("error_message"))


if __name__ == "__main__":
    main()

