"""CLI for Person cleanup sweep."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.person_cleanup.person_cleanup_runner import run_person_cleanup_sweep


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run Person cleanup sweep.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_person_cleanup_sweep(
        args.config,
        run_names=args.runs,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        progress=args.progress,
    )
    statuses = result.get("statuses", [])
    print("runs: %d" % len(statuses))
    print("errors: %d" % len([item for item in statuses if item.get("status") == "error"]))


if __name__ == "__main__":
    main()

