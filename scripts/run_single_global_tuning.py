"""Run one global association tuning preset."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.global_tuning.global_tuning_runner import run_single_global_tuning


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run one global association tuning preset.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    status = run_single_global_tuning(
        run_name=args.run_name,
        config_path=args.config,
        output_root=args.output_root,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        progress=args.progress,
    )
    print("run_name: %s" % status.get("run_name"))
    print("status: %s" % status.get("status"))
    if status.get("error_message"):
        print("error: %s" % status.get("error_message"))


if __name__ == "__main__":
    main()

