"""CLI for a single Person-aware association experiment."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_association.person_merge_runner import run_person_association_experiment


def main() -> None:
    """Run one experiment."""
    parser = argparse.ArgumentParser(description="Run one Person-aware association experiment.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    output_root = args.output_root
    if output_root is None:
        output_root = Path("output/person_association/baseline_v2_pseudo3d_fullcam/experiments") / args.run_name
    status = run_person_association_experiment(
        args.run_name,
        args.config,
        output_root,
        overwrite=args.overwrite,
        progress=args.progress,
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

