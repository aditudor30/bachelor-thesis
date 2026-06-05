"""CLI for comparing Person-aware association runs."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_association.person_association_comparison import compare_person_association_runs


def main() -> None:
    """Compare runs."""
    parser = argparse.ArgumentParser(description="Compare Person-aware association runs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    summary = compare_person_association_runs(args.config, output_root=args.output_root, progress=args.progress)
    print(json.dumps(summary.get("best_person_association_recommendation", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

