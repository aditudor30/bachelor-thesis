"""CLI for the Person-aware association sweep."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_association.person_merge_runner import run_person_association_sweep


def main() -> None:
    """Run the sweep."""
    parser = argparse.ArgumentParser(description="Run Person-aware association sweep.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-names", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    summary = run_person_association_sweep(
        args.config,
        run_names=args.run_names,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        progress=args.progress,
    )
    compact = {"statuses": summary.get("statuses", []), "verdict": summary.get("comparison", {}).get("best_person_association_recommendation", {}).get("verdict")}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

