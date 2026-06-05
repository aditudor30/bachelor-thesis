"""CLI for comparing ReID-guided Person association runs."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_reid_association.reid_association_comparison import compare_reid_person_association_runs


def main() -> None:
    """Compare ReID association runs."""
    parser = argparse.ArgumentParser(description="Compare ReID-guided Person association runs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    summary = compare_reid_person_association_runs(args.config, output_root=args.output_root, progress=args.progress)
    print(json.dumps(summary.get("best_reid_person_association_recommendation", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

