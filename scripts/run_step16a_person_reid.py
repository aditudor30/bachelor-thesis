"""CLI for the full Step 16A Person ReID pipeline."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_reid.reid_pipeline import run_step16a_person_reid


def main() -> None:
    """Run Step 16A."""
    parser = argparse.ArgumentParser(description="Run Step 16A Person ReID diagnostics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    summary = run_step16a_person_reid(args.config, show_progress=args.progress, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

