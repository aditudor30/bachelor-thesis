"""CLI for Step 16A Person ReID embedding aggregation."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_reid.reid_aggregation import aggregate_person_embeddings_from_config
from deep_oc_sort_3d.person_reid.reid_config import load_person_reid_config


def main() -> None:
    """Aggregate embeddings."""
    parser = argparse.ArgumentParser(description="Aggregate Person ReID embeddings.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    summary = aggregate_person_embeddings_from_config(load_person_reid_config(args.config), show_progress=args.progress, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

