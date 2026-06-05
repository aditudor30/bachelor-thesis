"""CLI for mining ReID-guided Person association pairs."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_reid_association.reid_merge_runner import mine_pairs_from_config


def main() -> None:
    """Mine candidate pairs."""
    parser = argparse.ArgumentParser(description="Mine ReID-guided Person candidate pairs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()
    _unused_overwrite = args.overwrite
    summary = mine_pairs_from_config(args.config, output_root=args.output_root, progress=args.progress)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

