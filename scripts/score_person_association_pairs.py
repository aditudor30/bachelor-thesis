"""CLI for scoring Person association candidate pairs."""

import argparse
import json
from pathlib import Path

from deep_oc_sort_3d.person_association.person_merge_runner import score_pairs_from_config


def main() -> None:
    """Run pair scoring."""
    parser = argparse.ArgumentParser(description="Score Person association candidate pairs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _unused_overwrite = args.overwrite
    summary = score_pairs_from_config(args.config, input_csv=args.input_csv, output_root=args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

