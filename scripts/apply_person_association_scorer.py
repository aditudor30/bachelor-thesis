"""CLI for Step 20C candidate feature adaptation and MLP scoring."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.learned_association_application.scorer_association_sweep import apply_person_association_scorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = apply_person_association_scorer(args.config, progress=args.progress, overwrite=args.overwrite)
    print("status:", summary.get("status"))
    print("candidate_pairs:", summary.get("candidate_pairs"))
    print("output_root:", summary.get("output_root"))


if __name__ == "__main__":
    main()
