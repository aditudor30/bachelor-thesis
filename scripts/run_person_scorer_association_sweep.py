"""CLI for the complete Step 20C conservative sweep."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.learned_association_application.scorer_association_sweep import run_person_scorer_association_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_person_scorer_association_sweep(args.config, progress=args.progress, overwrite=args.overwrite)
    selected = result.get("comparison", {}).get("selected", {})
    print("runs:", len(result.get("runs", [])))
    print("selected_variant:", selected.get("selected_variant"))
    print("verdict:", selected.get("verdict"))


if __name__ == "__main__":
    main()
