"""CLI for Step 20C comparison without rerunning association."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.learned_association_application.scorer_association_sweep import compare_person_scorer_association


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Accepted for CLI compatibility; comparison files are refreshed.")
    args = parser.parse_args()
    result = compare_person_scorer_association(args.config, progress=args.progress)
    print("variants:", len(result.get("variants", [])))
    print("selected_variant:", result.get("selected", {}).get("selected_variant"))
    print("verdict:", result.get("selected", {}).get("verdict"))


if __name__ == "__main__":
    main()
