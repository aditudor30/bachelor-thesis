"""CLI for a compact Step 20C summary."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.learned_association_application.scorer_association_sweep import summarize_person_scorer_association


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_person_scorer_association(args.root)
    for key in ("root", "verdict", "selected_variant", "candidate_pairs", "pairs_with_reid", "variants"):
        print("%s: %s" % (key, summary.get(key)))


if __name__ == "__main__":
    main()
