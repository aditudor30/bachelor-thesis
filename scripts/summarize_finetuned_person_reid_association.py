"""CLI for summarizing Step 18C outputs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import read_json


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Summarize fine-tuned Person ReID association outputs.")
    parser.add_argument("--root", default="output/person_reid_finetuned_association/baseline_v2_pseudo3d_fullcam")
    return parser.parse_args()


def main() -> None:
    """Print a compact summary."""
    args = parse_args()
    root = Path(args.root)
    selected = read_json(root / "comparison" / "selected_variant.json") or {}
    coverage = read_json(root / "diagnostics" / "embedding_coverage_summary.json") or {}
    distribution = read_json(root / "diagnostics" / "reid_score_distribution.json") or {}
    print("root: %s" % root)
    print("verdict: %s" % selected.get("verdict"))
    print("best_run: %s" % selected.get("best_run"))
    print("fragment_embeddings: %s" % coverage.get("valid_fragment_embeddings"))
    print("crop_embeddings: %s" % coverage.get("crop_embeddings"))
    print("pairs_with_reid: %s" % distribution.get("num_pairs_with_reid"))
    print("score_median: %s" % distribution.get("median"))


if __name__ == "__main__":
    main()
