"""Compute simple ReID similarity diagnostics."""

import argparse

from deep_oc_sort_3d.reid.reid_diagnostics import compute_pairwise_similarity_sample
from deep_oc_sort_3d.reid.reid_io import read_reid_embeddings_jsonl, write_reid_summary_json


def main() -> None:
    args = parse_args()
    records = read_reid_embeddings_jsonl(args.embeddings)
    summary = compute_pairwise_similarity_sample(records, max_pairs=args.max_pairs, same_class_only=args.same_class_only)
    write_reid_summary_json(summary, args.output)
    print("records: %d" % len(records))
    print("pairs: %s" % str(summary.get("num_pairs")))
    print("similarity_mean: %s" % str(summary.get("similarity_mean")))
    print("output: %s" % args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--same-class-only", dest="same_class_only", action="store_true", default=True)
    parser.add_argument("--all-classes", dest="same_class_only", action="store_false")
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main()

