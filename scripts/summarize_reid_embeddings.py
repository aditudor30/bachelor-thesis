"""Summarize ReID embedding outputs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.reid.reid_io import read_reid_embeddings_jsonl
from deep_oc_sort_3d.reid.reid_summary import print_reid_summary, summarize_reid_embeddings, write_summary_csv, write_summary_json


def main() -> None:
    args = parse_args()
    records = []
    for path in sorted(args.reid_root.rglob("*.jsonl")):
        if "embedding" in path.name:
            records.extend(read_reid_embeddings_jsonl(path))
    summary = summarize_reid_embeddings(records)
    print_reid_summary(summary)
    output = args.reid_root / "summaries" / "reid_summary_from_files.json"
    write_summary_json(summary, output)
    write_summary_csv(summary, output.with_suffix(".csv"))
    print("summary: %s" % output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reid-root", type=Path, required=True)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main()

