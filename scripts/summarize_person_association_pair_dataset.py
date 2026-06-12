"""Print a compact summary of a generated Step 20A dataset."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.learned_association.pair_dataset_io import read_csv_rows, read_json


def main() -> None:
    """Read generated artifacts without rebuilding the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    verdict = read_json(root / "diagnostics" / "dataset_verdict.json", {}) or {}
    labels = read_json(root / "diagnostics" / "label_distribution.json", {}) or {}
    fragments = read_csv_rows(root / "metadata" / "fragments_all.csv")
    missing = read_csv_rows(root / "features" / "missing_feature_report.csv")
    print("root: %s" % root)
    print("verdict: %s" % verdict.get("verdict"))
    print("ready_for_step_20b: %s" % verdict.get("ready_for_step_20b"))
    print("fragments: %d" % len(fragments))
    print("valid fragments: %s" % verdict.get("num_valid_fragments"))
    print("fragments with embeddings: %s" % verdict.get("num_fragments_with_embedding"))
    print("positive pairs: %s" % labels.get("positive_pairs"))
    print("negative pairs: %s" % labels.get("negative_pairs"))
    print("hard negatives: %s" % labels.get("hard_negatives"))
    print("balanced train pairs: %s" % labels.get("balanced_train_pairs"))
    print("balanced val pairs: %s" % labels.get("balanced_val_pairs"))
    print("feature missingness:")
    for row in missing:
        print("  %s: %s" % (row.get("feature"), row.get("missing_rate")))
    reasons = verdict.get("reasons") or []
    if reasons:
        print("warnings/reasons: %s" % ", ".join(str(item) for item in reasons))


if __name__ == "__main__":
    main()
