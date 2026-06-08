"""Summarize SmartSpaces Person ReID dataset outputs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.reid_training.reid_dataset_io import read_csv_rows, read_json


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Summarize SmartSpaces Person ReID dataset.")
    parser.add_argument("--root", default="output/reid_training/person_smartspaces_v1", type=Path)
    args = parser.parse_args()
    root = args.root
    summary = read_json(root / "diagnostics" / "dataset_summary.json") or {}
    warnings = read_json(root / "diagnostics" / "warnings.json") or {}
    crops, _fields = read_csv_rows(root / "metadata" / "all_crops.csv")
    triplets, _triplet_fields = read_csv_rows(root / "pairs_triplets" / "triplets_train.csv")
    print("root:", root)
    print("verdict:", summary.get("verdict"))
    print("metadata rows:", len(crops))
    print("total valid crops:", summary.get("total_crops"))
    print("train crops:", summary.get("train_crops"))
    print("val crops:", summary.get("val_crops"))
    print("train identities:", summary.get("num_train_identities"))
    print("val identities:", summary.get("num_val_identities"))
    print("identity overlap:", summary.get("identity_overlap_train_val"))
    print("train triplets:", len(triplets))
    print("rare identity count:", warnings.get("rare_identity_count"))
    print("invalid crop count:", warnings.get("invalid_crop_count"))


if __name__ == "__main__":
    main()

