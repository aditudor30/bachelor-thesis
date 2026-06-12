"""Print the most important Step 21C results."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_tuning.tuning_io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ByteTrack coverage tuning")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    selected = read_json(args.root / "comparison" / "selected_bytetrack_config.json")
    verdict = read_json(args.root / "comparison" / "verdict.json")
    metrics = selected.get("selected_metrics") or {}
    print("verdict: %s" % verdict.get("label"))
    print("selected_variant: %s" % selected.get("selected_variant"))
    print("local_records_retention: %s" % metrics.get("local_records_retention"))
    print("gt_matched_retention: %s" % metrics.get("gt_matched_retention"))
    print("track1_rows_retention: %s" % metrics.get("track1_rows_retention"))
    print("multi_camera_tracks_retention: %s" % metrics.get("multi_camera_tracks_retention"))
    print("median_track_length: %s" % metrics.get("median_track_length"))
    print("short_track_ratio_le3: %s" % metrics.get("short_track_ratio_le3"))
    print("global_purity_mean: %s" % metrics.get("global_purity_mean"))
    print("false_merge_rate: %s" % metrics.get("false_merge_rate"))
    print("track1_validation_errors: %s" % metrics.get("track1_validation_errors"))


if __name__ == "__main__":
    main()

