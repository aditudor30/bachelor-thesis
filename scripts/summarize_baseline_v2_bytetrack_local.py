"""Print the most important Step 21B summary fields."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize baseline V2 ByteTrack-local comparison")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = read_json(args.root / "baseline_v2_bytetrack_full_summary.json")
    verdict = read_json(args.root / "verdict.json")
    candidate = summary.get("variants", {}).get("baseline_v2_pseudo3d_fullcam_bytetrack_local", {})
    local = candidate.get("local_tracking", {})
    global_metrics = candidate.get("global_association", {})
    track1 = candidate.get("track1", {})
    print("verdict: %s" % verdict.get("label"))
    print("local tracks: %s" % local.get("num_tracks"))
    print("local median length: %s" % local.get("median_track_length"))
    print("local fragmentation: %s" % local.get("approx_fragmentation"))
    print("global tracks: %s" % global_metrics.get("global_tracks"))
    print("global fragmentation: %s" % global_metrics.get("fragmentation_approx"))
    print("track1 rows: %s" % track1.get("rows"))
    print("track1 validation errors: %s" % track1.get("validation_errors"))


if __name__ == "__main__":
    main()
