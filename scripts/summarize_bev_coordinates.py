"""Summarize BEV coordinates and outliers without plotting."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.visualization3d.bev_summary import (
    compute_bev_coordinate_summary,
    print_bev_summary,
    write_bev_summary_csv,
    write_bev_summary_json,
)
from deep_oc_sort_3d.visualization3d.bev_track_selection import (
    load_bev_tracks_from_generic_csv,
    load_bev_tracks_from_track1,
    remove_invalid_coordinate_tracks,
)


def main() -> None:
    args = parse_args()
    if args.generic_csv is not None:
        tracks = load_bev_tracks_from_generic_csv(args.generic_csv)
    elif args.track1 is not None:
        tracks = load_bev_tracks_from_track1(args.track1, scene_id=args.scene_id)
    else:
        raise ValueError("Provide --generic-csv or --track1")
    tracks = remove_invalid_coordinate_tracks(tracks)
    summary = compute_bev_coordinate_summary(tracks, args.lower_percentile, args.upper_percentile)
    write_bev_summary_json(summary, args.output)
    if args.csv_output is not None:
        write_bev_summary_csv(summary, args.csv_output)
    print_bev_summary(summary)
    print("recommendation: use percentile clipping for paper figures; filtering is visualization-only")
    print("output: %s" % args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic-csv", type=Path, default=None)
    parser.add_argument("--track1", type=Path, default=None)
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--lower-percentile", type=float, default=2.0)
    parser.add_argument("--upper-percentile", type=float, default=98.0)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main()

