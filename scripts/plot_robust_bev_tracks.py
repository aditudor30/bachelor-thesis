"""Plot robust coordinate-space BEV trajectories from generic CSV or Track 1."""

import argparse
from pathlib import Path
from typing import Any

from deep_oc_sort_3d.visualization3d.bev_summary import (
    compute_bev_coordinate_summary,
    print_bev_summary,
    write_bev_summary_json,
)
from deep_oc_sort_3d.visualization3d.bev_track_selection import (
    filter_bev_tracks,
    load_bev_tracks_from_generic_csv,
    load_bev_tracks_from_track1,
    remove_invalid_coordinate_tracks,
)
from deep_oc_sort_3d.visualization3d.robust_bev import plot_robust_bev_tracks


def main() -> None:
    args = parse_args()
    tracks = load_tracks(args)
    tracks = remove_invalid_coordinate_tracks(tracks)
    tracks = filter_bev_tracks(
        tracks,
        min_track_length=args.min_track_length,
        max_tracks=args.max_tracks,
        class_id=args.class_id,
        class_name=args.class_name,
        sort_by=args.sort_by,
    )
    summary = compute_bev_coordinate_summary(tracks, args.lower_percentile, args.upper_percentile)
    plot_summary = plot_robust_bev_tracks(
        tracks,
        args.output,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        use_percentile_clipping=args.use_percentile_clipping,
        show_start_end=args.show_start_end,
        draw_points=args.draw_points,
        max_tracks=args.max_tracks,
        equal_aspect=True,
        note_not_map_aligned=True,
        figsize=(args.figsize_width, args.figsize_height),
    )
    summary["plot"] = plot_summary
    if args.summary_output is not None:
        write_bev_summary_json(summary, args.summary_output)
    if args.pdf_output is not None:
        plot_robust_bev_tracks(
            tracks,
            args.pdf_output,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            use_percentile_clipping=args.use_percentile_clipping,
            show_start_end=args.show_start_end,
            draw_points=args.draw_points,
            max_tracks=args.max_tracks,
            equal_aspect=True,
            note_not_map_aligned=True,
            figsize=(args.figsize_width, args.figsize_height),
        )
    print_bev_summary(summary)
    print("output: %s" % args.output)
    if args.summary_output is not None:
        print("summary_output: %s" % args.summary_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic-csv", type=Path, default=None)
    parser.add_argument("--track1", type=Path, default=None)
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--pdf-output", type=Path, default=None)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--min-track-length", type=int, default=5)
    parser.add_argument("--max-tracks", type=int, default=100)
    parser.add_argument("--sort-by", choices=["length", "mean_confidence", "duration"], default="length")
    parser.add_argument("--x-column", default="center_x")
    parser.add_argument("--y-column", default="center_y")
    parser.add_argument("--lower-percentile", type=float, default=2.0)
    parser.add_argument("--upper-percentile", type=float, default=98.0)
    parser.add_argument("--no-percentile-clipping", dest="use_percentile_clipping", action="store_false", default=True)
    parser.add_argument("--show-start-end", dest="show_start_end", action="store_true", default=True)
    parser.add_argument("--no-show-start-end", dest="show_start_end", action="store_false")
    parser.add_argument("--draw-points", action="store_true", default=False)
    parser.add_argument("--figsize-width", type=float, default=8.0)
    parser.add_argument("--figsize-height", type=float, default=8.0)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def load_tracks(args: argparse.Namespace):
    """Load tracks from the selected source."""
    if args.generic_csv is not None:
        return load_bev_tracks_from_generic_csv(args.generic_csv, x_column=args.x_column, y_column=args.y_column)
    if args.track1 is not None:
        x_column = "x" if args.x_column == "center_x" else args.x_column
        y_column = "y" if args.y_column == "center_y" else args.y_column
        return load_bev_tracks_from_track1(args.track1, scene_id=args.scene_id, x_column_name=x_column, y_column_name=y_column)
    raise ValueError("Provide --generic-csv or --track1")


if __name__ == "__main__":
    main()
