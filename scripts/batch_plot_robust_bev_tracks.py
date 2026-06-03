"""Batch plot robust BEV figures for all generic export scene CSVs."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List

from deep_oc_sort_3d.visualization3d.bev_summary import compute_bev_coordinate_summary, write_bev_summary_json
from deep_oc_sort_3d.visualization3d.bev_track_selection import (
    filter_bev_tracks,
    load_bev_tracks_from_generic_csv,
    load_bev_tracks_from_track1,
    remove_invalid_coordinate_tracks,
)
from deep_oc_sort_3d.visualization3d.robust_bev import plot_robust_bev_tracks


def main() -> None:
    args = parse_args()
    if args.track1 is not None:
        scene_ids = args.scenes if args.scenes is not None else []
        if not scene_ids:
            raise ValueError("--scenes must contain numeric scene ids when using --track1 batch mode")
        items = [("Warehouse_%03d" % int(scene_id), args.track1, int(scene_id)) for scene_id in scene_ids]
    else:
        items = find_generic_scene_files(args.generic_export_root, args.scenes)
    written = 0
    for scene_name, path, scene_id in _progress_iter(items, args.progress, "plot robust BEV", "scene"):
        tracks = load_one(path, scene_id, args)
        output = args.output_root / ("%s_robust_bev.png" % scene_name)
        summary_output = args.output_root / ("%s_robust_bev_summary.json" % scene_name)
        summary = compute_bev_coordinate_summary(tracks, args.lower_percentile, args.upper_percentile)
        plot_summary = plot_robust_bev_tracks(
            tracks,
            output,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            use_percentile_clipping=args.use_percentile_clipping,
            show_start_end=True,
            draw_points=False,
            max_tracks=args.max_tracks,
            equal_aspect=True,
            note_not_map_aligned=True,
            figsize=(args.figsize_width, args.figsize_height),
        )
        summary["plot"] = plot_summary
        write_bev_summary_json(summary, summary_output)
        written += 1
        print("%s tracks=%d points=%d" % (output, summary.get("num_tracks", 0), summary.get("num_points", 0)))
    print("figures_written: %d" % written)
    print("output_root: %s" % args.output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic-export-root", type=Path, default=None)
    parser.add_argument("--track1", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--min-track-length", type=int, default=5)
    parser.add_argument("--max-tracks", type=int, default=100)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--sort-by", choices=["length", "mean_confidence", "duration"], default="length")
    parser.add_argument("--lower-percentile", type=float, default=2.0)
    parser.add_argument("--upper-percentile", type=float, default=98.0)
    parser.add_argument("--figsize-width", type=float, default=8.0)
    parser.add_argument("--figsize-height", type=float, default=8.0)
    parser.add_argument("--no-percentile-clipping", dest="use_percentile_clipping", action="store_false", default=True)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def find_generic_scene_files(root: Path, scenes: Any) -> List[Any]:
    """Find generic scene CSV files under a root."""
    if root is None:
        raise ValueError("--generic-export-root is required unless --track1 is used")
    scene_filter = None if scenes is None else set([str(item) for item in scenes])
    items = []
    for path in sorted(root.glob("*.csv")):
        scene_name = path.stem
        if scene_filter is not None and scene_name not in scene_filter:
            continue
        items.append((scene_name, path, None))
    return items


def load_one(path: Path, scene_id: Any, args: argparse.Namespace):
    if args.track1 is not None:
        tracks = load_bev_tracks_from_track1(path, scene_id=scene_id)
    else:
        tracks = load_bev_tracks_from_generic_csv(path)
    tracks = remove_invalid_coordinate_tracks(tracks)
    return filter_bev_tracks(
        tracks,
        min_track_length=args.min_track_length,
        max_tracks=args.max_tracks,
        class_id=args.class_id,
        class_name=args.class_name,
        sort_by=args.sort_by,
    )


def _progress_iter(values: List[Any], show_progress: bool, desc: str, unit: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit=unit)


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value


if __name__ == "__main__":
    main()
