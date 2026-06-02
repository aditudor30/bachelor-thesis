"""Build local tracklets from one local tracks CSV."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List

from deep_oc_sort_3d.tracking.track_io import read_local_tracks_csv
from deep_oc_sort_3d.tracking.track_types import LocalTrackRecord
from deep_oc_sort_3d.tracklets.tracklet_builder import LocalTrackletBuilder
from deep_oc_sort_3d.tracklets.tracklet_eval import evaluate_tracklets, save_tracklet_eval_json
from deep_oc_sort_3d.tracklets.tracklet_io import write_tracklets_csv, write_tracklets_jsonl


def build_local_tracklets(args: Any) -> None:
    """Build local tracklets for one local track CSV."""
    records = read_local_tracks_csv(args.tracks)
    builder = _builder_from_args(args)
    tracklets = _build_with_progress(builder, records, args.progress, "build local tracklets")
    write_tracklets_csv(tracklets, args.output_csv)
    if args.output_jsonl is not None:
        write_tracklets_jsonl(tracklets, args.output_jsonl)
    metrics = evaluate_tracklets(tracklets)
    if args.summary_output is not None:
        save_tracklet_eval_json(metrics, args.summary_output)
    print("records: %d" % len(records))
    print("tracklets: %d" % len(tracklets))
    print("valid_for_mtmc: %s" % metrics.get("valid_for_mtmc"))
    print("Wrote CSV: %s" % args.output_csv)
    if args.output_jsonl is not None:
        print("Wrote JSONL: %s" % args.output_jsonl)


def _build_with_progress(
    builder: LocalTrackletBuilder,
    records: List[LocalTrackRecord],
    show_progress: bool,
    desc: str,
):
    grouped = builder.group_records_by_track(records)
    items = sorted(grouped.items(), key=lambda item: item[0])
    tracklets = []
    for _track_id, track_records in _progress_iter(items, show_progress, desc):
        tracklets.append(builder.build_one_tracklet(track_records))
    return tracklets


def _builder_from_args(args: Any) -> LocalTrackletBuilder:
    return LocalTrackletBuilder(
        min_length=args.min_length,
        min_mean_confidence=args.min_mean_confidence,
        smooth_trajectory=args.smooth_trajectory,
        smoothing_window=args.smoothing_window,
    )


def _progress_iter(values: List[Any], show_progress: bool, desc: str) -> Iterable[Any]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="track")


def _print_progress_iter(values: List[Any], desc: str) -> Iterable[Any]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or (index + 1) % 500 == 0 or index + 1 == total:
            print("%s: track %d/%d" % (desc, index + 1, total))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Build local tracklets from one local track CSV.")
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--min-mean-confidence", type=float, default=0.01)
    parser.add_argument("--smoothing-window", type=int, default=5)
    smooth_group = parser.add_mutually_exclusive_group()
    smooth_group.add_argument("--smooth-trajectory", dest="smooth_trajectory", action="store_true")
    smooth_group.add_argument("--no-smooth-trajectory", dest="smooth_trajectory", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(smooth_trajectory=True, progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    build_local_tracklets(args)


if __name__ == "__main__":
    main()
