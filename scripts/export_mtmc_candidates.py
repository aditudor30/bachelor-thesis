"""Export MTMC-ready candidates from one tracklet file."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List

from deep_oc_sort_3d.mtmc.candidate_builder import MTMCCandidateBuilder
from deep_oc_sort_3d.mtmc.candidate_io import write_candidates_csv, write_candidates_jsonl
from deep_oc_sort_3d.mtmc.candidate_summary import (
    print_candidate_summary,
    summarize_candidates,
    write_candidate_summary_json,
)
from deep_oc_sort_3d.tracklets.tracklet_io import read_tracklets_file
from deep_oc_sort_3d.tracklets.tracklet_types import LocalTracklet


def export_mtmc_candidates(args: Any) -> None:
    """Export candidates for one tracklet file."""
    tracklets = read_tracklets_file(args.tracklets)
    builder = _builder_from_args(args)
    candidates = _build_with_progress(builder, tracklets, args.subset, args.progress, "export MTMC candidates")
    output_candidates = candidates if args.export_rejected else [item for item in candidates if item.is_candidate]
    write_candidates_csv(output_candidates, args.output_csv)
    if args.output_jsonl is not None:
        write_candidates_jsonl(output_candidates, args.output_jsonl)
    summary = summarize_candidates(candidates)
    if args.summary_output is not None:
        write_candidate_summary_json(summary, args.summary_output)
    print_candidate_summary(summary)
    print("tracklets: %d" % len(tracklets))
    print("exported rows: %d" % len(output_candidates))
    print("Wrote CSV: %s" % args.output_csv)


def _builder_from_args(args: Any) -> MTMCCandidateBuilder:
    return MTMCCandidateBuilder(
        min_length=args.min_length,
        min_mean_confidence=args.min_mean_confidence,
        allowed_quality_flags=args.allowed_quality_flags,
        require_valid_for_mtmc=args.require_valid_for_mtmc,
        require_3d=args.require_3d,
        trajectory_sample_rate=args.trajectory_sample_rate,
        max_trajectory_points=args.max_trajectory_points,
        class_allowlist=args.class_allowlist,
        class_blocklist=args.class_blocklist,
    )


def _build_with_progress(
    builder: MTMCCandidateBuilder,
    tracklets: List[LocalTracklet],
    subset: str,
    show_progress: bool,
    desc: str,
):
    output = []
    kept = 0
    for tracklet in _progress_iter(tracklets, show_progress, desc):
        candidate = builder.build_from_tracklet(tracklet, subset)
        output.append(candidate)
        if candidate.is_candidate:
            kept += 1
    return output


def _progress_iter(values: List[LocalTracklet], show_progress: bool, desc: str) -> Iterable[LocalTracklet]:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return _print_progress_iter(values, desc)
    return tqdm(values, desc=desc, unit="tracklet")


def _print_progress_iter(values: List[LocalTracklet], desc: str) -> Iterable[LocalTracklet]:
    total = len(values)
    for index, value in enumerate(values):
        if index == 0 or (index + 1) % 1000 == 0 or index + 1 == total:
            print("%s: tracklet %d/%d" % (desc, index + 1, total))
        yield value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Export MTMC candidates from one tracklet file.")
    parser.add_argument("--tracklets", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--min-mean-confidence", type=float, default=0.01)
    parser.add_argument("--allowed-quality-flags", nargs="+", default=["good", "fragmented"])
    parser.add_argument("--trajectory-sample-rate", type=int, default=5)
    parser.add_argument("--max-trajectory-points", type=int, default=50)
    parser.add_argument("--export-rejected", action="store_true")
    parser.add_argument("--class-allowlist", nargs="+", default=None)
    parser.add_argument("--class-blocklist", nargs="+", default=None)
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument("--require-valid-for-mtmc", dest="require_valid_for_mtmc", action="store_true")
    valid_group.add_argument("--no-require-valid-for-mtmc", dest="require_valid_for_mtmc", action="store_false")
    d_group = parser.add_mutually_exclusive_group()
    d_group.add_argument("--require-3d", dest="require_3d", action="store_true")
    d_group.add_argument("--no-require-3d", dest="require_3d", action="store_false")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(require_valid_for_mtmc=True, require_3d=False, progress=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    export_mtmc_candidates(args)


if __name__ == "__main__":
    main()
