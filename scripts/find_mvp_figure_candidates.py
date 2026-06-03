"""Find good frame candidates for MVP paper/demo figures."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.visualization3d.figure_candidate_selection import (
    FigureCandidate,
    scan_frame_records_for_candidates,
    select_top_candidates,
)
from deep_oc_sort_3d.visualization3d.figure_quality_scoring import explain_candidate_score


def main() -> None:
    args = parse_args()
    all_selected = []
    tracking = scan_frame_records_for_candidates(
        records_root=args.records_root,
        root=args.root,
        subsets=args.subsets,
        scenes=args.scenes,
        camera_ids=args.camera_ids,
        frame_stride=args.frame_stride,
        max_frames_per_camera=args.max_frames_per_camera,
        figure_type="tracking_2d",
        show_progress=args.progress,
    )
    cuboids = scan_frame_records_for_candidates(
        records_root=args.records_root,
        root=args.root,
        subsets=args.subsets,
        scenes=args.scenes,
        camera_ids=args.camera_ids,
        frame_stride=args.frame_stride,
        max_frames_per_camera=args.max_frames_per_camera,
        figure_type="cuboid_3d",
        show_progress=args.progress,
    )
    all_selected.extend(
        select_top_candidates(
            tracking,
            top_k=args.top_k,
            min_records=args.min_records,
            max_records=args.max_records,
            min_projectable_3d=0,
        )
    )
    all_selected.extend(
        select_top_candidates(
            cuboids,
            top_k=args.top_k,
            min_records=args.min_records,
            max_records=args.max_records,
            min_projectable_3d=args.min_projectable_3d,
        )
    )
    write_candidates_csv(all_selected, args.output)
    summary_path = args.summary_output
    if summary_path is None:
        summary_path = args.output.with_suffix(".summary.json")
    summary = summarize_candidates(all_selected)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("candidates_written: %d" % len(all_selected))
    print("output: %s" % args.output)
    print("summary: %s" % summary_path)
    for candidate in sorted(all_selected, key=lambda item: item.score, reverse=True)[: min(10, len(all_selected))]:
        print(
            "%s %s %s %s frame=%d records=%d score=%.4f"
            % (
                candidate.figure_type,
                candidate.subset,
                candidate.scene_name,
                candidate.camera_id,
                candidate.frame_id,
                candidate.num_records,
                candidate.score,
            )
        )
        print("  %s" % explain_candidate_score(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
    parser.add_argument("--frame-stride", type=int, default=50)
    parser.add_argument("--max-frames-per-camera", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-records", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=30)
    parser.add_argument("--min-projectable-3d", type=int, default=3)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def write_candidates_csv(candidates: List[FigureCandidate], path: Path) -> None:
    """Write candidates as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(candidates[0]).keys()) if candidates else [
        "subset",
        "split",
        "scene_name",
        "camera_id",
        "frame_id",
        "records_path",
        "figure_type",
        "num_records",
        "num_assigned",
        "num_classes",
        "class_counts",
        "num_projectable_3d",
        "projection_success_rate",
        "score",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            row = asdict(candidate)
            row["class_counts"] = json.dumps(row["class_counts"], sort_keys=True)
            writer.writerow(row)


def summarize_candidates(candidates: List[FigureCandidate]) -> Dict[str, Any]:
    """Summarize candidate rows."""
    per_type = {}
    per_subset = {}
    for candidate in candidates:
        per_type[candidate.figure_type] = per_type.get(candidate.figure_type, 0) + 1
        per_subset[candidate.subset] = per_subset.get(candidate.subset, 0) + 1
    return {
        "num_candidates": len(candidates),
        "per_type": per_type,
        "per_subset": per_subset,
        "top_scores": [candidate.score for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)[:10]],
    }


if __name__ == "__main__":
    main()

