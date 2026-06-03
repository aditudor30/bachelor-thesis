"""Export selected MVP figure candidates as PNG images."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2

from deep_oc_sort_3d.data.calibration import load_calibration_json
from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.frame_io import list_video_files, safe_read_video_frame
from deep_oc_sort_3d.visualization3d.figure_candidate_selection import FigureCandidate
from deep_oc_sort_3d.visualization3d.figure_export_manifest import FigureExportRecord, write_figure_manifest
from deep_oc_sort_3d.visualization3d.frame_visualization import draw_global_frame_records
from deep_oc_sort_3d.visualization3d.visualization_io import filter_records_by_frame, load_global_frame_records_csv


def main() -> None:
    args = parse_args()
    candidates = read_candidates(args.candidates)
    selected = select_for_export(candidates, args.max_tracking_figures, args.max_cuboid_figures)
    manifest = []
    counters = {}
    for candidate in _progress_iter(selected, args.progress, "export selected figures", "figure"):
        counters[candidate.figure_type] = counters.get(candidate.figure_type, 0) + 1
        record = export_candidate_figure(candidate, args, counters[candidate.figure_type])
        if record is not None:
            manifest.append(record)
    manifest_csv = args.output_root / "figure_manifest.csv"
    manifest_json = args.output_root / "figure_manifest.json"
    write_figure_manifest(manifest, manifest_csv)
    write_figure_manifest(manifest, manifest_json)
    print("figures_written: %d" % len(manifest))
    print("manifest_csv: %s" % manifest_csv)
    print("manifest_json: %s" % manifest_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-tracking-figures", type=int, default=5)
    parser.add_argument("--max-cuboid-figures", type=int, default=5)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    draw_2d = parser.add_mutually_exclusive_group()
    draw_2d.add_argument("--draw-2d", dest="draw_2d", action="store_true", default=True)
    draw_2d.add_argument("--no-draw-2d", dest="draw_2d", action="store_false")
    draw_3d = parser.add_mutually_exclusive_group()
    draw_3d.add_argument("--draw-3d", dest="draw_3d", action="store_true", default=True)
    draw_3d.add_argument("--no-draw-3d", dest="draw_3d", action="store_false")
    return parser.parse_args()


def export_candidate_figure(candidate: FigureCandidate, args: argparse.Namespace, figure_number: int) -> Optional[FigureExportRecord]:
    """Render one candidate figure."""
    records = load_global_frame_records_csv(candidate.records_path)
    frame_records = filter_records_by_frame(records, candidate.frame_id)
    frame_records = [record for record in frame_records if str(record.get("camera_id", "")) == candidate.camera_id]
    scene_paths = get_scene_paths(args.root, candidate.split, candidate.scene_name)
    video_path = find_video_path(scene_paths.videos_dir, candidate.camera_id)
    if video_path is None:
        print("warning: missing video for %s %s" % (candidate.scene_name, candidate.camera_id))
        return None
    image = safe_read_video_frame(video_path, candidate.frame_id)
    if image is None:
        print("warning: could not read frame %d from %s" % (candidate.frame_id, video_path))
        return None
    calibration = None
    if scene_paths.calibration_path is not None and scene_paths.calibration_path.exists():
        calibration = load_calibration_json(scene_paths.calibration_path).get(candidate.camera_id)
    draw_3d = bool(args.draw_3d and candidate.figure_type == "cuboid_3d")
    annotated, summary = draw_global_frame_records(
        image,
        frame_records,
        calibration=calibration,
        draw_2d=args.draw_2d,
        draw_3d=draw_3d,
        draw_labels=True,
    )
    figure_name = "%s_%02d" % (candidate.figure_type, figure_number)
    output_path = args.output_root / ("%s.png" % figure_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(
        "%s records=%d bbox=%d cuboids=%d failed=%d"
        % (
            output_path,
            len(frame_records),
            int(summary.get("bbox_drawn", 0)),
            int(summary.get("cuboid_projected", 0)),
            int(summary.get("cuboid_failed", 0)),
        )
    )
    return FigureExportRecord(
        figure_name=figure_name,
        figure_type=candidate.figure_type,
        subset=candidate.subset,
        scene_name=candidate.scene_name,
        camera_id=candidate.camera_id,
        frame_id=candidate.frame_id,
        input_path=candidate.records_path,
        output_path=str(output_path),
        score=candidate.score,
        caption_suggestion=caption_for_candidate(candidate),
        notes=candidate.notes,
    )


def read_candidates(path: Path) -> List[FigureCandidate]:
    """Read candidates exported by find_mvp_figure_candidates."""
    candidates = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidates.append(
                FigureCandidate(
                    subset=str(row.get("subset", "")),
                    split=str(row.get("split", "")),
                    scene_name=str(row.get("scene_name", "")),
                    camera_id=str(row.get("camera_id", "")),
                    frame_id=int(float(row.get("frame_id", 0))),
                    records_path=str(row.get("records_path", "")),
                    figure_type=str(row.get("figure_type", "")),
                    num_records=int(float(row.get("num_records", 0))),
                    num_assigned=int(float(row.get("num_assigned", 0))),
                    num_classes=int(float(row.get("num_classes", 0))),
                    class_counts=_json_dict(row.get("class_counts")),
                    num_projectable_3d=int(float(row.get("num_projectable_3d", 0))),
                    projection_success_rate=_optional_float(row.get("projection_success_rate")),
                    score=float(row.get("score", 0.0)),
                    notes=str(row.get("notes", "")),
                )
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def select_for_export(candidates: List[FigureCandidate], max_tracking: int, max_cuboid: int) -> List[FigureCandidate]:
    """Select a bounded number of candidates per figure type."""
    tracking = [candidate for candidate in candidates if candidate.figure_type == "tracking_2d"][: int(max_tracking)]
    cuboids = [candidate for candidate in candidates if candidate.figure_type == "cuboid_3d"][: int(max_cuboid)]
    return tracking + cuboids


def caption_for_candidate(candidate: FigureCandidate) -> str:
    """Return a short caption suggestion."""
    if candidate.figure_type == "cuboid_3d":
        return (
            "Diagnostic 3D cuboid projection for %s %s frame %d; cuboids are visualization diagnostics."
            % (candidate.scene_name, candidate.camera_id, candidate.frame_id)
        )
    return (
        "Frame-level 2D tracking result with propagated global identities for %s %s frame %d."
        % (candidate.scene_name, candidate.camera_id, candidate.frame_id)
    )


def find_video_path(videos_dir: Optional[Path], camera_id: str) -> Optional[Path]:
    if videos_dir is None:
        return None
    for path in list_video_files(videos_dir):
        if path.stem == camera_id:
            return path
    return None


def _json_dict(value: Any) -> Dict[str, int]:
    if value in (None, ""):
        return {}
    try:
        data = json.loads(str(value))
    except ValueError:
        return {}
    output = {}
    if isinstance(data, dict):
        for key, item in data.items():
            output[str(key)] = int(item)
    return output


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value


if __name__ == "__main__":
    main()
