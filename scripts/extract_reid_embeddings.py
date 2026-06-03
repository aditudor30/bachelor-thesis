"""Extract ReID embeddings for one scene/camera frame-record CSV."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.frame_io import list_video_files, safe_read_video_frame
from deep_oc_sort_3d.reid.crop_extraction import extract_crops_for_tracklet
from deep_oc_sort_3d.reid.embedding_aggregation import compute_tracklet_embedding
from deep_oc_sort_3d.reid.embedding_backends import build_embedding_backend
from deep_oc_sort_3d.reid.reid_io import write_reid_embeddings_jsonl, write_reid_embeddings_npy
from deep_oc_sort_3d.reid.reid_summary import print_reid_summary, summarize_reid_embeddings, write_summary_json


def main() -> None:
    args = parse_args()
    records = read_records(args.records)
    records = [record for record in records if str(record.get("camera_id", args.camera_id)) == args.camera_id]
    video_path = find_video_path(args.root, args.split, args.scene, args.camera_id)
    if video_path is None:
        raise FileNotFoundError("Missing video for %s %s" % (args.scene, args.camera_id))
    frame_loader = lambda frame_id: safe_read_video_frame(video_path, int(frame_id))
    backend = build_embedding_backend(backend_config_from_args(args))
    groups = group_records(records, key="local_track_id")
    embeddings = []
    crop_root = args.crop_output_dir
    if crop_root is None:
        crop_root = args.output_jsonl.parent / "crops"
    for _key, group_records_for_track in _progress_iter(list(groups.items()), args.progress, "extract ReID", "track"):
        crops, samples = extract_crops_for_tracklet(
            group_records_for_track,
            frame_loader=frame_loader,
            max_crops=args.max_crops_per_tracklet,
            padding_ratio=args.padding_ratio,
            resize=(args.resize_width, args.resize_height),
            save_crops=args.save_crops,
            crop_output_dir=crop_root,
        )
        if not crops:
            continue
        crop_embeddings = [backend.extract(crop) for crop in crops]
        track_embedding = compute_tracklet_embedding(crop_embeddings, samples, method=args.aggregation, backend=backend.name)
        if track_embedding is not None:
            embeddings.append(track_embedding)
    write_reid_embeddings_jsonl(embeddings, args.output_jsonl)
    if args.output_npy is not None:
        metadata_output = args.metadata_output
        if metadata_output is None:
            metadata_output = args.output_npy.with_suffix(".metadata.csv")
        write_reid_embeddings_npy(embeddings, args.output_npy, metadata_output)
    summary = summarize_reid_embeddings(embeddings)
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    write_summary_json(summary, summary_path)
    print_reid_summary(summary)
    print("output_jsonl: %s" % args.output_jsonl)
    print("summary: %s" % summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--backend", choices=["color_histogram", "dummy", "torchvision_resnet"], default="color_histogram")
    parser.add_argument("--aggregation", choices=["mean", "median", "confidence_weighted_mean"], default="mean")
    parser.add_argument("--max-crops-per-tracklet", type=int, default=8)
    parser.add_argument("--padding-ratio", type=float, default=0.10)
    parser.add_argument("--resize-width", type=int, default=128)
    parser.add_argument("--resize-height", type=int, default=256)
    parser.add_argument("--save-crops", dest="save_crops", action="store_true", default=False)
    parser.add_argument("--no-save-crops", dest="save_crops", action="store_false")
    parser.add_argument("--crop-output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def backend_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build backend config from CLI args."""
    return {
        "backend": args.backend,
        "resize": (args.resize_width, args.resize_height),
        "device": args.device,
        "fallback_backend": "color_histogram",
        "bins_per_channel": 16,
        "color_space": "rgb",
    }


def read_records(path: Path) -> List[Dict[str, Any]]:
    """Read frame global records from CSV."""
    records = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(dict(row))
    return records


def group_records(records: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by a key, skipping empty keys."""
    groups = {}
    for record in records:
        value = record.get(key)
        if value in (None, ""):
            continue
        groups.setdefault(str(value), []).append(record)
    return groups


def find_video_path(root: Path, split: str, scene: str, camera_id: str) -> Optional[Path]:
    """Find a camera video path."""
    scene_paths = get_scene_paths(root, split, scene)
    if scene_paths.videos_dir is None:
        return None
    for path in list_video_files(scene_paths.videos_dir):
        if path.stem == camera_id:
            return path
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
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == total:
            print("%s: %d/%d" % (desc, index + 1, total))
        yield value


if __name__ == "__main__":
    main()

