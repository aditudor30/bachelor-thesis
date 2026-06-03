"""Batch extraction of ReID embeddings from frame_global_records."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from deep_oc_sort_3d.data.dataset_structure import get_scene_paths
from deep_oc_sort_3d.data.frame_io import list_video_files, safe_read_video_frame
from deep_oc_sort_3d.reid.crop_extraction import extract_crops_for_tracklet
from deep_oc_sort_3d.reid.embedding_aggregation import compute_candidate_embedding, compute_tracklet_embedding
from deep_oc_sort_3d.reid.embedding_backends import build_embedding_backend
from deep_oc_sort_3d.reid.reid_io import (
    write_reid_embeddings_jsonl,
    write_reid_embeddings_npy,
    write_reid_summary_csv,
    write_reid_summary_json,
)
from deep_oc_sort_3d.reid.reid_types import ReIDEmbeddingRecord
from deep_oc_sort_3d.reid.reid_summary import summarize_reid_embeddings


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    options = build_options(config, args)
    backend = build_embedding_backend(backend_config(options))
    record_files = find_record_files(Path(options["records_root"]), options)
    all_records = []
    summaries = []
    for subset, scene_name, camera_id, records_path in _progress_iter(record_files, options["progress"], "batch ReID", "camera"):
        crop_records, tracklet_records, candidate_records, summary = process_camera(subset, scene_name, camera_id, records_path, options, backend)
        all_records.extend(crop_records)
        all_records.extend(tracklet_records)
        summaries.append(summary)
        if candidate_records:
            all_records.extend(candidate_records)
    global_summary = summarize_reid_embeddings(all_records)
    global_summary["camera_summaries"] = summaries
    summary_root = Path(options["output_root"]) / "summaries"
    write_reid_summary_json(global_summary, summary_root / "reid_extraction_summary.json")
    write_reid_summary_csv(summaries, summary_root / "reid_extraction_summary.csv")
    write_reid_summary_csv(per_field_rows(all_records, "class_name"), summary_root / "per_class_summary.csv")
    write_reid_summary_csv(per_field_rows(all_records, "subset"), summary_root / "per_subset_summary.csv")
    print("processed_cameras: %d" % len(summaries))
    print("total_embeddings: %d" % len(all_records))
    print("summary: %s" % (summary_root / "reid_extraction_summary.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--records-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--camera-ids", nargs="+", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--max-crops-per-tracklet", type=int, default=None)
    parser.add_argument("--save-crops", dest="save_crops", action="store_true", default=None)
    parser.add_argument("--no-save-crops", dest="save_crops", action="store_false")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--progress", dest="progress", action="store_true", default=None)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def process_camera(subset: str, scene_name: str, camera_id: str, records_path: Path, options: Dict[str, Any], backend: Any):
    """Process one scene/camera CSV."""
    split = subset_to_split(subset)
    records = read_records(records_path)
    video_path = find_video_path(Path(options["root"]), split, scene_name, camera_id)
    if video_path is None:
        return [], [], [], {"subset": subset, "scene_name": scene_name, "camera_id": camera_id, "status": "missing_video", "records": len(records)}
    frame_loader = lambda frame_id: safe_read_video_frame(video_path, int(frame_id))
    tracklet_groups = group_records(records, "local_track_id", options.get("max_tracks_per_camera"))
    tracklet_embeddings = []
    candidate_embeddings = []
    crop_embedding_records = []
    crop_base = Path(options["output_root"]) / "crops_debug" / subset / scene_name / camera_id
    for local_track_id, group in tracklet_groups.items():
        crops, samples = extract_crops_for_tracklet(
            group,
            frame_loader=frame_loader,
            max_crops=int(options["max_crops_per_tracklet"]),
            padding_ratio=float(options["padding_ratio"]),
            resize=(int(options["resize_width"]), int(options["resize_height"])),
            save_crops=bool(options["save_crops"]),
            crop_output_dir=crop_base,
        )
        if not crops:
            continue
        crop_embeddings = [backend.extract(crop) for crop in crops]
        for crop_embedding, sample in zip(crop_embeddings, samples):
            crop_embedding_records.append(
                ReIDEmbeddingRecord(
                    embedding_id="crop_%s_%s_%s_%s_%06d" % (
                        sample.subset,
                        sample.scene_name,
                        sample.camera_id,
                        str(sample.local_track_id),
                        int(sample.frame_id),
                    ),
                    subset=sample.subset,
                    split=sample.split,
                    scene_name=sample.scene_name,
                    camera_id=sample.camera_id,
                    frame_id=sample.frame_id,
                    local_track_id=sample.local_track_id,
                    global_track_id=sample.global_track_id,
                    candidate_id=sample.candidate_id,
                    class_id=sample.class_id,
                    class_name=sample.class_name,
                    embedding=crop_embedding,
                    embedding_dim=int(crop_embedding.size),
                    backend=backend.name,
                    num_crops=1,
                    crop_frame_ids=[sample.frame_id],
                    mean_confidence=sample.confidence,
                    notes="crop_embedding",
                )
            )
        tracklet_embedding = compute_tracklet_embedding(crop_embeddings, samples, method=str(options["aggregation"]), backend=backend.name)
        if tracklet_embedding is not None:
            tracklet_embeddings.append(tracklet_embedding)
        candidate_id = samples[0].candidate_id if samples else None
        if candidate_id not in (None, ""):
            candidate_embedding = compute_candidate_embedding(
                crop_embeddings,
                samples,
                candidate_id=str(candidate_id),
                method=str(options["aggregation"]),
                backend=backend.name,
            )
            if candidate_embedding is not None:
                candidate_embeddings.append(candidate_embedding)

    output_root = Path(options["output_root"])
    crop_embedding_root = output_root / "crop_embeddings" / subset / scene_name
    tracklet_root = output_root / "tracklet_embeddings" / subset / scene_name
    candidate_root = output_root / "candidate_embeddings" / subset / scene_name
    crop_jsonl = crop_embedding_root / ("%s_crop_embeddings.jsonl" % camera_id)
    crop_npy = crop_embedding_root / ("%s_crop_embeddings.npy" % camera_id)
    tracklet_jsonl = tracklet_root / ("%s_tracklet_embeddings.jsonl" % camera_id)
    tracklet_npy = tracklet_root / ("%s_tracklet_embeddings.npy" % camera_id)
    candidate_jsonl = candidate_root / ("%s_candidate_embeddings.jsonl" % camera_id)
    candidate_npy = candidate_root / ("%s_candidate_embeddings.npy" % camera_id)
    write_reid_embeddings_jsonl(crop_embedding_records, crop_jsonl)
    write_reid_embeddings_npy(crop_embedding_records, crop_npy, crop_npy.with_suffix(".metadata.csv"))
    write_reid_embeddings_jsonl(tracklet_embeddings, tracklet_jsonl)
    write_reid_embeddings_npy(tracklet_embeddings, tracklet_npy, tracklet_npy.with_suffix(".metadata.csv"))
    write_reid_embeddings_jsonl(candidate_embeddings, candidate_jsonl)
    write_reid_embeddings_npy(candidate_embeddings, candidate_npy, candidate_npy.with_suffix(".metadata.csv"))
    return crop_embedding_records, tracklet_embeddings, candidate_embeddings, {
        "subset": subset,
        "scene_name": scene_name,
        "camera_id": camera_id,
        "status": "ok",
        "records": len(records),
        "crop_embeddings": len(crop_embedding_records),
        "tracklet_embeddings": len(tracklet_embeddings),
        "candidate_embeddings": len(candidate_embeddings),
    }


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data.get("reid", data)


def build_options(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge config and CLI overrides."""
    options = dict(config)
    for key in ["root", "records_root", "output_root", "backend"]:
        value = getattr(args, key)
        if value is not None:
            options[key] = str(value)
    if args.subsets is not None:
        options["subsets"] = args.subsets
    if args.scenes is not None:
        options["scenes"] = args.scenes
    if args.camera_ids is not None:
        options["camera_ids"] = args.camera_ids
    if args.max_crops_per_tracklet is not None:
        options["max_crops_per_tracklet"] = args.max_crops_per_tracklet
    if args.save_crops is not None:
        options["save_crops"] = args.save_crops
    if args.progress is not None:
        options["progress"] = args.progress
    defaults = {
        "aggregation": "mean",
        "max_crops_per_tracklet": 8,
        "padding_ratio": 0.10,
        "resize_width": 128,
        "resize_height": 256,
        "save_crops": False,
        "max_tracks_per_camera": None,
        "progress": True,
        "backend_config": {},
    }
    for key, value in defaults.items():
        options.setdefault(key, value)
    return options


def backend_config(options: Dict[str, Any]) -> Dict[str, Any]:
    """Build backend config."""
    cfg = dict(options.get("backend_config", {}))
    cfg["backend"] = options.get("backend", "color_histogram")
    cfg["resize"] = (int(options.get("resize_width", 128)), int(options.get("resize_height", 256)))
    return cfg


def find_record_files(records_root: Path, options: Dict[str, Any]) -> List[Tuple[str, str, str, Path]]:
    """Find frame_global_records CSV files."""
    subsets = _filter_set(options.get("subsets"))
    scenes = _filter_set(options.get("scenes"))
    cameras = _filter_set(options.get("camera_ids"))
    output = []
    for subset_dir in sorted(records_root.iterdir()) if records_root.exists() else []:
        if not subset_dir.is_dir() or (subsets is not None and subset_dir.name not in subsets):
            continue
        for scene_dir in sorted(subset_dir.iterdir()):
            if not scene_dir.is_dir() or (scenes is not None and scene_dir.name not in scenes):
                continue
            for path in sorted(scene_dir.glob("*_global_records.csv")):
                camera_id = path.name.replace("_global_records.csv", "")
                if cameras is not None and camera_id not in cameras:
                    continue
                output.append((subset_dir.name, scene_dir.name, camera_id, path))
    return output


def read_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(dict(row))
    return records


def group_records(records: List[Dict[str, Any]], key: str, max_groups: Any) -> Dict[str, List[Dict[str, Any]]]:
    groups = {}
    for record in records:
        value = record.get(key)
        if value in (None, ""):
            continue
        groups.setdefault(str(value), []).append(record)
    ordered_keys = sorted(groups.keys(), key=lambda item: int(float(item)) if str(item).replace(".", "", 1).isdigit() else str(item))
    if max_groups is not None:
        ordered_keys = ordered_keys[: int(max_groups)]
    return {key_value: groups[key_value] for key_value in ordered_keys}


def find_video_path(root: Path, split: str, scene: str, camera_id: str) -> Optional[Path]:
    scene_paths = get_scene_paths(root, split, scene)
    if scene_paths.videos_dir is None:
        return None
    for path in list_video_files(scene_paths.videos_dir):
        if path.stem == camera_id:
            return path
    return None


def subset_to_split(subset: str) -> str:
    if subset == "official_val":
        return "val"
    if subset == "test":
        return "test"
    return "train"


def per_field_rows(records: List[Any], field: str) -> List[Dict[str, Any]]:
    counts = {}
    for record in records:
        key = str(getattr(record, field, ""))
        counts[key] = counts.get(key, 0) + 1
    return [{"name": key, "count": value} for key, value in sorted(counts.items())]


def _filter_set(value: Any) -> Optional[set]:
    if value in (None, ""):
        return None
    return set([str(item) for item in value])


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
