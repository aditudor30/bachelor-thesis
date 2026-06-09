"""Extract fine-tuned OSNet embeddings for V2 fullcam Person fragments."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deep_oc_sort_3d.data.dataset_structure import scene_name_to_id
from deep_oc_sort_3d.data.frame_io import list_video_files, safe_read_video_frame
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import (
    bool_text,
    frame_record_csv_files,
    optional_list,
    output_root_from_config,
    progress_iter,
    read_csv_rows,
    row_scene_id,
    safe_float,
    safe_int,
    write_csv_rows,
    write_json,
)
from deep_oc_sort_3d.reid_finetuned_association.fragment_embedding_builder import (
    aggregate_crop_embeddings_to_fragments,
    l2_normalize_vector,
    write_crop_embedding_jsonl,
    write_fragment_embedding_outputs,
)


CROP_METADATA_FIELDS = [
    "crop_embedding_id",
    "subset",
    "split",
    "scene_name",
    "scene_id",
    "camera_id",
    "frame_id",
    "class_id",
    "class_name",
    "object_id",
    "local_track_id",
    "tracklet_id",
    "candidate_id",
    "global_track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
    "crop_path_optional",
    "embedding_index",
    "valid_embedding",
    "invalid_reason",
    "matched_gt_object_id",
    "source_csv",
]


def extract_finetuned_person_embeddings_from_config(config: Dict[str, Any], show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Extract fine-tuned crop and fragment embeddings for Person records."""
    _unused_overwrite = overwrite
    output_root = output_root_from_config(config)
    embeddings_dir = output_root / "embeddings"
    summary_path = embeddings_dir / "embedding_extraction_summary.json"
    paths = config.get("paths", {})
    frame_root = Path(str(paths.get("v2_frame_global_records_root", "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam/frame_global_records")))
    dataset_root = Path(str(paths.get("dataset_root", "/path/to/MTMC_Tracking_2026")))
    if not dataset_root.exists():
        raise FileNotFoundError("dataset_root missing or invalid for fine-tuned ReID crop extraction: %s" % dataset_root)
    extraction_cfg = config.get("embedding_extraction", {})
    files = frame_record_csv_files(
        frame_root,
        subsets=optional_list(extraction_cfg.get("subsets")),
        scenes=optional_list(extraction_cfg.get("scenes")),
    )
    model, device, model_summary = build_finetuned_osnet_model(config)
    crop_rows: List[Dict[str, Any]] = []
    crop_embeddings: List[np.ndarray] = []
    file_summaries: List[Dict[str, Any]] = []
    for csv_path in progress_iter(files, show_progress, "fine-tuned ReID crop extraction", "file"):
        file_summary, rows, embeddings = extract_embeddings_from_frame_file(
            csv_path,
            dataset_root,
            model,
            device,
            config,
            output_root,
            show_progress=False,
        )
        file_summaries.append(file_summary)
        crop_rows.extend(rows)
        crop_embeddings.extend(embeddings)
    matrix = np.vstack([embedding.reshape(1, -1) for embedding in crop_embeddings]).astype(np.float32) if crop_embeddings else np.zeros((0, 0), dtype=np.float32)
    crop_metadata_path = embeddings_dir / "crop_embeddings_metadata.csv"
    write_csv_rows(crop_rows, crop_metadata_path, CROP_METADATA_FIELDS)
    crop_npy_path = embeddings_dir / "finetuned_crop_embeddings.npy"
    np.save(str(crop_npy_path), matrix.astype(np.float32))
    write_crop_embedding_jsonl(crop_rows, matrix, embeddings_dir / "crops" / "finetuned_crop_embeddings.jsonl")
    fragment_matrix, fragment_rows, fragment_jsonl_records = aggregate_crop_embeddings_to_fragments(
        matrix,
        crop_rows,
        backend="torchreid_osnet_finetuned",
    )
    fragment_summary = write_fragment_embedding_outputs(fragment_matrix, fragment_rows, fragment_jsonl_records, output_root)
    summary = summarize_embedding_extraction(file_summaries, crop_rows, matrix, fragment_summary, model_summary)
    summary["crop_embeddings"] = str(crop_npy_path)
    summary["crop_metadata"] = str(crop_metadata_path)
    write_json(summary, summary_path)
    write_json(embedding_coverage_summary(crop_rows, fragment_rows, file_summaries), output_root / "diagnostics" / "embedding_coverage_summary.json")
    return summary


def extract_embeddings_from_frame_file(
    csv_path: Path,
    dataset_root: Path,
    model: Any,
    device: str,
    config: Dict[str, Any],
    output_root: Path,
    show_progress: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[np.ndarray]]:
    """Extract sampled Person crop embeddings from one frame_global_records CSV."""
    rows, _fields = read_csv_rows(csv_path)
    extraction_cfg = config.get("embedding_extraction", {})
    class_id = int(extraction_cfg.get("class_id", 0))
    person_rows = [row for row in rows if safe_int(row.get("class_id"), -1) == class_id]
    selected = sample_rows_per_fragment(
        person_rows,
        max_crops=int(extraction_cfg.get("max_crops_per_fragment", 20)),
        strategy=str(extraction_cfg.get("sample_strategy", "uniform_temporal")),
    )
    subset = _first_value(rows, "subset")
    scene_name = _first_value(rows, "scene_name")
    split = infer_dataset_split(_first_value(rows, "split"), scene_name)
    camera_id = _first_value(rows, "camera_id")
    video_path = find_video_path(dataset_root, split, scene_name, camera_id)
    frame_cache: Dict[int, Optional[np.ndarray]] = {}
    batch_crops: List[np.ndarray] = []
    batch_rows: List[Dict[str, Any]] = []
    output_rows: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    missing_frame_count = 0
    invalid_bbox_count = 0
    invalid_crop_count = 0
    for row in progress_iter(selected, show_progress, "extract fine-tuned crops", "crop"):
        frame_id = safe_int(row.get("frame_id"), -1) or -1
        bbox = bbox_from_row(row)
        if bbox is None or not bbox_passes_size(bbox, extraction_cfg):
            invalid_bbox_count += 1
            continue
        if video_path is None:
            missing_frame_count += 1
            continue
        if frame_id not in frame_cache:
            frame_cache[frame_id] = safe_read_video_frame(video_path, frame_id)
        image = frame_cache[frame_id]
        if image is None:
            missing_frame_count += 1
            continue
        crop = crop_image_xyxy(image, bbox, padding_ratio=float(extraction_cfg.get("padding_ratio", 0.10)))
        if crop is None:
            invalid_crop_count += 1
            continue
        meta = crop_metadata_row(row, csv_path, bbox, len(output_rows), output_root, crop if bool(extraction_cfg.get("save_debug_crops", False)) else None)
        batch_crops.append(crop)
        batch_rows.append(meta)
        if len(batch_crops) >= int(extraction_cfg.get("batch_size", 128)):
            batch_embeddings = embed_crop_batch(model, device, batch_crops, config)
            output_rows.extend(batch_rows)
            embeddings.extend([batch_embeddings[index] for index in range(batch_embeddings.shape[0])])
            batch_crops = []
            batch_rows = []
    if batch_crops:
        batch_embeddings = embed_crop_batch(model, device, batch_crops, config)
        output_rows.extend(batch_rows)
        embeddings.extend([batch_embeddings[index] for index in range(batch_embeddings.shape[0])])
    for index, row in enumerate(output_rows):
        row["embedding_index"] = index
        row["valid_embedding"] = "1"
        row["invalid_reason"] = ""
    summary = {
        "source_csv": str(csv_path),
        "subset": subset,
        "split": split,
        "scene_name": scene_name,
        "camera_id": camera_id,
        "person_records": len(person_rows),
        "sampled_records": len(selected),
        "embedding_records": len(output_rows),
        "missing_frame_count": missing_frame_count,
        "invalid_bbox_count": invalid_bbox_count,
        "invalid_crop_count": invalid_crop_count,
        "video_path": "" if video_path is None else str(video_path),
    }
    return summary, output_rows, embeddings


def build_finetuned_osnet_model(config: Dict[str, Any]) -> Tuple[Any, str, Dict[str, Any]]:
    """Build OSNet and load the fine-tuned checkpoint from disk."""
    import torch
    import torchreid

    paths = config.get("paths", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("embedding_extraction", {})
    checkpoint_path = Path(str(paths.get("finetuned_checkpoint", "")))
    if not checkpoint_path.exists():
        raise FileNotFoundError("fine-tuned ReID checkpoint missing: %s" % checkpoint_path)
    device = resolve_device(str(training_cfg.get("device", "cuda")))
    num_classes = infer_checkpoint_num_classes(checkpoint_path)
    model = torchreid.models.build_model(
        name=str(model_cfg.get("architecture", "osnet_x1_0")),
        num_classes=max(1, int(num_classes)),
        pretrained=False,
        loss="triplet",
    )
    state = torch.load(str(checkpoint_path), map_location="cpu")
    payload = state.get("model_state_dict", state) if isinstance(state, dict) else state
    compatible = filter_compatible_state_dict(model, strip_module_prefix(payload))
    model.load_state_dict(compatible, strict=False)
    model.to(device)
    model.eval()
    return model, device, {"checkpoint": str(checkpoint_path), "device": device, "num_classes": int(num_classes), "compatible_keys": len(compatible)}


def embed_crop_batch(model: Any, device: str, crops: List[np.ndarray], config: Dict[str, Any]) -> np.ndarray:
    """Embed a crop batch with OSNet."""
    import torch

    if not crops:
        return np.zeros((0, int(config.get("model", {}).get("embedding_dim", 512))), dtype=np.float32)
    model_cfg = config.get("model", {})
    height = int(model_cfg.get("input_height", 256))
    width = int(model_cfg.get("input_width", 128))
    tensor = np.stack([preprocess_crop(crop, width=width, height=height) for crop in crops], axis=0)
    images = torch.from_numpy(tensor).float().to(device)
    use_amp = bool(config.get("embedding_extraction", {}).get("use_amp", True)) and str(device).startswith("cuda")
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
        else:
            output = model(images)
    if isinstance(output, tuple) or isinstance(output, list):
        output = output[-1]
    features = output.detach().cpu().numpy().reshape(output.shape[0], -1)
    return np.vstack([l2_normalize_vector(row).reshape(1, -1) for row in features]).astype(np.float32)


def preprocess_crop(crop: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize and normalize RGB crop for OSNet."""
    image = cv2.resize(np.asarray(crop), (int(width), int(height)), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return image.transpose(2, 0, 1)


def sample_rows_per_fragment(rows: List[Dict[str, Any]], max_crops: int, strategy: str) -> List[Dict[str, Any]]:
    """Uniformly sample rows per global fragment."""
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("subset", "")), str(row.get("scene_name", "")), str(row.get("camera_id", "")), str(row.get("global_track_id", "")))
        groups.setdefault(key, []).append(row)
    selected: List[Dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda item: str(item)):
        ordered = sorted(groups[key], key=lambda item: safe_int(item.get("frame_id"), -1) or -1)
        if len(ordered) <= int(max_crops):
            selected.extend(ordered)
        elif str(strategy) == "highest_confidence":
            top = sorted(ordered, key=lambda item: safe_float(item.get("confidence"), 0.0) or 0.0, reverse=True)[: int(max_crops)]
            selected.extend(sorted(top, key=lambda item: safe_int(item.get("frame_id"), -1) or -1))
        else:
            indices = uniform_indices(len(ordered), int(max_crops))
            selected.extend([ordered[index] for index in indices])
    return selected


def crop_metadata_row(row: Dict[str, Any], source_csv: Path, bbox: Tuple[float, float, float, float], embedding_index: int, output_root: Path, crop: Optional[np.ndarray]) -> Dict[str, Any]:
    """Create crop metadata row and optionally save debug crop."""
    crop_id = "%s__%s__%s__g%s__f%06d" % (
        str(row.get("subset", "")),
        str(row.get("scene_name", "")),
        str(row.get("camera_id", "")),
        str(row.get("global_track_id", "")),
        safe_int(row.get("frame_id"), -1) or -1,
    )
    crop_path = ""
    if crop is not None:
        crop_path = str(output_root / "embeddings" / "crops" / str(row.get("subset", "")) / str(row.get("scene_name", "")) / ("%s.jpg" % crop_id))
        Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return {
        "crop_embedding_id": crop_id,
        "subset": str(row.get("subset", "")),
        "split": str(row.get("split", "")),
        "scene_name": str(row.get("scene_name", "")),
        "scene_id": row_scene_id(row),
        "camera_id": str(row.get("camera_id", "")),
        "frame_id": safe_int(row.get("frame_id"), -1) or -1,
        "class_id": safe_int(row.get("class_id"), 0) or 0,
        "class_name": str(row.get("class_name", "Person")),
        "object_id": str(row.get("object_id", "")),
        "local_track_id": str(row.get("local_track_id", "")),
        "tracklet_id": str(row.get("tracklet_id", "")),
        "candidate_id": str(row.get("candidate_id", "")),
        "global_track_id": str(row.get("global_track_id", "")),
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "confidence": safe_float(row.get("confidence"), 0.0) or 0.0,
        "crop_path_optional": crop_path,
        "embedding_index": embedding_index,
        "valid_embedding": "1",
        "invalid_reason": "",
        "matched_gt_object_id": str(row.get("matched_gt_object_id", "")),
        "source_csv": str(source_csv),
    }


def summarize_embedding_extraction(file_summaries: List[Dict[str, Any]], crop_rows: List[Dict[str, Any]], matrix: np.ndarray, fragment_summary: Dict[str, Any], model_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize extraction output."""
    total_person = sum([int(row.get("person_records", 0)) for row in file_summaries])
    total_sampled = sum([int(row.get("sampled_records", 0)) for row in file_summaries])
    status = "ok" if len(crop_rows) > 0 else "no_crop_embeddings"
    warnings = []
    if len(crop_rows) <= 0:
        warnings.append("No crop embeddings were extracted. Check dataset_root, real split inference, video paths, and bbox fields.")
    return {
        "status": status,
        "warnings": warnings,
        "files": len(file_summaries),
        "total_person_records": total_person,
        "sampled_crop_records": total_sampled,
        "crop_embeddings": len(crop_rows),
        "crop_success_rate": float(len(crop_rows)) / float(total_sampled) if total_sampled else None,
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] > 0 else None,
        "fragment_summary": fragment_summary,
        "model": model_summary,
        "missing_frame_count": sum([int(row.get("missing_frame_count", 0)) for row in file_summaries]),
        "invalid_bbox_count": sum([int(row.get("invalid_bbox_count", 0)) for row in file_summaries]),
        "invalid_crop_count": sum([int(row.get("invalid_crop_count", 0)) for row in file_summaries]),
    }


def embedding_coverage_summary(crop_rows: List[Dict[str, Any]], fragment_rows: List[Dict[str, Any]], file_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize embedding coverage."""
    total_person = sum([int(row.get("person_records", 0)) for row in file_summaries])
    valid_fragments = [row for row in fragment_rows if str(row.get("valid_embedding", "")) == "1"]
    warnings = []
    if not crop_rows:
        warnings.append("No fine-tuned ReID crop embeddings are available; association sweep must be treated as no-ReID/no-op.")
    return {
        "status": "ok" if crop_rows else "no_crop_embeddings",
        "warnings": warnings,
        "total_person_records": total_person,
        "crop_embeddings": len(crop_rows),
        "fragment_embeddings": len(fragment_rows),
        "valid_fragment_embeddings": len(valid_fragments),
        "fragment_embedding_coverage": float(len(valid_fragments)) / float(len(fragment_rows)) if fragment_rows else None,
        "sampled_crop_rate_vs_person_records": float(len(crop_rows)) / float(total_person) if total_person else None,
    }


def bbox_from_row(row: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Read bbox from common final-export field variants."""
    candidates = [
        ("x1", "y1", "x2", "y2"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("xmin", "ymin", "xmax", "ymax"),
    ]
    for names in candidates:
        values = [safe_float(row.get(name), None) for name in names]
        if all(value is not None for value in values):
            if values[2] > values[0] and values[3] > values[1]:
                return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    return None


def bbox_passes_size(bbox: Tuple[float, float, float, float], config: Dict[str, Any]) -> bool:
    """Check minimum crop size."""
    return (bbox[2] - bbox[0]) >= float(config.get("min_crop_width", 12)) and (bbox[3] - bbox[1]) >= float(config.get("min_crop_height", 24))


def crop_image_xyxy(image: np.ndarray, bbox: Tuple[float, float, float, float], padding_ratio: float = 0.0) -> Optional[np.ndarray]:
    """Crop an RGB image by bbox."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = float(x2) - float(x1)
    box_h = float(y2) - float(y1)
    x1 = max(0, min(int(round(float(x1) - box_w * padding_ratio)), width - 1))
    y1 = max(0, min(int(round(float(y1) - box_h * padding_ratio)), height - 1))
    x2 = max(0, min(int(round(float(x2) + box_w * padding_ratio)), width))
    y2 = max(0, min(int(round(float(y2) + box_h * padding_ratio)), height))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def find_video_path(dataset_root: Path, split: str, scene_name: str, camera_id: str) -> Optional[Path]:
    """Find a per-camera video in the dataset."""
    resolved_split = infer_dataset_split(split, scene_name)
    videos_dir = Path(dataset_root) / str(resolved_split) / str(scene_name) / "videos"
    for video_path in list_video_files(videos_dir):
        if video_path.stem == str(camera_id):
            return video_path
    return None


def infer_dataset_split(split: str, scene_name: str) -> str:
    """Map diagnostic subsets to real dataset splits using scene id."""
    text = str(split)
    if text in ("train", "val", "test"):
        return text
    scene_id = scene_name_to_id(str(scene_name))
    if scene_id is None:
        return text
    if scene_id <= 19:
        return "train"
    if scene_id <= 22:
        return "val"
    return "test"


def resolve_device(requested: str) -> str:
    """Resolve device with CPU fallback."""
    import torch

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("warning: CUDA requested for fine-tuned ReID but unavailable; using CPU")
        return "cpu"
    return str(requested)


def infer_checkpoint_num_classes(checkpoint_path: Path) -> int:
    """Infer classifier size from a fine-tuned checkpoint payload."""
    import torch

    state = torch.load(str(checkpoint_path), map_location="cpu")
    payload = state.get("model_state_dict", state) if isinstance(state, dict) else {}
    for key, value in payload.items():
        if str(key).endswith("classifier.weight"):
            return int(value.shape[0])
    return 1


def strip_module_prefix(state: Any) -> Dict[str, Any]:
    """Strip DataParallel module prefix from a state dict."""
    if not isinstance(state, dict):
        return {}
    output = {}
    for key, value in state.items():
        text = str(key)
        if text.startswith("module."):
            text = text[len("module.") :]
        output[text] = value
    return output


def filter_compatible_state_dict(model: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Drop checkpoint keys incompatible with the instantiated OSNet."""
    model_state = model.state_dict()
    output = {}
    for key, value in state.items():
        if key not in model_state:
            continue
        try:
            if tuple(value.shape) != tuple(model_state[key].shape):
                continue
        except AttributeError:
            continue
        output[key] = value
    return output


def uniform_indices(length: int, count: int) -> List[int]:
    """Return uniformly spaced integer indices."""
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    values = np.linspace(0, length - 1, int(count))
    return sorted(set([int(round(value)) for value in values]))[:count]


def _first_value(rows: List[Dict[str, Any]], key: str) -> str:
    if not rows:
        return ""
    return str(rows[0].get(key, ""))
