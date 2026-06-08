"""Create RGB-depth-GT-calibration alignment figures for documentation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from deep_oc_sort_3d.data.ground_truth import GroundTruthObject
from deep_oc_sort_3d.data.sample_dataset import SmartSpacesFrameDataset
from deep_oc_sort_3d.final_freeze.freeze_io import load_yaml, progress_iter, write_json
from deep_oc_sort_3d.geometry.depth_sampling import clip_bbox_to_image, sample_depth_robust
from deep_oc_sort_3d.training.target_builder import DEFAULT_CLASS_MAPPING


DEFAULT_CAPTION = (
    "Vizualizare calitativa pentru validarea alinierii RGB-Depth-GT-Calibration. "
    "Panoul din stanga prezinta frame-ul RGB cu bbox-uri ground truth si punctele "
    "de sampling utilizate pentru estimarea profunzimii, iar panoul din dreapta "
    "afiseaza harta de profunzime corespunzatoare, cu aceleasi bbox-uri si puncte "
    "suprapuse. Suprapunerea consistenta a adnotarilor pe cele doua reprezentari "
    "confirma alinierea spatiala intre imaginea RGB, depth map si adnotarile 2D."
)


@dataclass
class AlignmentDisplayObject:
    """Selected GT object and drawing metadata for the alignment figure."""

    class_name: str
    class_id: int
    object_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    center_point: Tuple[float, float]
    bottom_center_point: Tuple[float, float]
    sampling_patch_xyxy: Tuple[float, float, float, float]
    depth_center: Optional[float]
    depth_bottom_center: Optional[float]
    depth_center_median: Optional[float]
    depth_valid: bool
    area: float


def load_frame_sample(
    dataset_root: Union[str, Path],
    split: str,
    scene_name: str,
    camera_id: str,
    frame_id: int,
    depth_dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one RGB/depth/GT sample through the existing lazy frame dataset."""
    if split == "test":
        raise ValueError("RGB-depth-GT alignment figure requires train or val split, not test.")
    dataset = SmartSpacesFrameDataset(
        root=dataset_root,
        split=split,
        scene_name=scene_name,
        max_frames=frame_id + 1,
        camera_id=camera_id,
        load_rgb=True,
        load_depth=True,
        load_gt=True,
        depth_dataset_name=depth_dataset_name,
    )
    sample = dataset[frame_id]
    _attach_dataset_metadata(sample, dataset)
    return sample


def load_rgb_frame(sample: Dict[str, Any]) -> np.ndarray:
    """Return RGB frame from a loaded sample or raise a clear error."""
    rgb = sample.get("rgb")
    if rgb is None:
        raise ValueError("RGB frame is missing for selected sample.")
    return np.asarray(rgb)


def load_depth_frame(sample: Dict[str, Any]) -> np.ndarray:
    """Return depth frame from a loaded sample or raise a clear error."""
    depth = sample.get("depth")
    if depth is None:
        raise ValueError("Depth frame is missing for selected sample.")
    return np.asarray(depth)


def load_gt_for_frame(sample: Dict[str, Any]) -> List[GroundTruthObject]:
    """Return GT objects from a loaded sample or raise a clear error."""
    gt_objects = sample.get("gt_objects")
    if gt_objects is None:
        raise ValueError("Ground truth is missing for selected sample.")
    return list(gt_objects)


def compute_sampling_points(
    bbox_xyxy: Tuple[float, float, float, float],
    patch_radius: int = 3,
) -> Dict[str, Tuple[float, float]]:
    """Compute center, bottom-center, and center patch coordinates for a bbox."""
    x1, y1, x2, y2 = bbox_xyxy
    left = float(min(x1, x2))
    right = float(max(x1, x2))
    top = float(min(y1, y2))
    bottom = float(max(y1, y2))
    width = max(right - left, 0.0)
    height = max(bottom - top, 0.0)
    center_x = left + width * 0.5
    center_y = top + height * 0.5
    bottom_center_y = bottom - 0.05 * height
    radius = float(max(int(patch_radius), 0))
    return {
        "center": (center_x, center_y),
        "bottom_center": (center_x, bottom_center_y),
        "patch_xyxy": (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
    }


def clip_bbox_float(
    bbox_xyxy: Tuple[float, float, float, float],
    image_shape: Tuple[int, ...],
) -> Optional[Tuple[float, float, float, float]]:
    """Clip a bbox to image bounds and return float xyxy coordinates."""
    clipped = clip_bbox_to_image(bbox_xyxy, image_shape)
    if clipped is None:
        return None
    left, top, right, bottom = clipped
    return (float(left), float(top), float(right), float(bottom))


def normalize_depth_for_display(
    depth: np.ndarray,
    percentile_min: float = 2,
    percentile_max: float = 98,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize depth to 0..1 with percentile clipping and invalid depth as 0."""
    depth_float = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_float) & (depth_float > 0)
    invalid_count = int(depth_float.size - int(valid.sum()))
    if not valid.any():
        normalized = np.full(depth_float.shape, np.nan, dtype=np.float32)
        return normalized, {
            "percentile_min_value": None,
            "percentile_max_value": None,
            "invalid_depth_count": invalid_count,
            "valid_depth_count": 0,
        }
    valid_values = depth_float[valid]
    low = float(np.percentile(valid_values, float(percentile_min)))
    high = float(np.percentile(valid_values, float(percentile_max)))
    if high <= low:
        normalized = np.zeros(depth_float.shape, dtype=np.float32)
    else:
        normalized = (depth_float - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    normalized[~valid] = np.nan
    return normalized, {
        "percentile_min_value": low,
        "percentile_max_value": high,
        "invalid_depth_count": invalid_count,
        "valid_depth_count": int(valid.sum()),
    }


def select_display_objects(
    gt_objects: List[GroundTruthObject],
    camera_id: str,
    depth: Optional[np.ndarray],
    max_objects: int = 5,
    min_bbox_area: float = 800.0,
    prefer_classes: Optional[List[str]] = None,
    sampling_patch_radius: int = 3,
) -> List[AlignmentDisplayObject]:
    """Select a small set of visible, large, depth-valid objects for display."""
    prefer = prefer_classes or []
    candidates: List[AlignmentDisplayObject] = []
    image_shape = depth.shape if depth is not None else (1000000, 1000000)
    for obj in gt_objects:
        bbox = obj.visible_bboxes_2d.get(camera_id)
        if bbox is None:
            continue
        clipped = clip_bbox_float(tuple(float(value) for value in bbox), image_shape)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < float(min_bbox_area):
            continue
        points = compute_sampling_points(clipped, patch_radius=sampling_patch_radius)
        depth_center = None
        depth_bottom = None
        depth_center_median = None
        if depth is not None:
            depth_center = sample_depth_robust(depth, clipped, method="center")
            depth_bottom = sample_depth_robust(depth, clipped, method="bottom_center")
            depth_center_median = sample_depth_robust(depth, clipped, method="center_median")
        candidates.append(
            AlignmentDisplayObject(
                class_name=str(obj.object_type),
                class_id=int(DEFAULT_CLASS_MAPPING.get(str(obj.object_type), -1)),
                object_id=int(obj.object_id),
                bbox_xyxy=clipped,
                center_point=points["center"],
                bottom_center_point=points["bottom_center"],
                sampling_patch_xyxy=(points["patch_xyxy"][0], points["patch_xyxy"][1], points["patch_xyxy"][2], points["patch_xyxy"][3]),
                depth_center=depth_center,
                depth_bottom_center=depth_bottom,
                depth_center_median=depth_center_median,
                depth_valid=depth_center_median is not None,
                area=area,
            )
        )
    candidates.sort(key=lambda item: _selection_score(item, prefer), reverse=True)
    return candidates[: int(max_objects)]


def draw_rgb_panel(ax: Any, rgb: np.ndarray, objects: List[AlignmentDisplayObject], config: Dict[str, Any]) -> None:
    """Draw RGB panel with selected GT boxes and sampling points."""
    ax.imshow(rgb)
    ax.set_title("RGB frame with GT boxes and depth sampling points", fontsize=10)
    ax.axis("off")
    _draw_overlays(ax, objects, config)


def draw_depth_panel(ax: Any, depth: np.ndarray, objects: List[AlignmentDisplayObject], config: Dict[str, Any]) -> Dict[str, Any]:
    """Draw depth panel with selected GT boxes and sampling points."""
    depth_vis, stats = normalize_depth_for_display(
        depth,
        percentile_min=float(config.get("depth_percentile_min", 2)),
        percentile_max=float(config.get("depth_percentile_max", 98)),
    )
    cmap = _depth_colormap()
    ax.imshow(depth_vis, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("Aligned depth map with the same boxes and points", fontsize=10)
    ax.axis("off")
    _draw_overlays(ax, objects, config)
    return stats


def create_alignment_panel_figure(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    progress: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create the RGB-depth alignment figure from a YAML config."""
    config = load_yaml(Path(config_path))
    if overrides:
        config = _apply_overrides(config, overrides)
    figure_config = dict(config.get("figure_rgb_depth_gt_alignment", {}))
    paths = config.get("paths", {})
    sample_config = dict(config.get("sample", {}))
    auto_select = dict(config.get("auto_select", {}))
    dataset_root = Path(str(paths.get("dataset_root", "")))
    split = str(sample_config.get("split", "val"))
    scene_name = str(sample_config.get("scene_name", "Warehouse_020"))
    camera_id = str(sample_config.get("camera_id", "Camera_0000"))
    frame_id = int(sample_config.get("frame_id", 100))

    if split == "test":
        raise ValueError("This figure requires train/val because it overlays GT and depth.")

    sample, objects = _select_sample_and_objects(
        dataset_root=dataset_root,
        split=split,
        scene_name=scene_name,
        camera_id=camera_id,
        frame_id=frame_id,
        figure_config=figure_config,
        auto_select=auto_select,
        progress=progress,
    )
    output_dir = Path(str(figure_config.get("output_dir", "figures/alignment")))
    output_name = str(figure_config.get("output_name", "rgb_depth_gt_calibration_alignment_panel"))
    return create_alignment_panel_from_sample(
        sample=sample,
        objects=objects,
        figure_config=figure_config,
        output_dir=output_dir,
        output_name=output_name,
        overwrite=overwrite,
    )


def create_alignment_panel_from_sample(
    sample: Dict[str, Any],
    objects: List[AlignmentDisplayObject],
    figure_config: Dict[str, Any],
    output_dir: Path,
    output_name: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create panel figure and metadata from an already-loaded sample."""
    rgb = load_rgb_frame(sample)
    depth = load_depth_frame(sample)
    metadata_context = _sample_metadata_context(sample)
    return create_alignment_panel_from_arrays(
        rgb=rgb,
        depth=depth,
        objects=objects,
        figure_config=figure_config,
        output_dir=output_dir,
        output_name=output_name,
        metadata_context=metadata_context,
        overwrite=overwrite,
    )


def create_alignment_panel_from_arrays(
    rgb: np.ndarray,
    depth: np.ndarray,
    objects: List[AlignmentDisplayObject],
    figure_config: Dict[str, Any],
    output_dir: Path,
    output_name: str,
    metadata_context: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create panel figure from arrays and selected display objects."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if tuple(rgb.shape[:2]) != tuple(depth.shape[:2]):
        raise ValueError("RGB and depth frames must have the same height/width for alignment visualization.")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / ("%s.png" % output_name)
    pdf_path = output_dir / ("%s.pdf" % output_name)
    metadata_path = output_dir / ("%s_metadata.json" % output_name)
    caption_path = output_dir / ("%s_caption.txt" % output_name)
    _ensure_can_write([png_path, pdf_path, metadata_path, caption_path], overwrite=overwrite)

    dpi = int(figure_config.get("dpi", 300))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    draw_rgb_panel(axes[0], rgb, objects, figure_config)
    depth_stats = draw_depth_panel(axes[1], depth, objects, figure_config)
    _draw_legend(axes[1])
    fig.tight_layout(pad=0.3)
    fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(pdf_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    metadata = write_metadata(
        metadata_path=metadata_path,
        metadata_context=metadata_context or {},
        objects=objects,
        depth_stats=depth_stats,
        output_png=png_path,
        output_pdf=pdf_path,
    )
    caption_path.write_text(str(figure_config.get("caption", DEFAULT_CAPTION)) + "\n", encoding="utf-8")
    metadata["caption_path"] = str(caption_path)
    write_json(metadata, metadata_path)
    return metadata


def write_metadata(
    metadata_path: Path,
    metadata_context: Dict[str, Any],
    objects: List[AlignmentDisplayObject],
    depth_stats: Dict[str, Any],
    output_png: Path,
    output_pdf: Path,
) -> Dict[str, Any]:
    """Write figure metadata JSON."""
    metadata = dict(metadata_context)
    metadata.update(
        {
            "num_objects_selected": len(objects),
            "selected_objects": [display_object_to_dict(obj) for obj in objects],
            "depth_visualization": depth_stats,
            "output_png": str(output_png),
            "output_pdf": str(output_pdf),
            "output_metadata": str(metadata_path),
        }
    )
    write_json(metadata, metadata_path)
    return metadata


def display_object_to_dict(obj: AlignmentDisplayObject) -> Dict[str, Any]:
    """Convert display object to JSON-serializable metadata."""
    return {
        "class_name": obj.class_name,
        "class_id": obj.class_id,
        "object_id": obj.object_id,
        "bbox": [float(value) for value in obj.bbox_xyxy],
        "center_point": [float(value) for value in obj.center_point],
        "bottom_center_point": [float(value) for value in obj.bottom_center_point],
        "sampling_patch_xyxy": [float(value) for value in obj.sampling_patch_xyxy],
        "depth_center": _optional_float(obj.depth_center),
        "depth_bottom_center": _optional_float(obj.depth_bottom_center),
        "depth_center_median": _optional_float(obj.depth_center_median),
        "depth_valid": bool(obj.depth_valid),
        "bbox_area": float(obj.area),
    }


def _select_sample_and_objects(
    dataset_root: Path,
    split: str,
    scene_name: str,
    camera_id: str,
    frame_id: int,
    figure_config: Dict[str, Any],
    auto_select: Dict[str, Any],
    progress: bool,
) -> Tuple[Dict[str, Any], List[AlignmentDisplayObject]]:
    if bool(auto_select.get("enabled", False)):
        frame_start = int(auto_select.get("frame_start", 0))
        frame_end = int(auto_select.get("frame_end", frame_id))
        min_valid_objects = int(auto_select.get("min_valid_objects", 3))
        dataset = SmartSpacesFrameDataset(
            root=dataset_root,
            split=split,
            scene_name=scene_name,
            max_frames=frame_end + 1,
            camera_id=camera_id,
            load_rgb=True,
            load_depth=True,
            load_gt=True,
        )
        last_frame = min(frame_end, max(len(dataset) - 1, 0))
        for current_frame in progress_iter(range(frame_start, last_frame + 1), progress, "auto-select alignment frame", "frame"):
            sample = dataset[current_frame]
            _attach_dataset_metadata(sample, dataset)
            objects = _objects_for_sample(sample, figure_config, auto_select)
            valid_count = len([obj for obj in objects if obj.depth_valid])
            if len(objects) >= min_valid_objects and valid_count >= min_valid_objects:
                return sample, objects
        raise ValueError("Auto-select did not find a frame with enough valid visible objects.")

    sample = load_frame_sample(dataset_root, split, scene_name, camera_id, frame_id)
    objects = _objects_for_sample(sample, figure_config, auto_select)
    if not objects:
        raise ValueError("Selected frame has no displayable GT objects. Enable auto_select or choose another frame.")
    return sample, objects


def _objects_for_sample(sample: Dict[str, Any], figure_config: Dict[str, Any], auto_select: Dict[str, Any]) -> List[AlignmentDisplayObject]:
    gt_objects = load_gt_for_frame(sample)
    depth = load_depth_frame(sample)
    prefer_classes = auto_select.get("prefer_classes")
    if prefer_classes is not None:
        prefer_classes = [str(item) for item in prefer_classes]
    return select_display_objects(
        gt_objects=gt_objects,
        camera_id=str(sample.get("camera_id", "")),
        depth=depth,
        max_objects=int(figure_config.get("max_objects", 5)),
        min_bbox_area=float(figure_config.get("min_bbox_area", 800)),
        prefer_classes=prefer_classes,
        sampling_patch_radius=int(figure_config.get("sampling_patch_radius", 3)),
    )


def _draw_overlays(ax: Any, objects: List[AlignmentDisplayObject], config: Dict[str, Any]) -> None:
    from matplotlib.patches import Rectangle

    colors = ["#f97316", "#22c55e", "#38bdf8", "#e879f9", "#facc15", "#a78bfa"]
    show_labels = bool(config.get("show_labels", True))
    show_center = bool(config.get("show_center_point", True))
    show_bottom = bool(config.get("show_bottom_center_point", True))
    show_patch = bool(config.get("show_sampling_patch", True))
    for index, obj in enumerate(objects):
        color = colors[index % len(colors)]
        x1, y1, x2, y2 = obj.bbox_xyxy
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.4))
        if show_patch:
            px1, py1, px2, py2 = obj.sampling_patch_xyxy
            ax.add_patch(Rectangle((px1, py1), px2 - px1, py2 - py1, fill=False, edgecolor="#ffffff", linewidth=0.9, linestyle="--"))
        if show_center:
            ax.scatter([obj.center_point[0]], [obj.center_point[1]], s=28, c="#fde047", marker="o", edgecolors="#111111", linewidths=0.5)
        if show_bottom:
            ax.scatter([obj.bottom_center_point[0]], [obj.bottom_center_point[1]], s=34, c="#ec4899", marker="x", linewidths=1.4)
        if show_labels:
            label = "%s %d" % (obj.class_name, obj.object_id)
            ax.text(
                x1,
                max(0.0, y1 - 4.0),
                label,
                color="white",
                fontsize=7,
                bbox={"facecolor": color, "alpha": 0.75, "pad": 1.5, "edgecolor": "none"},
            )


def _draw_legend(ax: Any) -> None:
    ax.scatter([], [], s=28, c="#fde047", marker="o", edgecolors="#111111", linewidths=0.5, label="center")
    ax.scatter([], [], s=34, c="#ec4899", marker="x", linewidths=1.4, label="bottom-center")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.75)


def _depth_colormap() -> Any:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="black")
    return cmap


def _selection_score(obj: AlignmentDisplayObject, prefer_classes: List[str]) -> Tuple[int, int, float]:
    preferred = 1 if obj.class_name in prefer_classes else 0
    valid = 1 if obj.depth_valid else 0
    return (valid, preferred, obj.area)


def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(config)
    sample = dict(output.get("sample", {}))
    figure = dict(output.get("figure_rgb_depth_gt_alignment", {}))
    auto_select = dict(output.get("auto_select", {}))
    paths = dict(output.get("paths", {}))
    if overrides.get("dataset_root") is not None:
        paths["dataset_root"] = overrides["dataset_root"]
    for key in ["split", "scene_name", "camera_id", "frame_id"]:
        if key in overrides and overrides[key] is not None:
            sample[key] = overrides[key]
    for key in ["output_dir", "max_objects"]:
        if key in overrides and overrides[key] is not None:
            figure[key] = overrides[key]
    if overrides.get("no_labels") is True:
        figure["show_labels"] = False
    if overrides.get("auto_select") is not None:
        auto_select["enabled"] = bool(overrides.get("auto_select"))
    output["sample"] = sample
    output["figure_rgb_depth_gt_alignment"] = figure
    output["auto_select"] = auto_select
    output["paths"] = paths
    return output


def _sample_metadata_context(sample: Dict[str, Any]) -> Dict[str, Any]:
    calibration = sample.get("calibration")
    return {
        "dataset_root": str(sample.get("dataset_root", "")),
        "split": sample.get("split"),
        "scene_name": sample.get("scene_name"),
        "scene_id": sample.get("scene_id"),
        "camera_id": sample.get("camera_id"),
        "frame_id": sample.get("frame_id"),
        "rgb_path": _path_to_string(sample.get("rgb_path")),
        "depth_path": _path_to_string(sample.get("depth_path")),
        "map_path": _path_to_string(sample.get("map_path")),
        "calibration_available": calibration is not None,
        "calibration_path": _path_to_string(_scene_path_attr(sample, "calibration_path")),
        "ground_truth_path": _path_to_string(_scene_path_attr(sample, "ground_truth_path")),
    }


def _attach_dataset_metadata(sample: Dict[str, Any], dataset: SmartSpacesFrameDataset) -> None:
    sample["dataset_root"] = str(dataset.root)
    sample["scene_paths"] = dataset.scene_paths


def _scene_path_attr(sample: Dict[str, Any], name: str) -> Optional[Path]:
    scene_paths = sample.get("scene_paths")
    if scene_paths is not None and hasattr(scene_paths, name):
        return getattr(scene_paths, name)
    return None


def _path_to_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _ensure_can_write(paths: Iterable[Path], overwrite: bool) -> None:
    if overwrite:
        return
    for path in paths:
        if path.exists():
            raise FileExistsError("Output exists; pass --overwrite to replace: %s" % path)
