"""Diagnostic crop grids for the SmartSpaces Person ReID dataset."""

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from deep_oc_sort_3d.reid_training.reid_dataset_config import output_root_from_config
from deep_oc_sort_3d.reid_training.reid_dataset_io import group_by, read_csv_rows, write_json


def build_reid_dataset_figures_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build diagnostic crop grids."""
    if not bool(config.get("figures", {}).get("enabled", True)):
        return {"status": "disabled"}
    output_root = output_root_from_config(config)
    rows, _fields = read_csv_rows(output_root / "metadata" / "all_crops.csv")
    valid_rows = [row for row in rows if str(row.get("is_valid_crop", "")) == "1"]
    figure_root = output_root / "figures"
    max_images = int(config.get("figures", {}).get("max_grid_images", 64))
    thumb_size = int(config.get("figures", {}).get("thumbnail_size", 128))
    outputs = {
        "sample_crops_grid_train": figure_root / "sample_crops_grid_train.png",
        "sample_crops_grid_val": figure_root / "sample_crops_grid_val.png",
        "identity_examples_grid": figure_root / "identity_examples_grid.png",
        "hard_cases_grid": figure_root / "hard_cases_grid.png",
    }
    make_crop_grid([row for row in valid_rows if row.get("split") == "train"][:max_images], outputs["sample_crops_grid_train"], thumb_size, "Train Person crops")
    make_crop_grid([row for row in valid_rows if row.get("split") == "val"][:max_images], outputs["sample_crops_grid_val"], thumb_size, "Val Person crops")
    make_crop_grid(_identity_example_rows(valid_rows, max_images), outputs["identity_examples_grid"], thumb_size, "Identity examples")
    make_crop_grid(_hard_case_rows(valid_rows, max_images), outputs["hard_cases_grid"], thumb_size, "Small / hard crop cases")
    summary = {"status": "ok", "figures": {key: str(value) for key, value in outputs.items()}}
    write_json(summary, figure_root / "figures_summary.json")
    return summary


def make_crop_grid(rows: List[Dict[str, Any]], output_path: Path, thumbnail_size: int, title: str) -> None:
    """Create a grid image from crop rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        image = Image.new("RGB", (thumbnail_size * 4, thumbnail_size * 2), color=(245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "No crop images available for: %s" % title, fill=(20, 20, 20))
        image.save(str(output_path))
        return
    cols = min(8, max(1, int(round(len(rows) ** 0.5))))
    rows_count = int((len(rows) + cols - 1) / cols)
    label_h = 26
    canvas = Image.new("RGB", (cols * thumbnail_size, rows_count * (thumbnail_size + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        crop = _read_crop(Path(str(row.get("crop_path", ""))), thumbnail_size)
        col = index % cols
        grid_row = int(index / cols)
        x = col * thumbnail_size
        y = grid_row * (thumbnail_size + label_h)
        canvas.paste(crop, (x, y))
        label = "%s %s f%s" % (row.get("identity_id", ""), row.get("camera_id", ""), row.get("frame_id", ""))
        draw.text((x + 2, y + thumbnail_size + 2), label[:24], fill=(0, 0, 0))
    canvas.save(str(output_path))


def _identity_example_rows(rows: List[Dict[str, Any]], max_images: int) -> List[Dict[str, Any]]:
    groups = group_by(rows, "identity_id")
    selected: List[Dict[str, Any]] = []
    for _identity, values in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        ordered = sorted(values, key=lambda row: (str(row.get("camera_id", "")), int(row.get("frame_id", -1))))
        selected.extend(ordered[:4])
        if len(selected) >= max_images:
            break
    return selected[:max_images]


def _hard_case_rows(rows: List[Dict[str, Any]], max_images: int) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get("bbox_area", 0.0)))[:max_images]


def _read_crop(path: Path, thumbnail_size: int) -> Image.Image:
    try:
        image = Image.open(str(path)).convert("RGB")
    except Exception:
        image = Image.new("RGB", (thumbnail_size, thumbnail_size), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)
        draw.text((4, 4), "missing", fill=(0, 0, 0))
        return image
    image.thumbnail((thumbnail_size, thumbnail_size))
    canvas = Image.new("RGB", (thumbnail_size, thumbnail_size), color=(255, 255, 255))
    x = int((thumbnail_size - image.size[0]) / 2)
    y = int((thumbnail_size - image.size[1]) / 2)
    canvas.paste(image, (x, y))
    return canvas

