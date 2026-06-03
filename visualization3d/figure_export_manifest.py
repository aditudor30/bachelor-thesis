"""Manifest helpers for selected MVP visualization figures."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class FigureExportRecord:
    """One exported figure record."""

    figure_name: str
    figure_type: str
    subset: str
    scene_name: str
    camera_id: Optional[str]
    frame_id: Optional[int]
    input_path: str
    output_path: str
    score: Optional[float]
    caption_suggestion: str
    notes: str


FIGURE_MANIFEST_FIELDS = [
    "figure_name",
    "figure_type",
    "subset",
    "scene_name",
    "camera_id",
    "frame_id",
    "input_path",
    "output_path",
    "score",
    "caption_suggestion",
    "notes",
]


def write_figure_manifest(records: List[FigureExportRecord], path: Union[str, Path]) -> None:
    """Write a figure manifest as CSV or JSON based on the file suffix."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        data = [asdict(record) for record in records]
        output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIGURE_MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_figure_manifest(path: Union[str, Path]) -> List[FigureExportRecord]:
    """Read a figure manifest from CSV or JSON."""
    input_path = Path(path)
    if input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        return [FigureExportRecord(**item) for item in data]
    records = []
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                FigureExportRecord(
                    figure_name=str(row.get("figure_name", "")),
                    figure_type=str(row.get("figure_type", "")),
                    subset=str(row.get("subset", "")),
                    scene_name=str(row.get("scene_name", "")),
                    camera_id=_optional_str(row.get("camera_id")),
                    frame_id=_optional_int(row.get("frame_id")),
                    input_path=str(row.get("input_path", "")),
                    output_path=str(row.get("output_path", "")),
                    score=_optional_float(row.get("score")),
                    caption_suggestion=str(row.get("caption_suggestion", "")),
                    notes=str(row.get("notes", "")),
                )
            )
    return records


def summarize_figure_manifest(records: List[FigureExportRecord]) -> Dict[str, Any]:
    """Summarize selected/exported figures."""
    per_type = {}
    per_scene = {}
    for record in records:
        per_type[record.figure_type] = per_type.get(record.figure_type, 0) + 1
        per_scene[record.scene_name] = per_scene.get(record.scene_name, 0) + 1
    return {
        "num_figures": len(records),
        "per_type": per_type,
        "per_scene": per_scene,
        "outputs": [record.output_path for record in records],
    }


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

