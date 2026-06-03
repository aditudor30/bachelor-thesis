"""Mapping helpers from generic MVP export columns to Track 1 columns."""

import csv
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from deep_oc_sort_3d.final_export.track1_export_types import Track1ExportSchema


GENERIC_TRACKING_COLUMNS = [
    "scene_name",
    "camera_id",
    "frame_id",
    "global_track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "w",
    "h",
    "center_x",
    "center_y",
    "center_z",
    "width_3d",
    "length_3d",
    "height_3d",
    "yaw",
]


def load_generic_tracking_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load one generic tracking CSV as dictionaries."""
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def infer_available_generic_columns(rows_or_header: Any) -> List[str]:
    """Infer available generic columns from rows, header list, or CSV DictReader metadata."""
    if rows_or_header is None:
        return []
    if isinstance(rows_or_header, list):
        if not rows_or_header:
            return []
        first = rows_or_header[0]
        if isinstance(first, dict):
            return [str(key) for key in first.keys()]
        return [str(item) for item in rows_or_header]
    fieldnames = getattr(rows_or_header, "fieldnames", None)
    if fieldnames is not None:
        return [str(item) for item in fieldnames]
    return []


def build_track1_mapping(schema: Track1ExportSchema, generic_columns: List[str]) -> Dict[str, str]:
    """Build a conservative exact-name mapping from official columns to generic columns."""
    if not schema.schema_confirmed:
        warnings.warn(
            "Track 1 schema is not confirmed; returning empty mapping.",
            RuntimeWarning,
        )
        return {}
    generic_set = set(generic_columns)
    mapping = {}
    for column in schema.columns:
        mapping[column] = column if column in generic_set else ""
    return mapping


def write_track1_mapping_report(
    mapping: Dict[str, str],
    output_path: Union[str, Path],
    schema: Any = None,
    generic_columns: Any = None,
) -> None:
    """Write a Markdown mapping report."""
    output = Path(output_path)
    lines = []
    lines.append("# Track 1 Mapping")
    lines.append("")
    if schema is not None:
        lines.append("- schema_confirmed: `%s`" % bool(getattr(schema, "schema_confirmed", False)))
        lines.append("- schema_source: `%s`" % str(getattr(schema, "source", "")))
        lines.append("")
    lines.append("## Generic Columns")
    lines.append("")
    columns = generic_columns if generic_columns is not None else GENERIC_TRACKING_COLUMNS
    for column in columns:
        lines.append("- `%s`" % column)
    lines.append("")
    lines.append("## Mapping")
    lines.append("")
    if not mapping:
        lines.append("Official mapping is not available because the Track 1 schema is not confirmed.")
        lines.append("")
        lines.append("TODO: confirm official schema, then map official_column -> generic_column or transform.")
    else:
        lines.append("| official_column | generic_column | status |")
        lines.append("| --- | --- | --- |")
        for official, generic in mapping.items():
            status = "ok" if generic else "missing"
            lines.append("| `%s` | `%s` | %s |" % (official, generic, status))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
