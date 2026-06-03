"""Types and schema helpers for Track 1 export scaffolding."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class Track1ExportSchema:
    """Description of the official Track 1 export schema.

    When ``schema_confirmed`` is False, this object is intentionally incomplete
    and must not be used to claim an official submission file.
    """

    schema_confirmed: bool
    source: str
    columns: List[str]
    has_header: Optional[bool]
    delimiter: str
    frame_indexing: Optional[str]
    id_scope: Optional[str]
    notes: str


@dataclass
class Track1ExportRow:
    """One schema-driven Track 1 row."""

    values: Dict[str, Any]


def default_unconfirmed_track1_schema(source: str = "not_confirmed") -> Track1ExportSchema:
    """Return a safe unconfirmed Track 1 schema placeholder."""
    return Track1ExportSchema(
        schema_confirmed=False,
        source=source,
        columns=[],
        has_header=None,
        delimiter=",",
        frame_indexing="unknown",
        id_scope="unknown",
        notes="Official Track 1 schema is not confirmed. Do not produce final track1.txt.",
    )


def track1_schema_from_dict(data: Dict[str, Any]) -> Track1ExportSchema:
    """Build a Track1ExportSchema from a dictionary."""
    return Track1ExportSchema(
        schema_confirmed=bool(data.get("schema_confirmed", False)),
        source=str(data.get("source", "")),
        columns=[str(item) for item in data.get("columns", [])],
        has_header=_optional_bool(data.get("has_header")),
        delimiter=str(data.get("delimiter", ",")),
        frame_indexing=_optional_str(data.get("frame_indexing", "unknown")),
        id_scope=_optional_str(data.get("id_scope", "unknown")),
        notes=str(data.get("notes", "")),
    )


def track1_schema_to_dict(schema: Track1ExportSchema) -> Dict[str, Any]:
    """Convert a Track1ExportSchema to a dictionary."""
    return {
        "schema_confirmed": schema.schema_confirmed,
        "source": schema.source,
        "columns": list(schema.columns),
        "has_header": schema.has_header,
        "delimiter": schema.delimiter,
        "frame_indexing": schema.frame_indexing,
        "id_scope": schema.id_scope,
        "notes": schema.notes,
    }


def load_track1_schema_yaml(path: Union[str, Path]) -> Track1ExportSchema:
    """Load a Track1ExportSchema from YAML."""
    schema_path = Path(path)
    data = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Track 1 schema YAML must be a mapping.")
    section = data.get("track1_schema", data)
    if not isinstance(section, dict):
        raise ValueError("track1_schema section must be a mapping.")
    if "source" not in section:
        section["source"] = str(schema_path)
    return track1_schema_from_dict(section)


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
