"""Explicit 3D provenance metadata schema for future pseudo-3D outputs."""

from typing import Any, Dict, List


CENTER_3D_SOURCE_VALUES = [
    "gt",
    "depth_sampled",
    "pseudo3d_bbox_height",
    "pseudo3d_ground_plane",
    "pseudo3d_motion_smoothed",
    "class_default",
    "propagated",
    "unknown",
]

DIMENSIONS_3D_SOURCE_VALUES = ["gt", "class_prior", "predicted", "default", "unknown"]
YAW_SOURCE_VALUES = ["gt", "motion_direction", "class_default", "predicted", "unknown"]
DEPTH_SOURCE_VALUES = [
    "depth_map",
    "bbox_height_prior",
    "ground_plane_intersection",
    "temporal_smoothing",
    "unknown",
]


def build_source_metadata_schema() -> Dict[str, Any]:
    """Build the JSON schema-like contract for 3D provenance fields."""
    return {
        "name": "source_metadata_schema_3d",
        "version": "0.1",
        "description": "Provenance fields for future Observation3D and frame-level pseudo-3D outputs.",
        "fields": [
            _enum_field("center_3d_source", CENTER_3D_SOURCE_VALUES, required=True),
            _enum_field("dimensions_3d_source", DIMENSIONS_3D_SOURCE_VALUES, required=True),
            _enum_field("yaw_source", YAW_SOURCE_VALUES, required=True),
            _enum_field("depth_source", DEPTH_SOURCE_VALUES, required=True),
            _bool_field("is_gt_derived", required=True),
            _bool_field("is_estimated_for_test", required=True),
            _optional_string_field("pseudo3d_method"),
            _optional_string_field("pseudo3d_version"),
            _optional_float_field("pseudo3d_confidence"),
            _optional_bool_field("projection_valid"),
            _optional_string_field("projection_error_reason"),
            _string_field("source_notes", required=True),
        ],
    }


def build_source_metadata_schema_markdown(schema: Dict[str, Any]) -> str:
    """Build Markdown documentation for the source metadata schema."""
    lines = [
        "# 3D Source Metadata Schema",
        "",
        "This schema is defined in Step 15B and is not yet integrated into Observation3D.",
        "Future pseudo-3D outputs should write these fields so provenance is auditable.",
        "",
        "## Fields",
        "",
    ]
    for field in schema.get("fields", []):
        lines.extend(
            [
                "### `%s`" % field.get("name"),
                "",
                "- Type: `%s`" % field.get("type"),
                "- Required: `%s`" % field.get("required"),
                "- Description: %s" % field.get("description"),
            ]
        )
        values = field.get("allowed_values", [])
        if values:
            lines.append("- Allowed values: `%s`" % "`, `".join(values))
        lines.append("")
    lines.extend(
        [
            "## Rules",
            "",
            "- Do not mark test records as `gt` or `is_gt_derived=true`.",
            "- Use `unknown` when provenance cannot be proven from explicit metadata.",
            "- `pseudo3d_confidence` is a diagnostic confidence, not benchmark accuracy.",
            "- Projection fields describe geometric/projective validity, not object identity correctness.",
            "",
        ]
    )
    return "\n".join(lines)


def required_field_names(schema: Dict[str, Any]) -> List[str]:
    """Return required field names from the schema."""
    return [field["name"] for field in schema.get("fields", []) if field.get("required")]


def _enum_field(name: str, values: List[str], required: bool) -> Dict[str, Any]:
    return {
        "name": name,
        "type": "string",
        "required": required,
        "allowed_values": values,
        "description": "Explicit provenance enum for %s." % name,
    }


def _bool_field(name: str, required: bool) -> Dict[str, Any]:
    return {"name": name, "type": "bool", "required": required, "description": "Boolean provenance guard."}


def _optional_bool_field(name: str) -> Dict[str, Any]:
    return {"name": name, "type": "Optional[bool]", "required": False, "description": "Optional boolean diagnostic."}


def _optional_string_field(name: str) -> Dict[str, Any]:
    return {"name": name, "type": "Optional[str]", "required": False, "description": "Optional string diagnostic."}


def _optional_float_field(name: str) -> Dict[str, Any]:
    return {"name": name, "type": "Optional[float]", "required": False, "description": "Optional numeric diagnostic."}


def _string_field(name: str, required: bool) -> Dict[str, Any]:
    return {"name": name, "type": "string", "required": required, "description": "Free-form provenance notes."}

