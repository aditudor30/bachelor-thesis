from deep_oc_sort_3d.priors3d.source_metadata_schema import (
    CENTER_3D_SOURCE_VALUES,
    build_source_metadata_schema,
    required_field_names,
)


def test_source_metadata_schema_contains_required_fields_and_enums():
    schema = build_source_metadata_schema()
    required = required_field_names(schema)
    fields = {field["name"]: field for field in schema["fields"]}

    assert "center_3d_source" in required
    assert "dimensions_3d_source" in required
    assert "yaw_source" in required
    assert "depth_source" in required
    assert "is_gt_derived" in required
    assert "is_estimated_for_test" in required
    assert "pseudo3d_bbox_height" in CENTER_3D_SOURCE_VALUES
    assert "unknown" in fields["center_3d_source"]["allowed_values"]

