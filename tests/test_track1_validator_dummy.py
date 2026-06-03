from deep_oc_sort_3d.final_export.track1_export_types import (
    Track1ExportSchema,
    default_unconfirmed_track1_schema,
)
from deep_oc_sort_3d.final_export.track1_validator import validate_track1_export


def test_track1_validator_unconfirmed_preview_is_not_official(tmp_path):
    path = tmp_path / "track1_unconfirmed_preview.csv"
    path.write_text("scene_name,frame_id\nWarehouse_023,0\n", encoding="utf-8")

    report = validate_track1_export(path, default_unconfirmed_track1_schema(), show_progress=False)

    assert report["status"] == "ok"
    assert not report["official_validation"]
    assert report["reason"] == "schema_not_confirmed"


def test_track1_validator_confirmed_schema_checks_columns(tmp_path):
    path = tmp_path / "track1.txt"
    path.write_text("scene_name,frame_id,global_track_id,x1\nWarehouse_023,0,10,1.0\n", encoding="utf-8")
    schema = _dummy_schema()

    report = validate_track1_export(path, schema, show_progress=False)

    assert report["status"] == "ok"
    assert report["official_validation"]
    assert report["num_errors"] == 0


def test_track1_validator_confirmed_schema_flags_invalid_values(tmp_path):
    path = tmp_path / "track1.txt"
    path.write_text("scene_name,frame_id,global_track_id,x1\nWarehouse_023,0,10,nan\n", encoding="utf-8")
    schema = _dummy_schema()

    report = validate_track1_export(path, schema, show_progress=False)

    assert report["status"] == "error"
    assert any("nan_or_inf:x1" in item for item in report["errors"])


def _dummy_schema():
    return Track1ExportSchema(
        schema_confirmed=True,
        source="dummy",
        columns=["scene_name", "frame_id", "global_track_id", "x1"],
        has_header=True,
        delimiter=",",
        frame_indexing="zero_based",
        id_scope="global",
        notes="dummy",
    )
