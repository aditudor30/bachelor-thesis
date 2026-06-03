import csv

from deep_oc_sort_3d.final_export.track1_export_types import (
    Track1ExportSchema,
    default_unconfirmed_track1_schema,
)
from deep_oc_sort_3d.final_export.track1_mapping import build_track1_mapping
from deep_oc_sort_3d.final_export.track1_writer import export_track1_from_generic


def test_track1_writer_unconfirmed_creates_preview_not_official(tmp_path):
    generic_root = _make_generic_export(tmp_path)
    schema = default_unconfirmed_track1_schema()
    output_path = tmp_path / "submission" / "track1.txt"

    summary = export_track1_from_generic(
        generic_export_root=generic_root,
        output_path=output_path,
        schema=schema,
        mapping={},
        subsets=["test"],
        show_progress=False,
    )

    assert not summary["official_export_created"]
    assert summary["reason"] == "schema_not_confirmed"
    assert (tmp_path / "submission" / "track1_unconfirmed_preview.csv").exists()
    assert not output_path.exists()


def test_track1_writer_confirmed_schema_writes_track1(tmp_path):
    generic_root = _make_generic_export(tmp_path)
    schema = Track1ExportSchema(
        schema_confirmed=True,
        source="dummy",
        columns=["scene_name", "frame_id", "global_track_id", "x1"],
        has_header=True,
        delimiter=",",
        frame_indexing="zero_based",
        id_scope="global",
        notes="dummy",
    )
    mapping = build_track1_mapping(schema, ["scene_name", "frame_id", "global_track_id", "x1"])
    output_path = tmp_path / "submission" / "track1.txt"

    summary = export_track1_from_generic(
        generic_export_root=generic_root,
        output_path=output_path,
        schema=schema,
        mapping=mapping,
        subsets=["test"],
        show_progress=False,
    )

    assert summary["official_export_created"]
    assert output_path.exists()
    rows = list(csv.reader(output_path.open("r", newline="", encoding="utf-8")))
    assert rows[0] == ["scene_name", "frame_id", "global_track_id", "x1"]
    assert rows[1] == ["Warehouse_023", "0", "10", "1.0"]


def _make_generic_export(tmp_path):
    root = tmp_path / "generic"
    scene_dir = root / "test"
    scene_dir.mkdir(parents=True)
    path = scene_dir / "Warehouse_023.csv"
    fields = [
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "scene_name": "Warehouse_023",
                "camera_id": "Camera_0000",
                "frame_id": "0",
                "global_track_id": "10",
                "class_id": "0",
                "class_name": "Person",
                "confidence": "0.9",
                "x1": "1.0",
                "y1": "2.0",
                "x2": "10.0",
                "y2": "20.0",
                "w": "9.0",
                "h": "18.0",
                "center_x": "0.0",
                "center_y": "0.0",
                "center_z": "0.0",
                "width_3d": "1.0",
                "length_3d": "1.0",
                "height_3d": "1.0",
                "yaw": "0.0",
            }
        )
    return root
