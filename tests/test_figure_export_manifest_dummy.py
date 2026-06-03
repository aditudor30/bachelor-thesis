from deep_oc_sort_3d.visualization3d.figure_export_manifest import (
    FigureExportRecord,
    read_figure_manifest,
    summarize_figure_manifest,
    write_figure_manifest,
)


def test_figure_manifest_round_trip(tmp_path):
    path = tmp_path / "figure_manifest.csv"
    records = [
        FigureExportRecord(
            figure_name="tracking_2d_01",
            figure_type="tracking_2d",
            subset="official_val",
            scene_name="Warehouse_020",
            camera_id="Camera_0000",
            frame_id=100,
            input_path="records.csv",
            output_path="tracking_2d_01.png",
            score=1.5,
            caption_suggestion="caption",
            notes="ok",
        )
    ]
    write_figure_manifest(records, path)
    loaded = read_figure_manifest(path)
    assert len(loaded) == 1
    assert loaded[0].figure_name == "tracking_2d_01"
    summary = summarize_figure_manifest(loaded)
    assert summary["num_figures"] == 1
    assert summary["per_type"]["tracking_2d"] == 1
