import csv

from deep_oc_sort_3d.visualization3d.figure_candidate_selection import (
    scan_frame_records_for_candidates,
    select_top_candidates,
)


def test_scan_frame_records_for_candidates_dummy(tmp_path):
    records_dir = tmp_path / "records" / "official_val" / "Warehouse_020"
    records_dir.mkdir(parents=True)
    path = records_dir / "Camera_0000_global_records.csv"
    write_dummy_records(path)

    candidates = scan_frame_records_for_candidates(
        records_dir.parent.parent,
        subsets=["official_val"],
        frame_stride=50,
        max_frames_per_camera=10,
        figure_type="tracking_2d",
        show_progress=False,
    )
    assert len(candidates) == 2
    top = select_top_candidates(candidates, top_k=1, min_records=3, max_records=30)
    assert len(top) == 1
    assert top[0].scene_name == "Warehouse_020"
    assert top[0].camera_id == "Camera_0000"


def write_dummy_records(path):
    fields = [
        "scene_name",
        "camera_id",
        "frame_id",
        "global_track_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
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
        for frame_id in [0, 50]:
            for index in range(4):
                writer.writerow(
                    {
                        "scene_name": "Warehouse_020",
                        "camera_id": "Camera_0000",
                        "frame_id": frame_id,
                        "global_track_id": index,
                        "class_name": "Person" if index < 3 else "Forklift",
                        "confidence": 0.9,
                        "x1": 1,
                        "y1": 2,
                        "x2": 20,
                        "y2": 40,
                        "center_x": 1.0,
                        "center_y": 2.0,
                        "center_z": 3.0,
                        "width_3d": 1.0,
                        "length_3d": 2.0,
                        "height_3d": 3.0,
                        "yaw": 0.0,
                    }
                )

