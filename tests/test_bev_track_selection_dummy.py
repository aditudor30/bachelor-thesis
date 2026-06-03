import csv

from deep_oc_sort_3d.visualization3d.bev_track_selection import (
    filter_bev_tracks,
    load_bev_tracks_from_generic_csv,
)


def test_load_and_filter_bev_tracks_from_generic_csv(tmp_path):
    path = tmp_path / "generic.csv"
    write_generic_csv(path)
    tracks = load_bev_tracks_from_generic_csv(path)
    assert len(tracks) == 3

    filtered = filter_bev_tracks(tracks, min_track_length=3, max_tracks=1, class_name="Person")
    assert len(filtered) == 1
    assert filtered[0].global_track_id == 1
    assert filtered[0].length == 4

    forklifts = filter_bev_tracks(tracks, min_track_length=1, max_tracks=None, class_id=1)
    assert len(forklifts) == 1
    assert forklifts[0].class_name == "Forklift"


def write_generic_csv(path):
    fields = ["frame_id", "global_track_id", "class_id", "class_name", "confidence", "center_x", "center_y"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(4):
            writer.writerow(
                {
                    "frame_id": frame,
                    "global_track_id": 1,
                    "class_id": 0,
                    "class_name": "Person",
                    "confidence": 0.9,
                    "center_x": float(frame),
                    "center_y": float(frame + 1),
                }
            )
        for frame in range(2):
            writer.writerow(
                {
                    "frame_id": frame,
                    "global_track_id": 2,
                    "class_id": 1,
                    "class_name": "Forklift",
                    "confidence": 0.8,
                    "center_x": float(frame + 10),
                    "center_y": float(frame + 11),
                }
            )
        writer.writerow(
            {
                "frame_id": 0,
                "global_track_id": 3,
                "class_id": 0,
                "class_name": "Person",
                "confidence": 0.1,
                "center_x": 100.0,
                "center_y": 100.0,
            }
        )

