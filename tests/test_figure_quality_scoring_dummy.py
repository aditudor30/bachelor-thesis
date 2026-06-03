from deep_oc_sort_3d.visualization3d.figure_candidate_selection import FigureCandidate
from deep_oc_sort_3d.visualization3d.figure_quality_scoring import (
    explain_candidate_score,
    score_cuboid_3d_candidate,
    score_tracking_2d_candidate,
)


def test_tracking_and_cuboid_scores_are_positive():
    tracking = FigureCandidate(
        subset="official_val",
        split="val",
        scene_name="Warehouse_020",
        camera_id="Camera_0000",
        frame_id=100,
        records_path="records.csv",
        figure_type="tracking_2d",
        num_records=10,
        num_assigned=10,
        num_classes=2,
        class_counts={"Person": 8, "Forklift": 2},
        num_projectable_3d=0,
        projection_success_rate=None,
        score=0.0,
        notes="",
    )
    cuboid = FigureCandidate(
        subset="official_val",
        split="val",
        scene_name="Warehouse_020",
        camera_id="Camera_0000",
        frame_id=100,
        records_path="records.csv",
        figure_type="cuboid_3d",
        num_records=8,
        num_assigned=8,
        num_classes=2,
        class_counts={"Person": 6, "Forklift": 2},
        num_projectable_3d=6,
        projection_success_rate=0.75,
        score=0.0,
        notes="projection_checked",
    )
    assert score_tracking_2d_candidate(tracking) > 0.0
    assert score_cuboid_3d_candidate(cuboid) > 0.0
    assert "records" in explain_candidate_score(tracking)

