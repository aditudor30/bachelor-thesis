import numpy as np

from deep_oc_sort_3d.tracking.association import (
    INF_COST,
    associate_detections_to_tracks,
    bbox_iou_xyxy,
    center_distance_3d,
    compute_association_cost,
)
from deep_oc_sort_3d.tracking.track_state import LocalTrack
from deep_oc_sort_3d.tracking.track_types import LocalTrackDetection


def _detection(frame_id, detection_id, class_id=0, center=None, bbox=None):
    if center is None:
        center = [0.0, 0.0, 0.0]
    if bbox is None:
        bbox = (10.0, 10.0, 30.0, 50.0)
    return LocalTrackDetection(
        scene_id=0,
        scene_name="Warehouse_000",
        split="train",
        camera_id="Camera_0000",
        frame_id=frame_id,
        detection_id=detection_id,
        class_id=class_id,
        class_name="Person" if class_id == 0 else "Forklift",
        confidence=0.9,
        bbox_xyxy=bbox,
        bbox_xywh=(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
        center_3d=np.asarray(center, dtype=float),
        dimensions_3d=np.asarray([1.0, 1.0, 1.0], dtype=float),
        yaw=0.0,
        object_id=100,
        matched_gt=True,
        matched_iou=0.8,
        source="dummy",
    )


def test_basic_distance_and_iou_helpers():
    assert bbox_iou_xyxy((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0
    assert center_distance_3d(np.asarray([0.0, 0.0, 0.0]), np.asarray([3.0, 4.0, 0.0])) == 5.0


def test_association_cost_rejects_class_mismatch():
    track = LocalTrack(1, _detection(0, 0, class_id=0))
    det = _detection(1, 1, class_id=1, center=[0.1, 0.0, 0.0])

    cost = compute_association_cost(track, det, frame_id=1, mode="oracle_3d", class_must_match=True)

    assert cost == INF_COST


def test_associate_detections_to_tracks_matches_close_3d_detection():
    track = LocalTrack(1, _detection(0, 0, center=[0.0, 0.0, 0.0]))
    det = _detection(1, 1, center=[0.2, 0.0, 0.0])

    matched, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
        detections=[det],
        tracks=[track],
        frame_id=1,
        config={"mode": "oracle_3d", "cost_threshold": 0.7, "max_3d_distance": 3.0},
    )

    assert matched == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_dets == []
