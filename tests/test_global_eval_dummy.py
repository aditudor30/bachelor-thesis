"""Dummy tests for global MTMC diagnostic evaluation."""

import numpy as np

from deep_oc_sort_3d.mtmc.global_eval import evaluate_global_tracks
from deep_oc_sort_3d.mtmc.global_types import GlobalTrack


def make_track(global_track_id, class_name, gt_counts, purity, num_cameras=2):
    return GlobalTrack(
        global_track_id=global_track_id,
        scene_name="Warehouse_020",
        subset="official_val",
        split="val",
        class_id=0,
        class_name=class_name,
        candidate_ids=["c%d" % global_track_id],
        camera_ids=["Camera_0000", "Camera_0001"][:num_cameras],
        local_track_ids=[global_track_id],
        start_frame=0,
        end_frame=10,
        duration=11,
        num_candidates=max(1, sum(gt_counts.values())),
        num_cameras=num_cameras,
        mean_confidence=0.8,
        max_confidence=0.9,
        trajectory_3d_sampled=[(0, 0.0, 0.0, 0.0)],
        center_3d_mean=np.asarray([0.0, 0.0, 0.0], dtype=float),
        majority_gt_object_id=int(max(gt_counts.keys(), key=lambda key: gt_counts[key])),
        gt_purity=purity,
        num_gt_ids=len(gt_counts),
        gt_id_counts=gt_counts,
        notes="dummy",
    )


def test_global_eval_reports_purity_false_merge_and_fragmentation():
    tracks = [
        make_track(0, "Person", {"1": 2}, 1.0),
        make_track(1, "Person", {"1": 1}, 1.0, num_cameras=1),
        make_track(2, "Person", {"2": 1, "3": 1}, 0.5),
    ]
    metrics = evaluate_global_tracks(tracks)
    assert metrics["num_global_tracks"] == 3
    assert metrics["num_multi_camera_tracks"] == 2
    assert metrics["false_merge_count"] == 1
    assert metrics["fragmentation_approx"] == 1
    assert metrics["global_purity_mean"] < 1.0
