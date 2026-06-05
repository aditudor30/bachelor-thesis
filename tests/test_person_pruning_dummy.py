from deep_oc_sort_3d.person_cleanup.person_pruning import should_drop_person_track
from deep_oc_sort_3d.person_cleanup.person_track_classifier import PersonTrackStats


def _stats(class_id, rows, mean_conf, max_conf):
    return PersonTrackStats(
        key=("test", "Warehouse_023", str(class_id), "1"),
        subset="test",
        scene_name="Warehouse_023",
        class_id=class_id,
        class_name="Person" if class_id == 0 else "Forklift",
        global_track_id="1",
        rows=rows,
        unique_frames=rows,
        cameras=["Camera_0000"],
        min_frame=0,
        max_frame=max(0, rows - 1),
        mean_confidence=mean_conf,
        max_confidence=max_conf,
        min_confidence=mean_conf,
    )


def test_person_pruning_drops_only_low_conf_short_person():
    policy = {"class_id": 0, "mode": "short_lowconf", "max_rows_per_track": 3, "mean_confidence_threshold": 0.03, "max_confidence_threshold": 0.08}

    assert should_drop_person_track(_stats(0, 2, 0.01, 0.02), policy)[0] is True
    assert should_drop_person_track(_stats(0, 2, 0.4, 0.6), policy)[0] is False
    assert should_drop_person_track(_stats(1, 2, 0.01, 0.02), policy)[0] is False

