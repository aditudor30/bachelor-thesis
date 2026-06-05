from deep_oc_sort_3d.person_cleanup.person_track_classifier import PersonTrackStats, classify_person_track


def test_person_track_classifier_labels_short_low_conf_and_high_conf():
    low = PersonTrackStats(
        key=("test", "Warehouse_023", "0", "1"),
        subset="test",
        scene_name="Warehouse_023",
        class_id=0,
        class_name="Person",
        global_track_id="1",
        rows=1,
        unique_frames=1,
        cameras=["Camera_0000"],
        min_frame=0,
        max_frame=0,
        mean_confidence=0.01,
        max_confidence=0.02,
        min_confidence=0.01,
    )
    high = PersonTrackStats(
        key=("test", "Warehouse_023", "0", "2"),
        subset="test",
        scene_name="Warehouse_023",
        class_id=0,
        class_name="Person",
        global_track_id="2",
        rows=2,
        unique_frames=2,
        cameras=["Camera_0000"],
        min_frame=0,
        max_frame=1,
        mean_confidence=0.5,
        max_confidence=0.6,
        min_confidence=0.4,
    )

    assert classify_person_track(low).category == "likely_false_positive_short_fragment"
    assert classify_person_track(high).category == "high_confidence_short_fragment"

