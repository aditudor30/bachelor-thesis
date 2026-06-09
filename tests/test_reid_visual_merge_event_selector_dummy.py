from deep_oc_sort_3d.reid_visual_decision.merge_event_selector import select_events_for_review


def test_reid_visual_selector_keeps_diverse_high_priority_events():
    events = [
        {"merge_event_id": "a", "reid_similarity": "0.91", "temporal_gap": "2", "spatial_distance": "1"},
        {"merge_event_id": "b", "reid_similarity": "0.801", "temporal_gap": "2", "spatial_distance": "1"},
        {"merge_event_id": "c", "reid_similarity": "0.82", "temporal_gap": "300", "spatial_distance": "1"},
        {"merge_event_id": "d", "reid_similarity": "0.83", "temporal_gap": "2", "spatial_distance": "15"},
    ]
    selected = select_events_for_review(events, max_events=3, threshold=0.80)
    assert len(selected) == 3
    ids = set([row["merge_event_id"] for row in selected])
    assert "a" in ids

