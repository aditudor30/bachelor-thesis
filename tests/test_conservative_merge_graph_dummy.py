from deep_oc_sort_3d.learned_association_application.conservative_merge_graph import (
    ConservativeUnionFind,
    build_conservative_merge_mapping,
)


def _row(pair_id, left, right, camera_a="Camera_0000", camera_b="Camera_0001"):
    return {
        "pair_id": pair_id,
        "subset": "official_val",
        "scene_name": "Warehouse_020",
        "class_id": 0,
        "class_name": "Person",
        "global_track_a": str(left),
        "global_track_b": str(right),
        "camera_a": camera_a,
        "camera_b": camera_b,
        "frame_start_a": 0,
        "frame_end_a": 10,
        "frame_start_b": 20,
        "frame_end_b": 30,
        "mlp_score": 0.9,
        "reid_similarity": 0.9,
        "spatial_distance": 2.0,
        "temporal_gap": 10,
        "valid_for_merge": 1,
    }


def test_conservative_graph_merges_safe_edge():
    mapping, accepted, rejected, summary = build_conservative_merge_mapping(
        [_row("p1", 1, 2)],
        "mlp_geometry_safe",
        {"thresholds": {}, "constraints": {"max_component_size": 4}, "candidate_scoring": {"require_valid_reid": True}},
    )
    assert len(accepted) == 1
    assert len(rejected) == 0
    assert len(mapping) == 2
    assert summary["merged_components"] == 1


def test_union_find_rejects_same_camera_overlap_and_size():
    graph = ConservativeUnionFind(max_component_size=2)
    a = ("val", "Warehouse_020", "0", "1")
    b = ("val", "Warehouse_020", "0", "2")
    graph.add(a, {"cameras": {"Camera_0000"}, "frame_start": 0, "frame_end": 10})
    graph.add(b, {"cameras": {"Camera_0000"}, "frame_start": 5, "frame_end": 15})
    merged, reason = graph.try_union(a, b)
    assert not merged
    assert reason == "component_same_camera_temporal_conflict"


def test_union_find_enforces_max_component_size():
    graph = ConservativeUnionFind(max_component_size=2)
    a = ("val", "Warehouse_020", "0", "1")
    b = ("val", "Warehouse_020", "0", "2")
    c = ("val", "Warehouse_020", "0", "3")
    graph.add(a, {"cameras": {"Camera_0000"}, "frame_start": 0, "frame_end": 5})
    graph.add(b, {"cameras": {"Camera_0001"}, "frame_start": 10, "frame_end": 15})
    graph.add(c, {"cameras": {"Camera_0002"}, "frame_start": 20, "frame_end": 25})
    assert graph.try_union(a, b)[0]
    merged, reason = graph.try_union(a, c)
    assert not merged
    assert reason == "max_component_size"
