"""Dummy tests for global association graph utilities."""

from deep_oc_sort_3d.mtmc.global_association_graph import (
    UnionFind,
    build_global_tracks_from_edges,
)
from deep_oc_sort_3d.mtmc.global_types import GlobalAssociationEdge
from tests.test_global_association_cost_dummy import make_candidate


def make_edge(candidate_a, candidate_b, accepted=True):
    return GlobalAssociationEdge(
        scene_name=candidate_a.scene_name,
        subset=candidate_a.subset,
        class_id=candidate_a.class_id,
        class_name=candidate_a.class_name,
        candidate_id_a=candidate_a.candidate_id,
        candidate_id_b=candidate_b.candidate_id,
        camera_id_a=candidate_a.camera_id,
        camera_id_b=candidate_b.camera_id,
        start_frame_a=candidate_a.start_frame,
        end_frame_a=candidate_a.end_frame,
        start_frame_b=candidate_b.start_frame,
        end_frame_b=candidate_b.end_frame,
        temporal_relation="overlap",
        overlap_frames=3,
        temporal_gap=0,
        mean_3d_distance=0.1,
        median_3d_distance=0.1,
        min_3d_distance=0.1,
        max_3d_distance=0.1,
        entry_exit_distance=None,
        velocity_angle_difference=0.0,
        cost=0.1,
        affinity=0.9,
        accepted=accepted,
        reject_reason="ok" if accepted else "far",
    )


def test_union_find_merges_components():
    uf = UnionFind(3)
    assert uf.find(0) == 0
    assert uf.union(0, 1)
    assert uf.find(0) == uf.find(1)
    assert not uf.union(0, 1)


def test_build_global_tracks_keeps_singletons():
    a = make_candidate("a", "Camera_0000", offset=0.0)
    b = make_candidate("b", "Camera_0001", offset=0.1)
    c = make_candidate("c", "Camera_0002", offset=20.0)
    tracks, mapping = build_global_tracks_from_edges([a, b, c], [make_edge(a, b)], {}, show_progress=False)
    sizes = sorted([track.num_candidates for track in tracks])
    assert sizes == [1, 2]
    assert mapping["a"] == mapping["b"]
    assert mapping["c"] != mapping["a"]
