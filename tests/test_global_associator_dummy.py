"""Dummy tests for the high-level global associator."""

from deep_oc_sort_3d.mtmc.global_associator import GlobalMTMCAssociator
from tests.test_global_association_cost_dummy import make_candidate


def test_associator_builds_multicamera_track_and_singleton():
    candidates = [
        make_candidate("a", "Camera_0000", offset=0.0),
        make_candidate("b", "Camera_0001", offset=0.1),
        make_candidate("c", "Camera_0002", offset=0.2),
        make_candidate("d", "Camera_0003", offset=50.0),
    ]
    associator = GlobalMTMCAssociator(config={"cost_threshold": 1.0})
    tracks, edges, mapping = associator.associate(candidates, show_progress=False)
    sizes = sorted([track.num_candidates for track in tracks])
    assert sizes == [1, 3]
    assert mapping["a"] == mapping["b"]
    assert mapping["a"] == mapping["c"]
    assert mapping["d"] != mapping["a"]
    assert len(edges) > 0
