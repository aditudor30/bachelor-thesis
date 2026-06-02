"""Dummy tests for global association I/O."""

from deep_oc_sort_3d.mtmc.global_io import (
    read_association_edges_csv,
    read_association_edges_jsonl,
    read_global_tracks_csv,
    read_global_tracks_jsonl,
    write_association_edges_csv,
    write_association_edges_jsonl,
    write_global_tracks_csv,
    write_global_tracks_jsonl,
)
from deep_oc_sort_3d.mtmc.global_association_graph import build_global_tracks_from_edges
from tests.test_global_association_graph_dummy import make_edge
from tests.test_global_association_cost_dummy import make_candidate


def test_global_track_and_edge_roundtrip(tmp_path):
    a = make_candidate("a", "Camera_0000", offset=0.0)
    b = make_candidate("b", "Camera_0001", offset=0.1)
    edge = make_edge(a, b)
    tracks, _mapping = build_global_tracks_from_edges([a, b], [edge], {}, show_progress=False)

    tracks_jsonl = tmp_path / "global_tracks.jsonl"
    tracks_csv = tmp_path / "global_tracks.csv"
    edges_jsonl = tmp_path / "association_edges.jsonl"
    edges_csv = tmp_path / "association_edges.csv"

    write_global_tracks_jsonl(tracks, tracks_jsonl)
    write_global_tracks_csv(tracks, tracks_csv)
    write_association_edges_jsonl([edge], edges_jsonl)
    write_association_edges_csv([edge], edges_csv)

    assert read_global_tracks_jsonl(tracks_jsonl)[0].num_candidates == 2
    assert read_global_tracks_csv(tracks_csv)[0].num_candidates == 2
    assert read_association_edges_jsonl(edges_jsonl)[0].accepted
    assert read_association_edges_csv(edges_csv)[0].accepted
