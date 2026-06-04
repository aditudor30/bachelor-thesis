"""Dummy tests for the ReID-aware global associator."""

import numpy as np

from deep_oc_sort_3d.mtmc.global_reid_associator import GlobalMTMCReIDAssociator
from deep_oc_sort_3d.reid.reid_io import write_reid_embeddings_jsonl
from deep_oc_sort_3d.reid.reid_types import ReIDEmbeddingRecord
from tests.test_global_association_cost_dummy import make_candidate


def _embedding(candidate_id, camera_id, local_track_id, embedding):
    return ReIDEmbeddingRecord(
        embedding_id=candidate_id,
        subset="official_val",
        split="val",
        scene_name="Warehouse_020",
        camera_id=camera_id,
        frame_id=None,
        local_track_id=local_track_id,
        global_track_id=None,
        candidate_id=candidate_id,
        class_id=0,
        class_name="Person",
        embedding=np.asarray(embedding, dtype=float),
        embedding_dim=2,
        backend="dummy",
        num_crops=1,
        crop_frame_ids=[0],
        mean_confidence=0.9,
        notes="dummy",
    )


def _config(root, use_reid=True):
    return {
        "global_mtmc": {
            "cost_threshold": 1.0,
            "enable_overlap_association": True,
            "enable_transition_association": False,
        },
        "reid": {
            "use_reid": use_reid,
            "reid_root": str(root),
            "appearance_weight": 0.10,
            "geometry_only_fallback": True,
        },
    }


def test_reid_associator_uses_reid_when_embeddings_exist(tmp_path):
    root = tmp_path / "reid"
    scene_root = root / "candidate_embeddings" / "official_val" / "Warehouse_020"
    a = make_candidate("a", "Camera_0000", offset=0.0)
    b = make_candidate("b", "Camera_0001", offset=0.1)
    write_reid_embeddings_jsonl(
        [
            _embedding("a", "Camera_0000", 1, [1.0, 0.0]),
            _embedding("b", "Camera_0001", 1, [1.0, 0.0]),
        ],
        scene_root / "embeddings.jsonl",
    )

    associator = GlobalMTMCReIDAssociator(config=_config(root))
    tracks, edges, mapping = associator.associate([a, b], show_progress=False)

    assert len(tracks) == 1
    assert mapping["a"] == mapping["b"]
    assert edges[0].accepted
    assert edges[0].used_reid
    assert edges[0].appearance_distance == 0.0


def test_reid_associator_falls_back_when_embedding_missing(tmp_path):
    a = make_candidate("a", "Camera_0000", offset=0.0)
    b = make_candidate("b", "Camera_0001", offset=0.1)

    associator = GlobalMTMCReIDAssociator(config=_config(tmp_path / "missing"))
    _tracks, edges, mapping = associator.associate([a, b], show_progress=False)

    assert mapping["a"] == mapping["b"]
    assert edges[0].accepted
    assert not edges[0].used_reid
    assert "fallback_geometry" in edges[0].reid_missing_reason


def test_reid_associator_can_disable_reid(tmp_path):
    a = make_candidate("a", "Camera_0000", offset=0.0)
    b = make_candidate("b", "Camera_0001", offset=0.1)

    associator = GlobalMTMCReIDAssociator(config=_config(tmp_path / "unused", use_reid=False))
    _tracks, edges, mapping = associator.associate([a, b], show_progress=False)

    assert mapping["a"] == mapping["b"]
    assert edges[0].accepted
    assert not edges[0].used_reid
    assert edges[0].reid_missing_reason == "reid_disabled"
