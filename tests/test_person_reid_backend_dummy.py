import numpy as np

from deep_oc_sort_3d.person_reid.reid_backends import build_person_reid_backend


def test_person_reid_dummy_backend_extracts_embedding():
    result = build_person_reid_backend({"name": "dummy", "allow_dummy": True, "embedding_dim": 12})
    crop = np.ones((32, 16, 3), dtype=np.uint8) * 127
    embedding = result.backend.extract(crop)
    assert result.available is True
    assert embedding.shape == (12,)


def test_person_reid_backend_reports_missing_weights():
    result = build_person_reid_backend({"name": "torchreid_osnet", "weights_path": None})
    assert result.available is False
    assert result.status == "backend_unavailable"

