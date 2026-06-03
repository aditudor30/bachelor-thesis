import numpy as np

from deep_oc_sort_3d.reid.embedding_backends import ColorHistogramBackend, DummyEmbeddingBackend, build_embedding_backend


def test_color_histogram_and_dummy_backends_are_deterministic():
    crop = np.ones((16, 8, 3), dtype=np.uint8) * 128
    backend = ColorHistogramBackend(bins_per_channel=8, resize=(8, 16))
    embedding = backend.extract(crop)
    assert embedding.shape[0] == 24
    assert np.isfinite(embedding).all()

    dummy = DummyEmbeddingBackend(embedding_dim=4)
    a = dummy.extract(crop)
    b = dummy.extract(crop)
    np.testing.assert_allclose(a, b)

    built = build_embedding_backend({"backend": "color_histogram", "bins_per_channel": 4, "resize": (8, 16)})
    assert built.embedding_dim == 12

