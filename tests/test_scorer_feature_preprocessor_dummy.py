import numpy as np

from deep_oc_sort_3d.learned_association.scorer_feature_preprocessor import PairFeaturePreprocessor


def test_preprocessor_imputes_scales_and_one_hot_encodes_camera_pair():
    train_rows = [
        {
            "reid_similarity": "0.9",
            "temporal_gap": "10",
            "camera_pair": "Camera_0000__Camera_0001",
            "temporal_order": "a_before_b",
            "fragment_quality_a": "good",
            "fragment_quality_b": "good",
        },
        {
            "reid_similarity": "",
            "temporal_gap": "30",
            "camera_pair": "Camera_0000__Camera_0002",
            "temporal_order": "overlap",
            "fragment_quality_a": "short",
            "fragment_quality_b": "good",
        },
    ]
    val_rows = [
        {
            "reid_similarity": "0.7",
            "temporal_gap": "20",
            "camera_pair": "Camera_0099__Camera_0100",
            "temporal_order": "a_before_b",
            "fragment_quality_a": "good",
            "fragment_quality_b": "good",
        }
    ]
    preprocessor = PairFeaturePreprocessor("median", True, "one_hot")

    train_matrix = preprocessor.fit_transform(train_rows)
    val_matrix = preprocessor.transform(val_rows)

    assert train_matrix.shape[0] == 2
    assert val_matrix.shape == (1, train_matrix.shape[1])
    assert np.all(np.isfinite(train_matrix))
    assert np.all(np.isfinite(val_matrix))
    assert "camera_pair=Camera_0000__Camera_0001" in preprocessor.output_features
    assert preprocessor.fill_values["reid_similarity"] == 0.9


def test_preprocessor_selects_only_available_numeric_features():
    rows = [{"temporal_gap": "5", "camera_pair": "a__b"}]
    preprocessor = PairFeaturePreprocessor().fit(rows)

    assert "temporal_gap" in preprocessor.numeric_features
    assert "reid_similarity" not in preprocessor.numeric_features
