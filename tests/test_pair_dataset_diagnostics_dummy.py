from deep_oc_sort_3d.learned_association.pair_dataset_diagnostics import determine_dataset_verdict


def test_diagnostics_verdict_ready_for_dense_dataset():
    result = determine_dataset_verdict(
        num_fragments=1000,
        num_valid_fragments=800,
        num_positive_pairs=1000,
        num_negative_pairs=3000,
        missing_reid_rate=0.1,
        warnings=[],
    )

    assert result["verdict"] == "person_pair_dataset_ready_for_training"
    assert result["ready_for_step_20b"] is True


def test_diagnostics_verdict_too_sparse():
    result = determine_dataset_verdict(
        num_fragments=100,
        num_valid_fragments=50,
        num_positive_pairs=10,
        num_negative_pairs=20,
        missing_reid_rate=0.0,
        warnings=[],
    )

    assert result["verdict"] == "person_pair_dataset_too_sparse"
    assert result["ready_for_step_20b"] is False
