from deep_oc_sort_3d.reid_ablation.ablation_metric_loader import make_variant_metric_row


def test_reid_ablation_metric_loader_common_schema():
    metrics = {
        "track1_validation_errors": 0,
        "track1_rows": 100,
        "person_rows": 60,
        "non_person_rows": 40,
        "global_purity_mean": 0.98,
        "false_merge_rate": 0.02,
        "fragmentation_approx": 10,
        "person_fragmentation_approx": 7,
    }
    row = make_variant_metric_row(
        "reid_medium",
        "reid_only",
        metrics,
        {"num_reid_merges": 2, "num_export_dropped_rows": 0},
    )
    assert row["variant_name"] == "reid_medium"
    assert row["source_type"] == "reid_only"
    assert row["track1_valid"] is True
    assert row["num_reid_merges"] == 2

