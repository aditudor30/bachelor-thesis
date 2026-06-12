from deep_oc_sort_3d.bytetrack_tuning.coverage_drop_analyzer import compute_coverage_drop_rows


def test_coverage_drop_computation():
    variants = {
        "dense": {
            "stage_counts": {"observations": 100, "local_records": 80},
            "dimensions": {
                "observations": {"per_class": {"Person": 70, "Forklift": 30}},
                "local_records": {"per_class": {"Person": 55, "Forklift": 25}},
            },
        }
    }
    rows = compute_coverage_drop_rows(variants)
    row = [item for item in rows if item["input_stage"] == "observations"][0]
    assert row["output_stage"] == "local_records"
    assert row["retention"] == 0.8
    assert row["drop_count"] == 20
    assert "Person" in row["top_affected_classes"]

