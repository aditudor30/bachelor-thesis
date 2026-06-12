from deep_oc_sort_3d.bytetrack_tuning.stage_retention_analyzer import (
    build_stage_retention_rows,
    compute_retention,
)


def test_stage_retention_computation():
    metrics = {
        "baselines": {
            "baseline_v2_current": {"stage_counts": {"local_records": 100}},
            "baseline_21b_bytetrack": {"stage_counts": {"local_records": 60}},
        },
        "variants": {"dense": {"stage_counts": {"local_records": 90}}},
    }
    rows = build_stage_retention_rows(metrics)
    row = [item for item in rows if item["stage"] == "local_records"][0]
    assert compute_retention(90, 100) == 0.9
    assert row["retention_vs_v2_current"] == 0.9
    assert row["retention_vs_21b_bytetrack"] == 1.5
    assert row["delta_vs_v2_current"] == -10

