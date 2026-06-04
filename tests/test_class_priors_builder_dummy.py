from deep_oc_sort_3d.priors3d.class_priors_builder import build_final_class_priors, final_priors_to_rows


def test_class_priors_builder_selects_median_and_confidence():
    rows = [
        {
            "class_id": "0",
            "class_name": "Person",
            "count": "1200",
            "width_mean": "0.8",
            "width_median": "0.7",
            "width_std": "0.1",
            "width_p05": "0.5",
            "width_p95": "1.0",
            "length_mean": "0.9",
            "length_median": "0.8",
            "length_std": "0.1",
            "length_p05": "0.6",
            "length_p95": "1.1",
            "height_mean": "1.8",
            "height_median": "1.7",
            "height_std": "0.1",
            "height_p05": "1.5",
            "height_p95": "2.0",
            "looks_constant_or_default": "false",
        },
        {
            "class_id": "1",
            "class_name": "Forklift",
            "count": "50",
            "width_mean": "2.0",
            "width_median": "2.0",
            "width_std": "0.0",
            "width_p05": "2.0",
            "width_p95": "2.0",
            "length_mean": "3.0",
            "length_median": "3.0",
            "length_std": "0.0",
            "length_p05": "3.0",
            "length_p95": "3.0",
            "height_mean": "2.5",
            "height_median": "2.5",
            "height_std": "0.0",
            "height_p05": "2.5",
            "height_p95": "2.5",
            "looks_constant_or_default": "true",
        },
    ]

    summary = build_final_class_priors(prior_csv_rows=rows)
    by_id = summary["classes_by_id"]
    flat = final_priors_to_rows(summary)

    assert by_id["0"]["robust_width"] == 0.7
    assert by_id["0"]["confidence_level"] == "high"
    assert by_id["1"]["confidence_level"] == "low"
    assert by_id["1"]["looks_constant"]
    assert len(flat) == 7


def test_class_priors_builder_marks_missing_class_as_fallback():
    summary = build_final_class_priors(prior_csv_rows=[])

    assert summary["classes_by_id"]["0"]["fallback_required"]
    assert summary["classes_by_id"]["0"]["confidence_level"] == "low"

