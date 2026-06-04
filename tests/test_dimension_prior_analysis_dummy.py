from deep_oc_sort_3d.priors3d.dimension_prior_analysis import compare_priors_to_generic_rows


def test_dimension_prior_analysis_computes_ratios_and_warnings():
    priors = {
        "classes": [
            {
                "class_id": 0,
                "class_name": "Person",
                "robust_width": 1.0,
                "robust_length": 2.0,
                "robust_height": 4.0,
            }
        ]
    }
    rows = [
        {"subset": "test", "class_id": 0, "width_3d": 1.0, "length_3d": 2.0, "height_3d": 4.0},
        {"subset": "test", "class_id": 0, "width_3d": 4.0, "length_3d": 8.0, "height_3d": 16.0},
    ]

    summary = compare_priors_to_generic_rows(
        priors,
        rows,
        {
            "compare_subsets": ["test"],
            "dimension_ratio_warning_low": 0.5,
            "dimension_ratio_warning_high": 2.0,
        },
    )
    row = summary["rows"][0]

    assert row["width_median_ratio_to_prior"] == 2.5
    assert row["records_close_to_prior"] == 1
    assert row["records_extreme"] == 1
    assert "width_median_ratio_outside_range" in row["warnings"]

