import pytest

from deep_oc_sort_3d.audit3d.class_3d_priors import (
    class_priors_to_rows,
    compare_class_priors_between_subsets,
    compute_class_dimension_priors,
)


def test_class_3d_priors_compute_stats_and_constant_flag():
    rows = [
        {"class_id": 0, "class_name": "Person", "width_3d": 1.0, "length_3d": 2.0, "height_3d": 3.0, "yaw": 0.0},
        {"class_id": 0, "class_name": "Person", "width_3d": 3.0, "length_3d": 2.0, "height_3d": 3.0, "yaw": 0.1},
        {"class_id": 0, "class_name": "Person", "width_3d": 5.0, "length_3d": 2.0, "height_3d": 3.0, "yaw": 0.2},
        {"class_id": 1, "class_name": "Forklift", "width_3d": 4.0, "length_3d": 5.0, "height_3d": 6.0, "yaw": 0.0},
        {"class_id": 1, "class_name": "Forklift", "width_3d": 4.0, "length_3d": 5.0, "height_3d": 6.0, "yaw": 0.0},
    ]

    priors = compute_class_dimension_priors(rows)
    csv_rows = class_priors_to_rows(priors)

    assert priors["classes"]["0"]["dimensions"]["width"]["median"] == 3.0
    assert priors["classes"]["0"]["dimensions"]["width"]["mean"] == 3.0
    assert priors["classes"]["0"]["dimensions"]["width"]["p05"] == pytest.approx(1.2)
    assert priors["classes"]["0"]["dimensions"]["width"]["p95"] == pytest.approx(4.8)
    assert priors["classes"]["1"]["looks_constant_or_default"]
    assert len(csv_rows) == 2


def test_class_3d_priors_compare_subsets():
    priors_a = compute_class_dimension_priors(
        [{"class_id": 0, "width_3d": 1.0, "length_3d": 2.0, "height_3d": 3.0}]
    )
    priors_b = compute_class_dimension_priors(
        [{"class_id": 0, "width_3d": 2.0, "length_3d": 3.0, "height_3d": 4.0}]
    )

    comparison = compare_class_priors_between_subsets(priors_a, priors_b)

    assert comparison["rows"][0]["width_median_delta"] == 1.0
