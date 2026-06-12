from deep_oc_sort_3d.learned_association.threshold_selector import select_threshold_labels


def test_threshold_selector_prefers_high_precision_low_fpr_for_strict():
    rows = [
        {
            "threshold": 0.5,
            "precision": 0.80,
            "recall": 0.95,
            "f1": 0.87,
            "false_positive_rate": 0.10,
            "false_negative_rate": 0.05,
            "tp": 95,
            "fp": 24,
            "tn": 216,
            "fn": 5,
        },
        {
            "threshold": 0.8,
            "precision": 0.97,
            "recall": 0.70,
            "f1": 0.81,
            "false_positive_rate": 0.01,
            "false_negative_rate": 0.30,
            "tp": 70,
            "fp": 2,
            "tn": 238,
            "fn": 30,
        },
        {
            "threshold": 0.9,
            "precision": 0.99,
            "recall": 0.30,
            "f1": 0.46,
            "false_positive_rate": 0.001,
            "false_negative_rate": 0.70,
            "tp": 30,
            "fp": 1,
            "tn": 239,
            "fn": 70,
        },
    ]

    selected = select_threshold_labels(
        rows,
        {
            "strict_min_precision": 0.95,
            "very_strict_min_precision": 0.98,
            "max_false_positive_rate_strict": 0.02,
        },
    )

    assert selected["strict"]["threshold"] == 0.8
    assert selected["very_strict"]["threshold"] == 0.9
