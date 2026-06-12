from deep_oc_sort_3d.learned_association.pair_balancer import balance_pairs


def test_balancer_keeps_positives_and_limits_negatives():
    positives = [
        {"pair_id": "p%d" % index, "same_identity": 1, "embedding_valid_pair": 1}
        for index in range(4)
    ]
    negatives = [
        {
            "pair_id": "n%d" % index,
            "same_identity": 0,
            "embedding_valid_pair": 1,
            "hard_negative": int(index < 3),
            "reid_similarity": 0.9 - index * 0.01,
        }
        for index in range(20)
    ]

    balanced = balance_pairs(positives + negatives, 2.0, 42, True)

    assert sum(row["same_identity"] == 1 for row in balanced) == 4
    assert sum(row["same_identity"] == 0 for row in balanced) == 8
    assert {"n0", "n1", "n2"}.issubset({row["pair_id"] for row in balanced})
