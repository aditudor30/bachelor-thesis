from deep_oc_sort_3d.learned_association.threshold_selector import scorer_verdict


def test_scorer_verdict_ready_when_learned_model_safely_beats_reid():
    evaluations = {
        "reid_only_baseline": {"overall_metrics": {"pr_auc": 0.70}},
        "mlp_pairwise_scorer": {
            "overall_metrics": {"pr_auc": 0.80},
            "selected_thresholds": {
                "strict": {"precision": 0.97, "false_positive_rate": 0.01}
            },
        },
    }
    selection = {"selected_model": "mlp_pairwise_scorer"}
    config = {
        "selection": {"min_pr_auc_gain_over_reid": 0.02},
        "evaluation": {
            "strict_min_precision": 0.95,
            "max_false_positive_rate_strict": 0.02,
        },
    }

    result = scorer_verdict(evaluations, selection, config)

    assert result["verdict"] == "person_scorer_ready_for_conservative_association"
    assert result["ready_for_step_20c"] is True


def test_scorer_verdict_keeps_baseline_when_gain_is_missing():
    evaluations = {
        "reid_only_baseline": {"overall_metrics": {"pr_auc": 0.75}},
        "mlp_pairwise_scorer": {
            "overall_metrics": {"pr_auc": 0.76},
            "selected_thresholds": {
                "strict": {"precision": 0.97, "false_positive_rate": 0.01}
            },
        },
    }
    result = scorer_verdict(
        evaluations,
        {"selected_model": "mlp_pairwise_scorer"},
        {
            "selection": {"min_pr_auc_gain_over_reid": 0.02},
            "evaluation": {},
        },
    )

    assert result["verdict"] == "person_scorer_baseline_only"
