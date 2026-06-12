"""Conservative threshold and model selection for Step 20B."""

from typing import Any, Dict, List, Optional, Sequence


def select_threshold_labels(
    sweep_rows: Sequence[Dict[str, Any]], evaluation_config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Select exploratory, balanced, strict and very-strict thresholds."""
    rows = list(sweep_rows)
    if not rows:
        return {}
    strict_precision = float(evaluation_config.get("strict_min_precision", 0.95))
    very_strict_precision = float(evaluation_config.get("very_strict_min_precision", 0.98))
    max_fpr = float(evaluation_config.get("max_false_positive_rate_strict", 0.02))
    exploratory = max(rows, key=lambda row: _value(row.get("recall"), -1.0))
    balanced = max(rows, key=lambda row: _value(row.get("f1"), -1.0))
    strict = _strict_choice(rows, strict_precision, max_fpr)
    very_strict = _strict_choice(rows, very_strict_precision, min(max_fpr, 0.01))
    return {
        "exploratory": _compact(exploratory),
        "balanced": _compact(balanced),
        "strict": _compact(strict),
        "very_strict": _compact(very_strict),
    }


def select_model_conservatively(
    model_evaluations: Dict[str, Dict[str, Any]],
    selection_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Select a model by strict precision, FPR, hard negatives, then PR-AUC."""
    if not model_evaluations:
        return {"selected_model": None, "reason": "no_models_evaluated"}
    reid = model_evaluations.get("reid_only_baseline", {})
    reid_pr_auc = _value(reid.get("overall_metrics", {}).get("pr_auc"), 0.0)
    min_gain = float(selection_config.get("min_pr_auc_gain_over_reid", 0.02))
    max_hard_fpr = float(selection_config.get("max_allowed_hard_negative_fpr", 0.02))
    require_gain = bool(selection_config.get("require_better_than_reid_only", True))

    ranked = []  # type: List[Any]
    for model_name, evaluation in model_evaluations.items():
        strict = evaluation.get("selected_thresholds", {}).get("strict", {})
        hard_fpr = evaluation.get("hard_negative_metrics", {}).get(
            "hard_negative_false_positive_rate"
        )
        pr_auc = _value(evaluation.get("overall_metrics", {}).get("pr_auc"), 0.0)
        precision = _value(strict.get("precision"), 0.0)
        fpr = _value(strict.get("false_positive_rate"), 1.0)
        recall = _value(strict.get("recall"), 0.0)
        hard_value = _value(hard_fpr, 1.0)
        eligible = hard_value <= max_hard_fpr
        if model_name != "reid_only_baseline" and require_gain:
            eligible = eligible and pr_auc >= reid_pr_auc + min_gain
        rank = (
            1 if eligible else 0,
            precision,
            -fpr,
            -hard_value,
            pr_auc,
            recall,
        )
        ranked.append((rank, model_name, evaluation))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if require_gain:
        learned_eligible = [
            item
            for item in ranked
            if item[1] != "reid_only_baseline" and item[0][0] == 1
        ]
        if not learned_eligible and "reid_only_baseline" in model_evaluations:
            selected_name = "reid_only_baseline"
            selected_evaluation = model_evaluations[selected_name]
            return {
                "selected_model": selected_name,
                "selected_thresholds": selected_evaluation.get("selected_thresholds", {}),
                "selection_metrics": selected_evaluation.get("overall_metrics", {}),
                "hard_negative_metrics": selected_evaluation.get("hard_negative_metrics", {}),
                "reason": "no_learned_model_met_conservative_gain_and_hard_negative_criteria",
                "reid_pr_auc": reid_pr_auc,
            }
    _, selected_name, selected_evaluation = ranked[0]
    return {
        "selected_model": selected_name,
        "selected_thresholds": selected_evaluation.get("selected_thresholds", {}),
        "selection_metrics": selected_evaluation.get("overall_metrics", {}),
        "hard_negative_metrics": selected_evaluation.get("hard_negative_metrics", {}),
        "reason": "conservative_precision_fpr_hard_negative_pr_auc_ranking",
        "reid_pr_auc": reid_pr_auc,
    }


def scorer_verdict(
    model_evaluations: Dict[str, Dict[str, Any]],
    selection: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return an honest readiness verdict for Step 20C."""
    selected_name = selection.get("selected_model")
    if selected_name is None:
        return {
            "verdict": "person_scorer_invalid_fix_required",
            "ready_for_step_20c": False,
            "reasons": ["no_model_selected"],
        }
    selected = model_evaluations.get(str(selected_name), {})
    strict = selected.get("selected_thresholds", {}).get("strict", {})
    precision = _value(strict.get("precision"), 0.0)
    fpr = _value(strict.get("false_positive_rate"), 1.0)
    reid_pr = _value(
        model_evaluations.get("reid_only_baseline", {}).get("overall_metrics", {}).get("pr_auc"),
        0.0,
    )
    selected_pr = _value(selected.get("overall_metrics", {}).get("pr_auc"), 0.0)
    min_gain = float(config.get("selection", {}).get("min_pr_auc_gain_over_reid", 0.02))
    reasons = ["limited_train_scene_coverage_Warehouse_014_to_016"]
    if selected_name == "reid_only_baseline" or selected_pr < reid_pr + min_gain:
        return {
            "verdict": "person_scorer_baseline_only",
            "ready_for_step_20c": True,
            "selected_model": selected_name,
            "reasons": reasons + ["learned_models_do_not_safely_beat_reid_baseline"],
        }
    strict_min = float(config.get("evaluation", {}).get("strict_min_precision", 0.95))
    max_fpr = float(config.get("evaluation", {}).get("max_false_positive_rate_strict", 0.02))
    if precision >= strict_min and fpr <= max_fpr:
        return {
            "verdict": "person_scorer_ready_for_conservative_association",
            "ready_for_step_20c": True,
            "selected_model": selected_name,
            "reasons": reasons,
        }
    return {
        "verdict": "person_scorer_promising_needs_threshold_tuning",
        "ready_for_step_20c": True,
        "selected_model": selected_name,
        "reasons": reasons + ["strict_precision_or_fpr_target_not_met"],
    }


def _strict_choice(
    rows: Sequence[Dict[str, Any]], min_precision: float, max_fpr: float
) -> Dict[str, Any]:
    eligible = [
        row
        for row in rows
        if _value(row.get("precision"), 0.0) >= min_precision
        and _value(row.get("false_positive_rate"), 1.0) <= max_fpr
    ]
    if eligible:
        return max(
            eligible,
            key=lambda row: (
                _value(row.get("recall"), 0.0),
                _value(row.get("precision"), 0.0),
                -float(row.get("threshold", 0.0)),
            ),
        )
    return max(
        rows,
        key=lambda row: (
            _value(row.get("precision"), 0.0),
            -_value(row.get("false_positive_rate"), 1.0),
            _value(row.get("recall"), 0.0),
        ),
    )


def _compact(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "tp",
        "fp",
        "tn",
        "fn",
    )
    return {key: row.get(key) for key in keys}


def _value(value: Optional[Any], default: float) -> float:
    return default if value is None else float(value)
