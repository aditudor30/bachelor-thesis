"""Optional sklearn scorers for learned Person association."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from deep_oc_sort_3d.learned_association.scorer_io import save_pickle


def sklearn_available() -> bool:
    """Return whether sklearn can be imported."""
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


def train_sklearn_scorers(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Train enabled classical scorers, returning models and warnings."""
    models = {}  # type: Dict[str, Any]
    warnings = {}  # type: Dict[str, str]
    settings = config.get("models", {})
    if not sklearn_available():
        message = "sklearn is unavailable; classical scorers were skipped"
        for name in ("logistic_regression_l2", "gradient_boosting", "random_forest"):
            warnings[name] = message
        return models, warnings

    if bool(settings.get("train_logistic_regression", True)):
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                penalty="l2",
                C=float(settings.get("logistic_c", 1.0)),
                class_weight=settings.get("class_weight", "balanced"),
                max_iter=int(settings.get("logistic_max_iter", 1000)),
                random_state=int(config.get("person_association_scorer", {}).get("random_seed", 42)),
            )
            model.fit(train_x, train_y.astype(np.int64))
            models["logistic_regression_l2"] = model
            save_pickle(output_dir / "logistic_regression_l2.pkl", model)
        except Exception as exc:
            warnings["logistic_regression_l2"] = str(exc)

    if bool(settings.get("train_gradient_boosting", True)):
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=int(settings.get("gradient_boosting_estimators", 150)),
                learning_rate=float(settings.get("gradient_boosting_learning_rate", 0.05)),
                max_depth=int(settings.get("gradient_boosting_max_depth", 3)),
                random_state=int(config.get("person_association_scorer", {}).get("random_seed", 42)),
            )
            model.fit(train_x, train_y.astype(np.int64))
            models["gradient_boosting"] = model
            save_pickle(output_dir / "gradient_boosting.pkl", model)
        except Exception as exc:
            warnings["gradient_boosting"] = str(exc)

    if bool(settings.get("train_random_forest", True)):
        try:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=int(settings.get("random_forest_estimators", 200)),
                max_depth=_optional_int(settings.get("random_forest_max_depth", 12)),
                min_samples_leaf=int(settings.get("random_forest_min_samples_leaf", 2)),
                class_weight=settings.get("class_weight", "balanced"),
                n_jobs=int(settings.get("random_forest_n_jobs", -1)),
                random_state=int(config.get("person_association_scorer", {}).get("random_seed", 42)),
            )
            model.fit(train_x, train_y.astype(np.int64))
            models["random_forest"] = model
            save_pickle(output_dir / "random_forest.pkl", model)
        except Exception as exc:
            warnings["random_forest"] = str(exc)
    return models, warnings


def sklearn_probability_scores(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities from an sklearn classifier."""
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(matrix)
        return np.asarray(values[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        logits = np.asarray(model.decision_function(matrix), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    return np.asarray(model.predict(matrix), dtype=np.float64)


def model_feature_importance(model: Any, feature_names: Any) -> Dict[str, float]:
    """Extract coefficients or feature importance when available."""
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64).reshape(-1)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=np.float64)).reshape(-1)
    if values is None or len(values) != len(feature_names):
        return {}
    return {str(name): float(value) for name, value in zip(feature_names, values)}


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, "", "none", "None"):
        return None
    return int(value)
