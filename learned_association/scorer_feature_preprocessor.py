"""Train-fitted feature selection, imputation, scaling and encoding."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from deep_oc_sort_3d.learned_association.pair_dataset_io import safe_float


RECOMMENDED_NUMERIC_FEATURES = (
    "reid_similarity",
    "reid_distance",
    "temporal_gap",
    "temporal_overlap",
    "duration_a",
    "duration_b",
    "duration_ratio",
    "same_camera",
    "cross_camera",
    "camera_pair_seen_in_train",
    "start_distance_3d",
    "end_distance_3d",
    "min_endpoint_distance_3d",
    "center_mean_distance_3d",
    "spatial_distance_xy",
    "spatial_distance_z",
    "velocity_cosine",
    "velocity_difference",
    "speed_difference",
    "expected_position_error",
    "motion_consistency_score",
    "num_obs_a",
    "num_obs_b",
    "mean_conf_a",
    "mean_conf_b",
    "min_conf_a",
    "min_conf_b",
    "bbox_area_mean_a",
    "bbox_area_mean_b",
    "bbox_height_mean_a",
    "bbox_height_mean_b",
    "gt_purity_a",
    "gt_purity_b",
    "same_camera_temporal_conflict",
    "temporal_overlap_conflict",
    "large_spatial_gap_flag",
    "large_temporal_gap_flag",
    "low_quality_fragment_flag",
    "missing_reid_flag",
    "missing_geometry_flag",
)

DEFAULT_CATEGORICAL_FEATURES = (
    "camera_pair",
    "temporal_order",
    "fragment_quality_a",
    "fragment_quality_b",
)

NON_SCALED_FEATURES = {
    "same_camera",
    "cross_camera",
    "camera_pair_seen_in_train",
    "same_camera_temporal_conflict",
    "temporal_overlap_conflict",
    "large_spatial_gap_flag",
    "large_temporal_gap_flag",
    "low_quality_fragment_flag",
    "missing_reid_flag",
    "missing_geometry_flag",
}


class PairFeaturePreprocessor:
    """Fit preprocessing only on train and transform train/val identically."""

    def __init__(
        self,
        missing_value_strategy: str = "median",
        scale_continuous_features: bool = True,
        camera_pair_encoding: str = "one_hot",
        enabled_numeric_features: Optional[Sequence[str]] = None,
        enabled_categorical_features: Optional[Sequence[str]] = None,
    ) -> None:
        self.missing_value_strategy = str(missing_value_strategy)
        self.scale_continuous_features = bool(scale_continuous_features)
        self.camera_pair_encoding = str(camera_pair_encoding)
        self.enabled_numeric_features = list(
            RECOMMENDED_NUMERIC_FEATURES
            if enabled_numeric_features is None
            else enabled_numeric_features
        )
        self.enabled_categorical_features = list(
            DEFAULT_CATEGORICAL_FEATURES
            if enabled_categorical_features is None
            else enabled_categorical_features
        )
        self.numeric_features = []  # type: List[str]
        self.categorical_features = []  # type: List[str]
        self.category_values = {}  # type: Dict[str, List[str]]
        self.fill_values = {}  # type: Dict[str, float]
        self.means = {}  # type: Dict[str, float]
        self.scales = {}  # type: Dict[str, float]
        self.output_features = []  # type: List[str]
        self.fitted = False

    def fit(self, rows: Sequence[Dict[str, Any]]) -> "PairFeaturePreprocessor":
        """Select available columns and fit train-only preprocessing state."""
        if not rows:
            raise ValueError("Cannot fit feature preprocessing on an empty dataset")
        self.numeric_features = [
            name for name in self.enabled_numeric_features if _has_any_numeric(rows, name)
        ]
        configured_categorical = list(self.enabled_categorical_features)
        if self.camera_pair_encoding.lower() == "none" and "camera_pair" in configured_categorical:
            configured_categorical.remove("camera_pair")
        self.categorical_features = [
            name for name in configured_categorical if any(str(row.get(name) or "") for row in rows)
        ]

        for feature in self.numeric_features:
            values = [safe_float(row.get(feature)) for row in rows]
            valid = np.asarray([value for value in values if value is not None], dtype=np.float64)
            if self.missing_value_strategy == "zero" or valid.size == 0:
                fill_value = 0.0
            else:
                fill_value = float(np.median(valid))
            filled = np.asarray(
                [fill_value if value is None else value for value in values], dtype=np.float64
            )
            mean = float(np.mean(filled)) if filled.size else 0.0
            scale = float(np.std(filled)) if filled.size else 1.0
            if scale < 1e-8:
                scale = 1.0
            self.fill_values[feature] = fill_value
            self.means[feature] = mean
            self.scales[feature] = scale

        for feature in self.categorical_features:
            values = sorted({str(row.get(feature) or "unknown") for row in rows})
            self.category_values[feature] = values

        self.output_features = list(self.numeric_features)
        for feature in self.categorical_features:
            self.output_features.extend(
                ["%s=%s" % (feature, value) for value in self.category_values[feature]]
            )
        self.fitted = True
        return self

    def transform(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Transform rows into a float32 model matrix."""
        if not self.fitted:
            raise RuntimeError("PairFeaturePreprocessor must be fitted before transform")
        matrix = np.zeros((len(rows), len(self.output_features)), dtype=np.float32)
        for row_index, row in enumerate(rows):
            column_index = 0
            for feature in self.numeric_features:
                value = safe_float(row.get(feature), self.fill_values[feature])
                if value is None:
                    value = self.fill_values[feature]
                if self.scale_continuous_features and feature not in NON_SCALED_FEATURES:
                    value = (value - self.means[feature]) / self.scales[feature]
                matrix[row_index, column_index] = float(value)
                column_index += 1
            for feature in self.categorical_features:
                current = str(row.get(feature) or "unknown")
                for category in self.category_values[feature]:
                    matrix[row_index, column_index] = 1.0 if current == category else 0.0
                    column_index += 1
        return matrix

    def fit_transform(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Fit and transform the training rows."""
        self.fit(rows)
        return self.transform(rows)

    def summary(self) -> Dict[str, Any]:
        """Return JSON-serializable preprocessing provenance."""
        return {
            "missing_value_strategy": self.missing_value_strategy,
            "scale_continuous_features": self.scale_continuous_features,
            "camera_pair_encoding": self.camera_pair_encoding,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "category_values": self.category_values,
            "fill_values": self.fill_values,
            "means": self.means,
            "scales": self.scales,
            "output_features": self.output_features,
            "input_dim": len(self.output_features),
        }


def build_preprocessor_from_config(config: Dict[str, Any]) -> PairFeaturePreprocessor:
    """Construct a preprocessor from the scorer feature config."""
    settings = config.get("features", {})
    groups = {
        "use_reid": ["reid_similarity", "reid_distance"],
        "use_temporal": [
            "temporal_gap",
            "temporal_overlap",
            "duration_a",
            "duration_b",
            "duration_ratio",
            "same_camera",
            "cross_camera",
            "camera_pair_seen_in_train",
        ],
        "use_geometry": [
            "start_distance_3d",
            "end_distance_3d",
            "min_endpoint_distance_3d",
            "center_mean_distance_3d",
            "spatial_distance_xy",
            "spatial_distance_z",
        ],
        "use_motion": [
            "velocity_cosine",
            "velocity_difference",
            "speed_difference",
            "expected_position_error",
            "motion_consistency_score",
        ],
        "use_quality": [
            "num_obs_a",
            "num_obs_b",
            "mean_conf_a",
            "mean_conf_b",
            "min_conf_a",
            "min_conf_b",
            "bbox_area_mean_a",
            "bbox_area_mean_b",
            "bbox_height_mean_a",
            "bbox_height_mean_b",
            "gt_purity_a",
            "gt_purity_b",
        ],
        "use_conflict_flags": [
            "same_camera_temporal_conflict",
            "temporal_overlap_conflict",
            "large_spatial_gap_flag",
            "large_temporal_gap_flag",
            "low_quality_fragment_flag",
            "missing_reid_flag",
            "missing_geometry_flag",
        ],
    }
    enabled_numeric = []  # type: List[str]
    for setting_name, feature_names in groups.items():
        if bool(settings.get(setting_name, True)):
            enabled_numeric.extend(feature_names)
    enabled_categorical = []  # type: List[str]
    if bool(settings.get("use_camera_pair", True)):
        enabled_categorical.append("camera_pair")
    if bool(settings.get("use_temporal", True)):
        enabled_categorical.append("temporal_order")
    if bool(settings.get("use_quality", True)):
        enabled_categorical.extend(["fragment_quality_a", "fragment_quality_b"])
    return PairFeaturePreprocessor(
        missing_value_strategy=str(settings.get("missing_value_strategy", "median")),
        scale_continuous_features=bool(settings.get("scale_continuous_features", True)),
        camera_pair_encoding=str(settings.get("camera_pair_encoding", "one_hot")),
        enabled_numeric_features=enabled_numeric,
        enabled_categorical_features=enabled_categorical,
    )


def _has_any_numeric(rows: Sequence[Dict[str, Any]], feature: str) -> bool:
    return any(safe_float(row.get(feature)) is not None for row in rows)
