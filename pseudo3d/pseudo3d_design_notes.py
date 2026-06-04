"""Human-readable pseudo-3D design notes for Step 15B."""


PSEUDO3D_DESIGN_SCOPE = (
    "Step 15B defines priors, metadata, config, and API contracts. "
    "It does not implement or activate the pseudo-3D estimator."
)


PSEUDO3D_ALLOWED_TEST_INPUTS = [
    "rgb_frame",
    "bbox_2d",
    "class_id",
    "calibration",
    "class_priors",
    "optional_track_history",
]


PSEUDO3D_FORBIDDEN_TEST_INPUTS = ["ground_truth", "depth_map"]

