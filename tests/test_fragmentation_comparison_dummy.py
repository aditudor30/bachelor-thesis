"""Dummy tests for fragmentation comparison and root cause."""

from deep_oc_sort_3d.fragmentation_audit.fragmentation_comparison import build_full_comparison, build_stage_comparison
from deep_oc_sort_3d.fragmentation_audit.fragmentation_root_cause import analyze_root_cause


def test_fragmentation_comparison_deltas():
    comp = build_stage_comparison(
        "local_tracking",
        {"num_tracks": 10, "short_ratio": 0.1},
        {"num_tracks": 15, "short_ratio": 0.3},
    )

    assert comp["deltas"]["num_tracks"] == 5.0
    assert comp["deltas"]["short_ratio"] == 0.19999999999999998


def test_fragmentation_root_cause_prefers_local_when_local_grows():
    comparison = build_full_comparison(
        {
            "local_tracking": {
                "baseline_v1": {"num_tracks": 10, "short_ratio": 0.1},
                "baseline_v2": {"num_tracks": 30, "short_ratio": 0.4},
            },
            "global_association": {
                "baseline_v1": {"global_tracks": 10, "fragmentation_approx": 1},
                "baseline_v2": {"global_tracks": 10, "fragmentation_approx": 1},
            },
        }
    )
    cause = analyze_root_cause(comparison)

    assert cause["verdict"] in ("local_tracking_dominant", "mixed_causes")
    assert cause["tuning_recommendations"]

