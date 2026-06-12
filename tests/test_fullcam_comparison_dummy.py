"""Dummy tests for fullcam baseline comparison."""

import json

from deep_oc_sort_3d.baseline_v2_fullcam.fullcam_comparison import (
    compare_fullcam_from_config,
    compute_metric_deltas,
    decide_fullcam_verdict,
)


def test_fullcam_metric_deltas_and_verdict():
    v1 = {
        "track1": {"rows": 10, "validation_errors": 0},
        "observations": {"pseudo3d_used_rate": None, "fallback_original_used_rate": None},
        "global_association": {"global_tracks": 5, "multi_camera_tracks": 2, "global_purity_mean": 0.95, "false_merge_rate": 0.1, "fragmentation_approx": 4},
    }
    v2 = {
        "track1": {"rows": 12, "validation_errors": 0},
        "observations": {"pseudo3d_used_rate": 0.98, "fallback_original_used_rate": 0.01, "metadata_complete_rate": 0.99},
        "global_association": {"global_tracks": 6, "multi_camera_tracks": 3, "global_purity_mean": 0.95, "false_merge_rate": 0.1, "fragmentation_approx": 4},
    }
    deltas = compute_metric_deltas(v1, v2)
    summary = {"baseline_v2_fullcam": v2, "deltas": deltas}
    verdict = decide_fullcam_verdict(summary)

    assert deltas["track1_rows_delta"] == 2.0
    assert verdict["label"] == "baseline_v2_fullcam_ready_for_submission"


def test_fullcam_compare_reads_dummy_outputs(tmp_path):
    v1_track1 = tmp_path / "v1_track1"
    v2_track1 = tmp_path / "v2_track1"
    v2_pipeline = tmp_path / "v2_pipeline"
    v1_track1.mkdir(parents=True)
    v2_track1.mkdir(parents=True)
    (v1_track1 / "track1.txt").write_text("23 0 1 0 1 2 3 1 1 1 0\n", encoding="utf-8")
    (v2_track1 / "track1.txt").write_text("23 0 1 0 1 2 3 1 1 1 0\n", encoding="utf-8")
    (v2_track1 / "validation").mkdir(parents=True)
    (v2_track1 / "validation" / "track1_validation_summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "num_errors": 0,
                "checks": {"duplicate_key_count": 0, "sorting_issues": 0},
                "distribution": {
                    "per_scene_rows": {"23": 1},
                    "per_class_rows": {"0": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    (v2_pipeline / "summaries").mkdir(parents=True)
    (v2_pipeline / "summaries" / "pseudo3d_observation_summary.json").write_text(
        json.dumps({"output_observations": 1, "pseudo3d_used": 1, "pseudo3d_used_rate": 1.0}),
        encoding="utf-8",
    )

    config = {
        "paths": {
            "baseline_v1_track1_root": str(v1_track1),
            "output_track1_root": str(v2_track1),
            "output_pipeline_root": str(v2_pipeline),
        }
    }
    summary = compare_fullcam_from_config(config)

    assert summary["baseline_v1"]["track1"]["rows"] == 1
    assert summary["baseline_v2_fullcam"]["observations"]["pseudo3d_used_rate"] == 1.0
    assert summary["baseline_v2_fullcam"]["track1"]["per_scene_rows"] == {"23": 1}
    assert summary["baseline_v2_fullcam"]["track1"]["per_class_rows"] == {"0": 1}
