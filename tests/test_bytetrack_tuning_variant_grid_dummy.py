from pathlib import Path

from deep_oc_sort_3d.bytetrack_tuning.tuning_config import (
    build_variant_pipeline_config,
    variant_root,
)
from deep_oc_sort_3d.bytetrack_tuning.variant_grid import list_variants, validate_variant_grid


def _config(tmp_path):
    return {
        "bytetrack_coverage_tuning": {"output_root": str(tmp_path / "output")},
        "paths": {"dataset_root": "/tmp/data", "yolo_pipeline_root": "yolo", "v2_observations_root": "obs"},
        "subsets": {
            "sweep_eval": {"official_val": {"split": "val", "scenes": ["Warehouse_020"]}},
            "full_test": {"test": {"split": "test", "scenes": ["Warehouse_023"]}},
        },
        "tracking": {"class_agnostic_tracking": False, "allow_cross_class_matching": False},
        "variants": {
            "dense": {
                "track_high_thresh": 0.2,
                "track_low_thresh": 0.01,
                "new_track_thresh": 0.2,
                "match_thresh": 0.7,
                "second_stage_match_thresh": 0.35,
                "track_buffer": 90,
                "min_confidence_for_input": 0.001,
            }
        },
    }


def test_variant_grid_and_output_path(tmp_path):
    config = _config(tmp_path)
    assert list_variants(config) == ["dense"]
    assert validate_variant_grid(config) == []
    assert variant_root(config, "dense") == tmp_path / "output" / "sweep_runs" / "dense"

    resolved = build_variant_pipeline_config(config, "dense", include_full_test=True)
    assert resolved["bytetrack_style"]["track_buffer"] == 90
    assert Path(resolved["paths"]["output_local_tracks_root"]).parts[-3:] == ("sweep_runs", "dense", "local_tracks")
    assert "test" in resolved["full_rerun"]["subsets"]


def test_variant_grid_rejects_cross_class_tracking(tmp_path):
    config = _config(tmp_path)
    config["tracking"]["allow_cross_class_matching"] = True
    assert "allow_cross_class_matching must be false" in validate_variant_grid(config)
