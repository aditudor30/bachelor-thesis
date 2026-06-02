import yaml

from deep_oc_sort_3d.scripts.check_mvp_outputs import (
    load_mvp_baseline_config,
    validate_mvp_baseline_config,
)


def test_mvp_baseline_config_validation_dummy(tmp_path):
    config_path = tmp_path / "mvp.yaml"
    data = {
        "mvp": {
            "paths": {
                "pipeline_run_root": "output/pipeline",
                "local_tracks_root": "output/local_tracks",
                "tracklets_root": "output/tracklets",
                "mtmc_candidates_root": "output/candidates",
                "motion_clean_candidates_root": "output/candidates_clean",
                "global_mtmc_root": "output/global",
                "final_export_root": "output/final",
            },
            "subsets": {
                "test": {
                    "split": "test",
                    "scenes": ["Warehouse_023"],
                },
            },
            "classes": {0: "Person"},
            "final_export": {
                "official_track1_export": "todo_until_schema_confirmed",
            },
        },
    }
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    config = load_mvp_baseline_config(config_path)
    report = validate_mvp_baseline_config(config)

    assert report["valid"]
    assert report["errors"] == []


def test_mvp_baseline_config_validation_reports_missing_paths():
    report = validate_mvp_baseline_config({"paths": {}, "subsets": {"test": {"scenes": []}}})

    assert not report["valid"]
    assert "missing_path_key:pipeline_run_root" in report["errors"]
