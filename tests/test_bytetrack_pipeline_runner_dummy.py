from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_pipeline_config import (
    configured_subsets,
    output_paths,
    selected_stages,
)


def test_stage_selection_and_output_paths():
    config = {
        "paths": {
            "output_local_tracks_root": "output/local_tracks/new",
            "output_tracklets_root": "output/tracklets/new",
        },
        "full_rerun": {
            "subsets": {
                "official_val": {"split": "val", "scenes": ["Warehouse_020"]},
                "test": {"split": "test", "scenes": ["Warehouse_023"]},
            }
        },
    }

    assert selected_stages("local_tracking") == ["local_tracking"]
    assert selected_stages(None)[0] == "precheck"
    assert configured_subsets(config, "full_rerun") == [
        ("official_val", "val", "Warehouse_020"),
        ("test", "test", "Warehouse_023"),
    ]
    paths = output_paths(config)
    assert str(paths["output_local_tracks_root"]).replace("\\", "/") == "output/local_tracks/new"
