from deep_oc_sort_3d.pipeline.run_config import load_pipeline_config, update_config, validate_pipeline_config


def test_load_pipeline_config_and_validate(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
pipeline:
  root: "/tmp/dataset"
  output_root: "output/pipeline_runs"
  run_name: "debug"
  detector_model: "model.pt"
  conf_threshold: 0.01
  imgsz: 1280
  device: "0"
  frame_stride: 1
  max_frames: 100
  camera_ids:
    - Camera_0000
  export_mot_like: true
  build_observations: true
  iou_threshold: 0.3
  depth_sampling_method: "center_median"
  class_must_match: true
subsets:
  official_val:
    split: "val"
    scenes:
      - Warehouse_020
""",
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)
    messages = validate_pipeline_config(config)

    assert messages == []
    assert config.run_name == "debug"
    assert config.scenes_by_subset["official_val"] == ["Warehouse_020"]
    assert config.camera_ids == ["Camera_0000"]


def test_update_config_simple_overrides(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
pipeline:
  root: "/tmp/dataset"
  output_root: "output/pipeline_runs"
  run_name: "debug"
  detector_model: "model.pt"
subsets:
  official_val:
    split: "val"
    scenes:
      - Warehouse_020
  test:
    split: "test"
    scenes:
      - Warehouse_023
""",
        encoding="utf-8",
    )
    config = load_pipeline_config(config_path)

    updated = update_config(config, run_name="debug2", camera_ids=["Camera_0001"], subsets=["test"])

    assert updated.run_name == "debug2"
    assert updated.camera_ids == ["Camera_0001"]
    assert list(updated.scenes_by_subset.keys()) == ["test"]
