from deep_oc_sort_3d.pipeline.pipeline_paths import (
    ensure_pipeline_dirs,
    get_detection_csv_path,
    get_mot_like_path,
    get_observation_jsonl_path,
    make_run_root,
)


def test_pipeline_paths_follow_expected_layout(tmp_path):
    run_root = make_run_root(tmp_path, "run_a")

    detection = get_detection_csv_path(run_root, "official_val", "Warehouse_020", "Camera_0000")
    mot_like = get_mot_like_path(run_root, "official_val", "Warehouse_020", "Camera_0000")
    observation = get_observation_jsonl_path(run_root, "official_val", "Warehouse_020", "Camera_0000")

    assert detection == run_root / "detections2d" / "official_val" / "Warehouse_020" / "Camera_0000.csv"
    assert mot_like == run_root / "mot_like" / "official_val" / "Warehouse_020" / "Camera_0000.txt"
    assert observation == run_root / "observations3d" / "official_val" / "Warehouse_020" / "Camera_0000.jsonl"


def test_ensure_pipeline_dirs_creates_top_level_dirs(tmp_path):
    run_root = tmp_path / "run_a"

    ensure_pipeline_dirs(run_root)

    assert (run_root / "detections2d").is_dir()
    assert (run_root / "mot_like").is_dir()
    assert (run_root / "observations3d").is_dir()
    assert (run_root / "summaries").is_dir()
    assert (run_root / "visualizations").is_dir()
