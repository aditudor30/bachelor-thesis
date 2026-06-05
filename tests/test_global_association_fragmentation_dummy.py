"""Dummy tests for global association fragmentation audit."""

from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds
from deep_oc_sort_3d.fragmentation_audit.global_association_fragmentation_audit import audit_global_association


def test_global_association_fragmentation_counts_tracks_and_edges(tmp_path):
    root = tmp_path / "global"
    scene_root = root / "official_val" / "Warehouse_020"
    scene_root.mkdir(parents=True)
    (scene_root / "global_tracks.csv").write_text(
        "subset,scene_name,class_name,global_track_id,num_candidates,num_cameras,duration,camera_ids_json,gt_purity\n"
        "official_val,Warehouse_020,Person,1,2,2,20,\"[\"\"Camera_0000\"\", \"\"Camera_0001\"\"]\",1.0\n"
        "official_val,Warehouse_020,Person,2,1,1,1,\"[\"\"Camera_0000\"\"]\",1.0\n",
        encoding="utf-8",
    )
    (scene_root / "association_edges.csv").write_text(
        "accepted,temporal_relation,reject_reason\n"
        "true,overlap,\n"
        "false,a_before_b,transition_cost_above_threshold\n",
        encoding="utf-8",
    )

    summary = audit_global_association(root, tmp_path / "summary.json", tmp_path / "diag", "dummy", FragmentationThresholds(), show_progress=False)

    assert summary["global_tracks"] == 2
    assert summary["multi_camera_tracks"] == 1
    assert summary["accepted_edges"] == 1
    assert summary["rejected_edges"] == 1

