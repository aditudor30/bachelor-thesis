"""Dummy tests for local tracking fragmentation audit."""

from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds
from deep_oc_sort_3d.fragmentation_audit.local_tracking_fragmentation_audit import audit_local_tracking


def test_local_tracking_fragmentation_audit_reads_csv(tmp_path):
    root = tmp_path / "local"
    path = root / "official_val" / "Warehouse_020" / "Camera_0000.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "subset,scene_name,camera_id,frame_id,local_track_id,class_name,matched_gt_object_id\n"
        "official_val,Warehouse_020,Camera_0000,0,1,Person,10\n"
        "official_val,Warehouse_020,Camera_0000,1,1,Person,10\n"
        "official_val,Warehouse_020,Camera_0000,2,2,Person,10\n",
        encoding="utf-8",
    )

    summary = audit_local_tracking(root, tmp_path / "summary.json", tmp_path / "diag", "dummy", FragmentationThresholds(), show_progress=False)

    assert summary["total_records"] == 3
    assert summary["num_tracks"] == 2
    assert summary["gt_fragmentation_approx"] == 1

