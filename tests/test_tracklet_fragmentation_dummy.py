"""Dummy tests for tracklet fragmentation audit."""

from deep_oc_sort_3d.fragmentation_audit.fragmentation_types import FragmentationThresholds
from deep_oc_sort_3d.fragmentation_audit.tracklet_fragmentation_audit import audit_tracklets


def test_tracklet_fragmentation_audit_quality_counts(tmp_path):
    root = tmp_path / "tracklets"
    path = root / "official_val" / "Warehouse_020" / "Camera_0000.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "subset,scene_name,camera_id,class_name,local_track_id,length,is_valid_for_mtmc,quality_flag,gt_purity\n"
        "official_val,Warehouse_020,Camera_0000,Person,1,10,true,good,1.0\n"
        "official_val,Warehouse_020,Camera_0000,Person,2,1,false,short,0.0\n",
        encoding="utf-8",
    )

    summary = audit_tracklets(root, tmp_path / "summary.json", tmp_path / "diag", "dummy", FragmentationThresholds(), show_progress=False)

    assert summary["total_tracklets"] == 2
    assert summary["valid_tracklets"] == 1
    assert summary["quality_flags"]["short"] == 1

