from deep_oc_sort_3d.audit3d.source_3d_audit import (
    audit_3d_sources,
    infer_3d_source_from_record,
    write_missing_source_metadata_report,
)


def test_source_3d_audit_infers_explicit_and_unknown():
    assert infer_3d_source_from_record({"source_3d": "depth_sampled"}) == "depth_sampled"
    assert infer_3d_source_from_record({"matched_gt": "true"}) == "gt_matched"
    assert infer_3d_source_from_record({"frame_id": 1}) == "unknown"


def test_source_3d_audit_writes_missing_metadata_report(tmp_path):
    records = tmp_path / "records.csv"
    records.write_text(
        "subset,frame_id,source_3d,matched_gt\n"
        "test,0,depth_sampled,false\n"
        "test,1,,false\n",
        encoding="utf-8",
    )

    report = audit_3d_sources(records, show_progress=False)
    output = tmp_path / "missing_source_metadata_report.md"
    write_missing_source_metadata_report(report, output)

    assert report["source_counts"]["depth_sampled"] == 1
    assert report["source_counts"]["unknown"] == 1
    assert "source_3d" not in report["missing_recommended_metadata_fields"]
    assert output.exists()
    assert "unknown" in output.read_text(encoding="utf-8")

