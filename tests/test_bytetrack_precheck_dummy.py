from deep_oc_sort_3d.bytetrack_local_pipeline.bytetrack_precheck import (
    decide_precheck_verdict,
    precheck_allows_full_rerun,
)


def _row(name, median, short_ratio, fragmentation, switches, purity, nonperson, gt_records):
    return {
        "tracker_name": name,
        "status": "ok",
        "median_track_length": median,
        "short_track_ratio_le3": short_ratio,
        "approx_fragmentation": fragmentation,
        "approx_id_switches": switches,
        "local_purity_mean": purity,
        "nonperson_short_track_ratio_le3": nonperson,
        "false_merge_suspicion_rate": 0.05,
        "gt_matched_records": gt_records,
    }


def test_precheck_passes_large_safe_bytetrack_gain():
    rows = [
        _row("current_local_tracker", 6.0, 0.32, 2300, 4300, 0.964, 0.32, 60000),
        _row("bytetrack_style_yolo11m", 50.0, 0.08, 300, 450, 0.976, 0.08, 57000),
        _row("botsort_style_no_reid_yolo11m", 40.0, 0.10, 330, 470, 0.982, 0.08, 56500),
    ]

    verdict = decide_precheck_verdict(rows)

    assert verdict["label"] in (
        "bytetrack_precheck_pass_full_rerun_recommended",
        "bytetrack_precheck_pass_with_warnings",
    )
    assert precheck_allows_full_rerun(verdict)


def test_precheck_blocks_candidate_with_large_purity_drop():
    rows = [
        _row("current_local_tracker", 6.0, 0.32, 2300, 4300, 0.98, 0.20, 60000),
        _row("bytetrack_style_yolo11m", 50.0, 0.08, 300, 450, 0.90, 0.08, 57000),
    ]

    verdict = decide_precheck_verdict(rows)

    assert not precheck_allows_full_rerun(verdict)
    assert "purity_drop_above_0.01" in verdict["reasons"]
