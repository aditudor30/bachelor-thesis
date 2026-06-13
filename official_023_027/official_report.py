"""Final reports for official Track1 candidates covering scenes 023-027."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.official_023_027.official_config import frozen_output_root, output_root
from deep_oc_sort_3d.official_023_027.official_figures import write_official_figures
from deep_oc_sort_3d.official_023_027.official_track1_io import read_json


OFFICIAL_CLASS_NAMES = {
    0: "Person",
    1: "Forklift",
    2: "NovaCarter",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
    6: "PalletTruck",
}


def write_official_reports(config: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, str]:
    """Write processing-side and frozen-candidate reports."""
    write_official_figures(config, comparison)
    scene_audit = read_json(output_root(config) / "audit" / "test_scene_audit.json")
    mapping_audit = read_json(output_root(config) / "audit" / "class_mapping_audit.json")
    readiness = comparison.get("upload_readiness", {})
    lines = _report_lines(config, comparison, scene_audit, mapping_audit, readiness)
    processing_report = output_root(config) / "comparison" / "OFFICIAL_023_027_V2_V3_REPORT.md"
    frozen_report = frozen_output_root(config) / "comparison" / "OFFICIAL_UPLOAD_CANDIDATES_023_027_REPORT.md"
    processing_report.parent.mkdir(parents=True, exist_ok=True)
    frozen_report.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    processing_report.write_text(content, encoding="utf-8")
    frozen_report.write_text(content, encoding="utf-8")
    return {"processing_report": str(processing_report), "frozen_report": str(frozen_report)}


def _report_lines(
    config: Dict[str, Any],
    comparison: Dict[str, Any],
    scene_audit: Dict[str, Any],
    mapping_audit: Dict[str, Any],
    readiness: Dict[str, Any],
) -> List[str]:
    candidates = comparison.get("candidates", {})
    v2 = candidates.get("v2_current", {})
    v3 = candidates.get("v3_gap_aware_soft", {})
    mode = v2.get("mode") or v3.get("mode") or config.get("official_023_027", {}).get("mode", "incremental")
    lines = [
        "# Official Track1 candidates for Warehouse_023-027",
        "",
        "## Executive summary",
        "",
        "V2 official 023-027 is the coverage-first official candidate.",
        "V3 official 023-027 is the identity-quality-first official candidate.",
        "Official evaluation is needed to determine which trade-off is better.",
        "",
        "- Processing mode: `%s`" % mode,
        "- Required scenes: `Warehouse_023`, `Warehouse_024`, `Warehouse_025`, `Warehouse_026`, `Warehouse_027`",
        "- Dataset scene audit: `%s` (%s/%s scenes valid)" % (scene_audit.get("status"), scene_audit.get("ok_scenes"), scene_audit.get("scene_count")),
        "- Class mapping audit: `%s`" % mapping_audit.get("status"),
        "- V2 included scene IDs: `%s`" % v2.get("scene_ids"),
        "- V3 included scene IDs: `%s`" % v3.get("scene_ids"),
        "- Internal mapping remains unchanged inside the pipeline.",
        "- Official remap applied only to final Track1: `0->0, 1->1, 2->6, 3->3, 4->4, 5->5, 6->2`.",
        "- Float fields are written with exactly two decimal places.",
        "- Legacy `output/frozen_upload_candidates/` files are read-only inputs and are not modified.",
        "",
        "## Candidate summary",
        "",
        "| Candidate | Rows | Unique tracks | Validation errors | Duplicate keys | NaN/inf | Non-positive dimensions | Rounding issues |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _candidate_table_row("V2 official", v2),
        _candidate_table_row("V3 official", v3),
        "",
        "## Per-scene distribution",
        "",
        "| Scene | V2 official | V3 official | Delta |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(_distribution_lines(v2.get("per_scene_rows", {}), v3.get("per_scene_rows", {})))
    lines.extend(["", "## Per-class official distribution", "", "| Official class | V2 official | V3 official | Delta |", "|---|---:|---:|---:|"])
    lines.extend(_class_distribution_lines(v2.get("per_class_rows", {}), v3.get("per_class_rows", {})))
    lines.extend(
        [
            "",
            "## Upload artifacts",
            "",
            _readiness_line("V2 official", readiness.get("v2_current_official", {})),
            _readiness_line("V3 official", readiness.get("v3_gap_aware_soft_official", {})),
            "",
            "The two candidates must be uploaded as separate submissions. Upload V2 official first, then V3 official.",
            "Do not combine both candidates into one zip file.",
            "",
            "## Verdict",
            "",
            "`%s`" % comparison.get("verdict", {}).get("label"),
        ]
    )
    return lines


def _candidate_table_row(name: str, value: Dict[str, Any]) -> str:
    return "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
        name,
        value.get("track1_rows"), value.get("unique_tracks"), value.get("validation_errors"), value.get("duplicate_keys"),
        value.get("nan_inf_count"), value.get("non_positive_dimensions"), value.get("rounding_issues"),
    )


def _distribution_lines(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    keys = sorted(set(list(left.keys()) + list(right.keys())), key=lambda value: int(value))
    output = []
    for key in keys:
        a = int(left.get(key, 0))
        b = int(right.get(key, 0))
        output.append("| %s | %d | %d | %d |" % (key, a, b, b - a))
    return output


def _class_distribution_lines(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    keys = sorted(set(list(left.keys()) + list(right.keys())), key=lambda value: int(value))
    output = []
    for key in keys:
        class_id = int(key)
        a = int(left.get(key, 0))
        b = int(right.get(key, 0))
        label = "%d %s" % (class_id, OFFICIAL_CLASS_NAMES.get(class_id, "unknown"))
        output.append("| %s | %d | %d | %d |" % (label, a, b, b - a))
    return output


def _readiness_line(name: str, value: Dict[str, Any]) -> str:
    verification = value.get("zip_verification", {})
    return "- %s: ready=`%s`, Track1=`%s`, zip=`%s`, zip_size_mb=`%s`, zip_entries=`%s`, zip_lines=`%s`, track1_SHA256=`%s`, zip_SHA256=`%s`" % (
        name, value.get("ready"), value.get("track1_path"), value.get("zip_path"), value.get("zip_size_mb"),
        verification.get("names"), verification.get("line_count"), value.get("track1_sha256"), value.get("zip_sha256")
    )
