"""Human-readable Step 21F report generation."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.freeze_upload.freeze_config import output_root
from deep_oc_sort_3d.freeze_upload.freeze_figures import write_freeze_figures


def write_freeze_report(config: Dict[str, Any], comparison: Dict[str, Any]) -> Path:
    """Write the requested report and local diagnostic figures."""
    figures = write_freeze_figures(config, comparison)
    root = output_root(config)
    report_path = root / "comparison" / "V2_VS_V3_UPLOAD_CANDIDATES_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = comparison.get("candidates", {})
    readiness = comparison.get("upload_readiness", {})
    lines = [
        "# Step 21F - Frozen upload candidates",
        "",
        "## Executive summary",
        "",
        "This step freezes two existing Track 1 outputs without rerunning tracking, motion filtering, global association, or export.",
        "The candidates must be submitted separately. Local diagnostics describe trade-offs but do not replace the official evaluation server.",
        "",
        "## Candidates",
        "",
    ]
    for name in ["v2_current", "v3_gap_aware_soft"]:
        summary = candidates.get(name, {})
        ready = readiness.get(name, {})
        lines.extend(_candidate_section(name, summary, ready))
    lines.extend(
        [
            "## Local comparison",
            "",
            "| Metric | V2 current | V3 gap-aware soft | Delta V3-V2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparison.get("metrics", []):
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                row.get("metric"),
                _display(row.get("v2_current")),
                _display(row.get("v3_gap_aware_soft")),
                _display(row.get("delta_v3_minus_v2")),
            )
        )
    lines.extend(_distribution_section("Per-scene Track1 rows", candidates, "per_scene_rows"))
    lines.extend(_distribution_section("Per-class Track1 rows", candidates, "per_class_rows"))
    verdict = comparison.get("verdict", {})
    lines.extend(
        [
            "",
            "## Validation and upload readiness",
            "",
            "- Verdict: `%s`" % verdict.get("label"),
            "- V2 package: `%s`" % _ready_path(readiness, "v2_current"),
            "- V3 package: `%s`" % _ready_path(readiness, "v3_gap_aware_soft"),
            "- Each zip is expected to contain exactly one root-level file named `track1.txt`.",
            "",
            "## Recommendation",
            "",
            "V2 current is the coverage-first candidate.",
            "V3 gap_aware_soft is the identity-quality-first candidate.",
            "Official evaluation is required to determine which trade-off is better.",
            "Upload V2 current first, then upload V3 gap_aware_soft as a separate submission.",
            "",
            "## Figures",
            "",
            "Figure generation status: `%s`. Files are under `figures/`." % figures.get("status"),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _candidate_section(name: str, summary: Dict[str, Any], ready: Dict[str, Any]) -> List[str]:
    validation = summary.get("validation", {})
    rows_per_track = summary.get("rows_per_track", {})
    return [
        "### %s" % name,
        "",
        "- Meaning: %s" % (summary.get("description") or name),
        "- Frozen file: `%s`" % summary.get("frozen_track1_path"),
        "- SHA256: `%s`" % summary.get("sha256"),
        "- Rows: %s" % summary.get("track1_rows"),
        "- Unique global tracks: %s" % summary.get("unique_tracks"),
        "- Rows per track mean/median: %s / %s" % (_display(rows_per_track.get("mean")), _display(rows_per_track.get("median"))),
        "- Validation: `%s`, errors=%s" % (validation.get("status"), validation.get("num_errors")),
        "- Ready for upload: `%s`" % ready.get("ready"),
        "- Zip: `%s`" % ready.get("zip_path"),
        "- Zip SHA256: `%s`" % ready.get("zip_sha256"),
        "- Zip content verified: `%s`" % ready.get("package_verified"),
        "",
    ]


def _ready_path(readiness: Dict[str, Any], name: str) -> Any:
    return readiness.get(name, {}).get("zip_path")


def _distribution_section(title: str, candidates: Dict[str, Any], key: str) -> List[str]:
    v2 = candidates.get("v2_current", {}).get(key, {})
    v3 = candidates.get("v3_gap_aware_soft", {}).get(key, {})
    v2 = v2 if isinstance(v2, dict) else {}
    v3 = v3 if isinstance(v3, dict) else {}
    keys = sorted(set(list(v2.keys()) + list(v3.keys())), key=_sort_key)
    lines = ["", "## %s" % title, "", "| Key | V2 current | V3 gap-aware soft | Delta |", "|---|---:|---:|---:|"]
    for value in keys:
        left = int(v2.get(value, 0) or 0)
        right = int(v3.get(value, 0) or 0)
        lines.append("| %s | %d | %d | %d |" % (value, left, right, right - left))
    return lines


def _sort_key(value: Any) -> Any:
    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "%.6f" % value
    return str(value)
