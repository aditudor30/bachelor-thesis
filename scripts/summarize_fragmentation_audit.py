"""Print a compact summary of a fragmentation audit output root."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def summarize_fragmentation_audit(root: Path) -> Dict[str, Any]:
    """Print and return a compact audit summary."""
    report = root / "report" / "BASELINE_V2_FULLCAM_FRAGMENTATION_AUDIT_SUMMARY_COMPACT.json"
    comparison = root / "comparisons" / "v1_vs_v2_fragmentation_summary.json"
    root_cause = root / "diagnostics" / "root_cause_analysis.json"
    data = _read(report)
    comp = _read(comparison)
    cause = _read(root_cause)
    print("root: %s" % root)
    print("verdict: %s" % cause.get("verdict", data.get("root_cause", {}).get("verdict")))
    print("stages: %s" % ", ".join(data.get("stages", [])))
    print("high_level:")
    for key, value in sorted(comp.get("high_level", data.get("high_level", {})).items()):
        print("  %s: %s" % (key, value))
    print("recommendations:")
    for rec in cause.get("tuning_recommendations", data.get("root_cause", {}).get("tuning_recommendations", [])):
        print("  %s: %s" % (rec.get("area"), rec.get("action")))
    return {"summary": data, "comparison": comp, "root_cause": cause}


def _read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fragmentation audit outputs.")
    parser.add_argument("--root", type=Path, default=Path("output/baseline_v2_pseudo3d_fullcam_fragmentation_audit"))
    args = parser.parse_args()
    summarize_fragmentation_audit(args.root)


if __name__ == "__main__":
    main()

