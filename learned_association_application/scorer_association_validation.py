"""Validation summary helpers for Step 20C variants."""

from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.learned_association_application.scorer_association_io import read_json


def load_track1_validation(run_root: Path) -> Dict[str, Any]:
    """Load the existing Track1 validator report."""
    for path in (
        run_root / "validation" / "track1_validation_report.json",
        run_root / "track1_submission" / "track1_validation_report.json",
    ):
        value = read_json(path)
        if value is not None:
            return value
    return {"status": "not_available", "num_errors": None}


def validation_is_clean(report: Dict[str, Any]) -> bool:
    """Return whether validation explicitly completed without errors."""
    return str(report.get("status", "")).lower() == "ok" and int(report.get("num_errors") or 0) == 0
