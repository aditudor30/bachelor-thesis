"""Validation and enumeration of ByteTrack coverage variants."""

from typing import Any, Dict, List


REQUIRED_VARIANT_FIELDS = [
    "track_high_thresh",
    "track_low_thresh",
    "new_track_thresh",
    "match_thresh",
    "second_stage_match_thresh",
    "track_buffer",
    "min_confidence_for_input",
]


def list_variants(config: Dict[str, Any]) -> List[str]:
    """Return deterministic configured variant names."""
    return sorted(str(name) for name in config.get("variants", {}).keys())


def validate_variant_grid(config: Dict[str, Any]) -> List[str]:
    """Return human-readable variant configuration errors."""
    errors = []
    for name in list_variants(config):
        values = config.get("variants", {}).get(name, {})
        for field in REQUIRED_VARIANT_FIELDS:
            if field not in values:
                errors.append("%s missing %s" % (name, field))
        try:
            if float(values.get("track_low_thresh", 0.0)) > float(values.get("track_high_thresh", 0.0)):
                errors.append("%s track_low_thresh exceeds track_high_thresh" % name)
        except (TypeError, ValueError):
            errors.append("%s has non-numeric thresholds" % name)
    tracking = config.get("tracking", {})
    if bool(tracking.get("class_agnostic_tracking", False)):
        errors.append("class_agnostic_tracking must be false")
    if bool(tracking.get("allow_cross_class_matching", False)):
        errors.append("allow_cross_class_matching must be false")
    return errors

