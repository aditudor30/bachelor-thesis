"""Person-only pruning policies."""

from typing import Any, Dict, List, Set, Tuple

from deep_oc_sort_3d.person_cleanup.person_track_classifier import PersonTrackStats


TrackKey = Tuple[str, str, str, str]


def keys_to_drop_for_policy(
    stats: List[PersonTrackStats],
    policy: Dict[str, Any],
) -> Tuple[Set[TrackKey], List[Dict[str, Any]]]:
    """Return Person track keys dropped by a policy plus audit rows."""
    if not bool(policy.get("enabled", True)):
        rows = []
        for item in stats:
            row = item.to_dict()
            row["drop"] = False
            row["drop_reason"] = "pruning_disabled"
            rows.append(row)
        return set(), rows
    drop_keys = set()
    audit_rows = []
    for item in stats:
        drop, reason = should_drop_person_track(item, policy)
        if drop:
            drop_keys.add(item.key)
        row = item.to_dict()
        row["drop"] = bool(drop)
        row["drop_reason"] = reason
        audit_rows.append(row)
    return drop_keys, audit_rows


def should_drop_person_track(stats: PersonTrackStats, policy: Dict[str, Any]) -> Tuple[bool, str]:
    """Return pruning decision for one Person track."""
    if int(stats.class_id) != int(policy.get("class_id", 0)):
        return False, "non_person_preserved"
    apply_subsets = policy.get("apply_to_subsets")
    if apply_subsets is not None:
        subset_set = set([str(item) for item in apply_subsets])
        if stats.subset not in subset_set:
            return False, "subset_not_selected"
    mode = str(policy.get("mode", "short_lowconf"))
    if mode == "singletons":
        return _drop_singleton(stats, policy)
    if mode == "compact":
        return _drop_compact(stats, policy)
    return _drop_short_lowconf(stats, policy)


def _drop_short_lowconf(stats: PersonTrackStats, policy: Dict[str, Any]) -> Tuple[bool, str]:
    max_rows = int(policy.get("max_rows_per_track", 3))
    mean_thr = float(policy.get("mean_confidence_threshold", 0.03))
    max_thr = policy.get("max_confidence_threshold", 0.08)
    if stats.rows > max_rows:
        return False, "too_long"
    if stats.mean_confidence >= mean_thr:
        return False, "mean_confidence_high_enough"
    if max_thr is not None and stats.max_confidence >= float(max_thr):
        return False, "max_confidence_high_enough"
    return True, "person_short_lowconf"


def _drop_singleton(stats: PersonTrackStats, policy: Dict[str, Any]) -> Tuple[bool, str]:
    mean_thr = float(policy.get("mean_confidence_threshold", 0.05))
    max_thr = policy.get("max_confidence_threshold", 0.10)
    if stats.rows != 1:
        return False, "not_singleton"
    if stats.mean_confidence >= mean_thr:
        return False, "mean_confidence_high_enough"
    if max_thr is not None and stats.max_confidence >= float(max_thr):
        return False, "max_confidence_high_enough"
    return True, "person_singleton_lowconf"


def _drop_compact(stats: PersonTrackStats, policy: Dict[str, Any]) -> Tuple[bool, str]:
    max_rows = int(policy.get("max_rows_per_track", 5))
    mean_thr = float(policy.get("mean_confidence_threshold", 0.02))
    if stats.rows <= max_rows and stats.mean_confidence < mean_thr:
        return True, "person_compact_short_lowconf"
    return False, "kept_by_compact_policy"
