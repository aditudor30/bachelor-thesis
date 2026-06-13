"""Combine conservative recovery mechanisms into the only selectable V3.1 variant."""

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from deep_oc_sort_3d.v3_coverage_extension.recovery_source_loader import RecoveryTrack, TrackKey


def run_variant_builds(
    config: Dict[str, Any],
    requested_variants: Sequence[str],
    progress: bool = True,
) -> Dict[str, Any]:
    """Load V3 sources once, select requested variants and write Track1 outputs."""
    from pathlib import Path

    from deep_oc_sort_3d.official_023_027.official_track1_io import read_track1_rows
    from deep_oc_sort_3d.v3_coverage_extension.official_track1_builder import build_variant_track1
    from deep_oc_sort_3d.v3_coverage_extension.recovery_source_loader import load_recovery_sources
    from deep_oc_sort_3d.v3_coverage_extension.scene_class_targeted_recovery import select_scene_class_targeted
    from deep_oc_sort_3d.v3_coverage_extension.short_track_recovery import select_short_track_safe
    from deep_oc_sort_3d.v3_coverage_extension.single_camera_recovery import select_single_camera_clean
    from deep_oc_sort_3d.v3_coverage_extension.tentative_export_recovery import select_associated_tentative

    sources = load_recovery_sources(config, progress=progress)
    if not sources.tracks:
        raise RuntimeError("No V3 local-track recovery sources were found; run the 22B audit and inspect candidate_recovery_sources.json")
    if not sources.covered_track_keys:
        raise RuntimeError("No V3 baseline local-track assignments were found; refusing recovery because duplicate physical tracks cannot be excluded safely")
    baseline_rows = read_track1_rows(Path(str(config.get("paths", {}).get("v3_official_track1", ""))), progress=progress)
    short, short_summary = select_short_track_safe(sources.tracks, config)
    tentative, tentative_summary = select_associated_tentative(sources.tracks, config)
    targeted, targeted_summary = select_scene_class_targeted(sources.tracks, baseline_rows, config)
    single, single_summary = select_single_camera_clean(sources.tracks, config)
    selections = {
        "v3_short_track_safe": short,
        "v3_associated_tentative_export": tentative,
        "v3_scene_class_targeted_recovery": targeted,
        "v3_single_camera_keep_clean": single,
    }
    summaries = {
        "v3_short_track_safe": short_summary,
        "v3_associated_tentative_export": tentative_summary,
        "v3_scene_class_targeted_recovery": targeted_summary,
        "v3_single_camera_keep_clean": single_summary,
    }
    balanced, balanced_summary = build_balanced_extension(selections, len(baseline_rows), config)
    selections["v3_balanced_coverage_extension"] = balanced
    summaries["v3_balanced_coverage_extension"] = balanced_summary
    output = {}
    for variant in requested_variants:
        output[variant] = build_variant_track1(config, variant, selections.get(variant, []), summaries.get(variant, {}), progress=progress)
    return {"variants": output, "source_warnings": sources.warnings, "covered_track_keys": len(sources.covered_track_keys), "loaded_tracks": len(sources.tracks)}


def build_balanced_extension(
    selections: Mapping[str, Sequence[RecoveryTrack]],
    baseline_rows: int,
    config: Dict[str, Any],
) -> Tuple[List[RecoveryTrack], Dict[str, Any]]:
    """Union safe mechanisms with deterministic priorities and a hard row cap."""
    rules = config.get("recovery_rules", {}).get("balanced", {})
    targeting = config.get("targeting", {})
    target_scenes = set(int(value) for value in targeting.get("target_scenes", [24, 26, 23]))
    target_classes = set(int(value) for value in targeting.get("target_official_classes", [0, 1, 3]))
    priority = [str(value) for value in rules.get("source_priority", [])]
    hard_total = int(rules.get("hard_total_rows_max", 225000))
    available = max(0, hard_total - int(baseline_rows))
    chosen = {}
    source_by_key = {}
    for source in priority:
        ordered = sorted(selections.get(source, []), key=lambda item: (_target_score(item, target_scenes, target_classes), item.mean_confidence, item.length), reverse=True)
        for track in ordered:
            if track.key not in chosen:
                chosen[track.key] = track
                source_by_key[track.key] = source
    ordered_tracks = sorted(chosen.values(), key=lambda item: (_target_score(item, target_scenes, target_classes), item.mean_confidence, item.length), reverse=True)
    selected = []
    used_rows = 0
    by_source = defaultdict(int)
    for track in ordered_tracks:
        if used_rows + track.length > available:
            continue
        selected.append(track)
        used_rows += track.length
        by_source[source_by_key[track.key]] += 1
    target_rows = sum(track.length for track in selected if track.scene_id in target_scenes and track.official_class_id in target_classes)
    return selected, {
        "selected_tracks": len(selected), "selected_rows": used_rows, "baseline_rows": baseline_rows,
        "estimated_total_rows": baseline_rows + used_rows, "hard_total_rows_max": hard_total,
        "selected_tracks_by_source": dict(sorted(by_source.items())),
        "target_scene_class_rows": target_rows,
        "target_scene_class_share": float(target_rows) / float(used_rows) if used_rows else 0.0,
    }


def _target_score(track: RecoveryTrack, scenes: set, classes: set) -> int:
    score = 0
    if track.scene_id in scenes:
        score += 2
    if track.official_class_id in classes:
        score += 2
    if track.official_class_id in (2, 4, 5, 6):
        score -= 1
    return score
