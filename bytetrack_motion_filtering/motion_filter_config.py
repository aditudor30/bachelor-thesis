"""Configuration resolution for the Step 21E motion-filter sweep."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from deep_oc_sort_3d.bytetrack_motion_filtering.motion_filter_io import write_yaml


CLASS_NAMES = {
    0: "Person",
    1: "Forklift",
    2: "PalletTruck",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
    6: "NovaCarter",
}


def load_motion_filter_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate a Step 21E YAML config."""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Step 21E config must be a mapping")
    if not isinstance(value.get("variants"), dict) or not value.get("variants"):
        raise ValueError("Step 21E config requires a non-empty variants mapping")
    value["_config_path"] = str(path)
    return value


def output_root(config: Dict[str, Any]) -> Path:
    """Return the isolated Step 21E output root."""
    section = config.get("bytetrack_gap_aware_motion_filter", {})
    return Path(str(section.get("output_root", "output/bytetrack_motion_filtering/baseline_v2_pseudo3d_fullcam")))


def variant_root(config: Dict[str, Any], variant_name: str) -> Path:
    """Return one isolated filter-run root."""
    return output_root(config) / "filter_runs" / str(variant_name)


def variant_names(config: Dict[str, Any]) -> List[str]:
    """Return configured variants in YAML order."""
    return [str(name) for name in config.get("variants", {}).keys()]


def subset_entries(config: Dict[str, Any], include_test: bool = True) -> List[Tuple[str, str, str]]:
    """Return pipeline subset, dataset split and scene tuples."""
    output = []
    for subset, payload in config.get("subsets", {}).items():
        if not isinstance(payload, dict):
            continue
        split = str(payload.get("split", ""))
        if split == "test" and not include_test:
            continue
        for scene_name in payload.get("scenes", []) or []:
            output.append((str(subset), split, str(scene_name)))
    return sorted(set(output))


def candidate_root(config: Dict[str, Any]) -> Path:
    """Resolve the 21C candidate root, tolerating alternate variant layouts."""
    paths = config.get("paths", {})
    configured = Path(str(paths.get("bytetrack_21c_best_candidates_root", "")))
    if configured.exists() and _has_candidate_files(configured):
        return configured
    variant = Path(str(paths.get("bytetrack_21c_best_variant_root", "")))
    candidates = [
        variant / "candidates",
        variant / "mtmc_candidates",
        variant / "outputs" / "candidates",
    ]
    for path in candidates:
        if path.exists() and _has_candidate_files(path):
            return path
    return configured if str(configured) else variant / "candidates"


def source_local_tracks_root(config: Dict[str, Any]) -> Path:
    """Return the unchanged 21C local-track root used by final export."""
    variant = Path(str(config.get("paths", {}).get("bytetrack_21c_best_variant_root", "")))
    return variant / "local_tracks"


def velocity_priors_root(config: Dict[str, Any]) -> Path:
    """Return the velocity-prior output directory."""
    return output_root(config) / "velocity_priors"


def write_resolved_config(config: Dict[str, Any]) -> Path:
    """Write the resolved config without private runtime keys."""
    path = output_root(config) / "configs" / "resolved_config.yaml"
    write_yaml(path, {key: value for key, value in config.items() if not str(key).startswith("_")})
    return path


def _has_candidate_files(root: Path) -> bool:
    for pattern in ("*_candidates.jsonl", "*_candidates.csv"):
        for path in root.rglob(pattern):
            if not any(token in path.stem for token in ("_clean_", "_invalid_", "_suspicious_", "_unknown_")):
                return True
    return False

