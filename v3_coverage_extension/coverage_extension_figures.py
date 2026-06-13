"""Diagnostic figures for V2, V3 and V3.1 coverage."""

from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_config import output_root
from deep_oc_sort_3d.v3_coverage_extension.coverage_extension_io import read_json


def write_coverage_figures(config: Dict[str, Any]) -> List[str]:
    """Write requested summary figures when matplotlib is available."""
    if not bool(config.get("figures", {}).get("enabled", True)):
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    root = output_root(config)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    v2 = read_json(root / "audit" / "v2_official_reference_summary.json")
    v3 = read_json(root / "audit" / "v3_official_baseline_summary.json")
    summary = read_json(root / "comparison" / "v3_coverage_extension_summary.json")
    balanced = next((row for row in summary.get("variants", []) if row.get("variant") == "v3_balanced_coverage_extension"), {})
    outputs = []
    outputs.append(_bar(plt, figures / "rows_by_scene.png", "Rows by scene", v2.get("scene_distribution", {}), v3.get("scene_distribution", {}), balanced.get("scene_distribution", {})))
    outputs.append(_bar(plt, figures / "rows_by_class.png", "Rows by official class", v2.get("class_distribution", {}), v3.get("class_distribution", {}), balanced.get("class_distribution", {})))
    variants = summary.get("variants", [])
    outputs.append(_simple(plt, figures / "rows_delta_vs_v3.png", "Rows gained vs V3", [row.get("variant", "") for row in variants], [row.get("row_gain_vs_v3", 0) for row in variants]))
    outputs.append(_simple(plt, figures / "rows_per_track_distribution.png", "Median rows per track", [row.get("variant", "") for row in variants], [row.get("rows_per_track_median", 0) or 0 for row in variants]))
    outputs.append(_simple(plt, figures / "v2_v3_v31_summary.png", "V2 vs V3 vs V3.1 rows", ["V2", "V3", "V3.1"], [v2.get("rows", 0), v3.get("rows", 0), balanced.get("track1_rows", 0)]))
    return [value for value in outputs if value]


def _bar(plt: Any, path: Path, title: str, v2: Dict[str, int], v3: Dict[str, int], v31: Dict[str, int]) -> str:
    keys = sorted(set(v2.keys()).union(v3.keys()).union(v31.keys()), key=int)
    x = list(range(len(keys)))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar([value - 0.25 for value in x], [v2.get(key, 0) for key in keys], width=0.25, label="V2")
    axis.bar(x, [v3.get(key, 0) for key in keys], width=0.25, label="V3")
    axis.bar([value + 0.25 for value in x], [v31.get(key, 0) for key in keys], width=0.25, label="V3.1")
    axis.set_xticks(x)
    axis.set_xticklabels(keys)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return str(path)


def _simple(plt: Any, path: Path, title: str, labels: List[str], values: List[Any]) -> str:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(list(range(len(labels))), values)
    axis.set_xticks(list(range(len(labels))))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)
    return str(path)
