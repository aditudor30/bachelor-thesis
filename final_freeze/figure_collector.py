"""Collect final figures and generate compact metric plots."""

import shutil
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze.freeze_config import output_root_from_config
from deep_oc_sort_3d.final_freeze.freeze_io import (
    load_yaml,
    progress_iter,
    read_json,
    safe_float,
    write_csv_rows,
    write_json,
)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def collect_final_figures_from_config(config_path: Path, show_progress: bool = True) -> Dict[str, Any]:
    """Copy selected existing figures and generate metric plots."""
    config = load_yaml(config_path)
    output_root = output_root_from_config(config)
    rows = collect_existing_figures(config, show_progress=show_progress)
    generated = generate_metric_figures(config)
    rows.extend(generated)
    write_csv_rows(rows, output_root / "figures" / "figure_manifest.csv")
    write_json({"figures": rows}, output_root / "figures" / "figure_manifest.json")
    _write_captions(rows, output_root / "figures" / "FIGURE_CAPTIONS.md")
    return {"figures": rows, "num_figures": len(rows)}


def collect_existing_figures(config: Dict[str, Any], show_progress: bool = True) -> List[Dict[str, Any]]:
    """Collect a bounded set of existing figure files into final_freeze/figures."""
    output_root = output_root_from_config(config)
    max_per_category = int(config.get("figures", {}).get("max_per_category", 12))
    roots = _figure_roots(config)
    rows: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}
    candidates = _image_candidates(roots, output_root)
    for source in progress_iter(candidates, show_progress, "collect final figures", "figure"):
        category = classify_figure(source)
        if category_counts.get(category, 0) >= max_per_category:
            continue
        destination = output_root / "figures" / category / _destination_name(source, len(rows))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))
        category_counts[category] = category_counts.get(category, 0) + 1
        rows.append(
            {
                "kind": "copied",
                "category": category,
                "source_path": str(source),
                "destination_path": str(destination),
                "caption": default_caption(category, source),
            }
        )
    return rows


def generate_metric_figures(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate simple final metric bar plots if matplotlib is available."""
    output_root = output_root_from_config(config)
    data = read_json(output_root / "tables" / "final_baseline_comparison.json") or {}
    rows = list(data.get("variants", []))
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    generated: List[Dict[str, Any]] = []
    specs = [
        ("track1_rows", "Track1 Rows", "final_track1_rows.png"),
        ("multi_camera_tracks", "Multi-camera Tracks", "final_multi_camera_tracks.png"),
        ("fragmentation_approx", "Fragmentation Approx.", "final_fragmentation.png"),
        ("global_purity", "Global Purity", "final_global_purity.png"),
        ("false_merge_rate", "False Merge Rate", "final_false_merge_rate.png"),
    ]
    labels = [_short_variant_name(str(row.get("variant_name", ""))) for row in rows]
    for key, title, filename in specs:
        values = [safe_float(row.get(key), None) for row in rows]
        if all(value is None for value in values):
            continue
        numeric = [0.0 if value is None else value for value in values]
        destination = output_root / "figures" / "metric_charts" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 4))
        plt.bar(labels, numeric)
        plt.title(title)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(str(destination), dpi=160)
        plt.close()
        generated.append(
            {
                "kind": "generated",
                "category": "metric_charts",
                "source_path": "final_metrics_bundle",
                "destination_path": str(destination),
                "caption": "Final freeze metric chart for %s." % title,
            }
        )
    return generated


def classify_figure(path: Path) -> str:
    """Classify a figure by filename for report organization."""
    text = str(path).lower()
    if "bev" in text or "trajectory" in text or "trajector" in text:
        return "bev"
    if "cuboid" in text or "projection" in text or "3d" in text:
        return "cuboid_diagnostic"
    if "reid" in text or "embedding" in text:
        return "reid"
    if "panel" in text or "paper" in text or "final" in text:
        return "final_panel"
    if "track" in text or "global" in text or "local" in text:
        return "tracking"
    return "misc"


def default_caption(category: str, source: Path) -> str:
    """Build a short default caption for a collected figure."""
    return "%s figure collected from %s." % (category.replace("_", " "), source.name)


def _figure_roots(config: Dict[str, Any]) -> List[Path]:
    paths = config.get("paths", {})
    configured = config.get("figures", {}).get("search_roots", [])
    roots = [Path(str(item)) for item in configured]
    defaults = [
        "output/visualizations_mvp",
        "output/mvp_paper_figures",
        "output/paper_figures",
        "output/robust_bev",
        "output/final_mvp_exports/yolo11m_medium_conf001_transition",
        "output/final_mvp_exports/baseline_v2_pseudo3d_fullcam",
    ]
    for item in defaults:
        roots.append(Path(str(paths.get(item, item))))
    return _unique_existing_dirs(roots)


def _unique_existing_dirs(roots: List[Path]) -> List[Path]:
    output: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen or not root.exists() or not root.is_dir():
            continue
        seen.add(key)
        output.append(root)
    return output


def _image_candidates(roots: List[Path], output_root: Path) -> List[Path]:
    output = []
    freeze_root = output_root.resolve()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                continue
            try:
                if str(path.resolve()).startswith(str(freeze_root)):
                    continue
            except OSError:
                pass
            output.append(path)
    return output


def _destination_name(source: Path, index: int) -> str:
    stem = _slugify(source.stem)
    suffix = source.suffix.lower()
    return "%04d_%s%s" % (index, stem, suffix)


def _slugify(text: str) -> str:
    allowed = []
    for char in text:
        if char.isalnum():
            allowed.append(char.lower())
        else:
            allowed.append("_")
    return "_".join([item for item in "".join(allowed).split("_") if item])


def _short_variant_name(name: str) -> str:
    replacements = [
        ("V1 geometry-only", "V1"),
        ("V2 pseudo3D fullcam current", "V2"),
        ("V2 export_compact", "V2 compact"),
        ("ReID diagnostic / OSNet ablation", "ReID diag"),
    ]
    for source, target in replacements:
        if name == source:
            return target
    return name[:18]


def _write_captions(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = ["# Figure Captions", ""]
    for row in rows:
        lines.append("- `%s`: %s" % (row.get("destination_path"), row.get("caption")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
