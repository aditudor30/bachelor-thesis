"""Collect and generate final freeze v2 figures."""

import shutil
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_config import output_root_from_config
from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import load_yaml, progress_iter, read_json, safe_float, write_csv_rows, write_json, write_text


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def collect_final_freeze_v2_figures_from_config(config_path: Path, show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Collect existing figures and generate final charts."""
    _unused_overwrite = overwrite
    config = load_yaml(Path(config_path))
    output_root = output_root_from_config(config)
    rows = collect_existing_figures(config, show_progress=show_progress)
    if bool(config.get("figures", {}).get("generate_charts", True)):
        rows.extend(generate_final_charts(config))
    write_csv_rows(rows, output_root / "figures" / "figure_manifest.csv")
    write_json({"figures": rows}, output_root / "figures" / "figure_manifest.json")
    write_figure_captions(rows, output_root / "figures" / "captions" / "FIGURE_CAPTIONS.md")
    return {"num_figures": len(rows), "figures": rows}


def collect_existing_figures(config: Dict[str, Any], show_progress: bool = True) -> List[Dict[str, Any]]:
    """Copy selected existing figures into final freeze v2."""
    output_root = output_root_from_config(config)
    max_reid = int(config.get("figures", {}).get("max_reid_panels_per_category", 5))
    max_generic = int(config.get("figures", {}).get("max_per_category", 12))
    roots = figure_roots(config)
    candidates = image_candidates(roots, output_root)
    counts: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for source in progress_iter(candidates, show_progress, "collect final freeze v2 figures", "figure"):
        category = classify_figure(source)
        limit = max_reid if category == "reid_panels" else max_generic
        if counts.get(category, 0) >= limit:
            continue
        destination = output_root / "figures" / category / destination_name(source, len(rows))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))
        counts[category] = counts.get(category, 0) + 1
        rows.append(
            {
                "kind": "copied",
                "category": category,
                "source_path": str(source),
                "destination_path": str(destination),
                "diagnostic_only": "1" if category in ("reid_panels", "charts") else "0",
                "caption": default_caption(category, source),
            }
        )
    return rows


def generate_final_charts(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate compact final metric charts."""
    output_root = output_root_from_config(config)
    bundle = read_json(output_root / "tables" / "final_metrics_bundle.json") or {}
    variants = list(bundle.get("variants", []))
    if not variants:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    rows = []
    chart_specs = [
        ("track1_rows", "Track1 rows", "final_track1_rows.png"),
        ("person_fragmentation", "Person fragmentation", "final_person_fragmentation.png"),
        ("global_purity", "Global purity", "final_global_purity.png"),
        ("false_merge_rate", "False merge rate", "final_false_merge_rate.png"),
    ]
    labels = [short_name(str(row.get("variant_name", ""))) for row in variants]
    for key, title, filename in chart_specs:
        values = [safe_float(row.get(key), None) for row in variants]
        if all(value is None for value in values):
            continue
        numeric = [0.0 if value is None else float(value) for value in values]
        path = output_root / "figures" / "charts" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(labels, numeric, color="#4c78a8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=160)
        plt.close(fig)
        rows.append(
            {
                "kind": "generated",
                "category": "charts",
                "source_path": "final_metrics_bundle",
                "destination_path": str(path),
                "diagnostic_only": "0",
                "caption": "Final comparison chart for %s." % title,
            }
        )
    return rows


def figure_roots(config: Dict[str, Any]) -> List[Path]:
    """Return existing figure search roots."""
    roots = [Path(str(item)) for item in config.get("figures", {}).get("search_roots", [])]
    return unique_existing_dirs(roots)


def unique_existing_dirs(roots: List[Path]) -> List[Path]:
    """Deduplicate existing dirs."""
    output = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen or not root.exists() or not root.is_dir():
            continue
        seen.add(key)
        output.append(root)
    return output


def image_candidates(roots: List[Path], output_root: Path) -> List[Path]:
    """List image candidates outside output root."""
    output = []
    freeze_root = output_root.resolve()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                if str(path.resolve()).startswith(str(freeze_root)):
                    continue
            except OSError:
                pass
            output.append(path)
    return output


def classify_figure(path: Path) -> str:
    """Classify figure path into final folders."""
    text = str(path).lower()
    if "reid" in text or "merge" in text or "likely" in text or "ambiguous" in text:
        return "reid_panels"
    if "bev" in text or "trajectory" in text:
        return "bev"
    if "cuboid" in text or "3d" in text or "projection" in text:
        return "qualitative_3d"
    if "track" in text or "bbox" in text or "2d" in text:
        return "qualitative_2d"
    if "chart" in text or "comparison" in text or "distribution" in text:
        return "charts"
    return "qualitative_2d"


def default_caption(category: str, source: Path) -> str:
    """Return thesis-friendly default caption."""
    if category == "bev":
        return "Coordinate-space BEV visualization of estimated 3D trajectories. The plot is not map-aligned; percentile clipping is used only for visualization."
    if category == "reid_panels":
        return "Example of a fine-tuned ReID merge candidate. The panel compares representative crops from two Person fragments before merging and reports the ReID similarity, temporal gap and geometric constraints used for the association decision."
    if category == "qualitative_3d":
        return "Qualitative pseudo3D/cuboid diagnostic visualization for the 3D provenance-backed MVP."
    if category == "qualitative_2d":
        return "Qualitative 2D tracking visualization showing detections and local/global track identities."
    return "Final freeze figure collected from `%s`." % source.name


def write_figure_captions(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write required figure captions."""
    lines = [
        "# Figure Captions",
        "",
        "## Required Captions",
        "",
        "- Pipeline: End-to-end 3D MTMC pipeline from RGB detections to local tracks, pseudo3D observations, global association and Track1 export.",
        "- RGB-Depth alignment: RGB frame, depth supervision and visible 2D ground truth boxes used for training diagnostics.",
        "- 2D tracking: Local 2D tracklets produced per camera before cross-camera association.",
        "- Pseudo3D cuboids: Diagnostic projection of estimated 3D cuboids on RGB frames.",
        "- BEV: Coordinate-space BEV visualization of estimated 3D trajectories. The plot is not map-aligned; percentile clipping is used only for visualization.",
        "- ReID likely_good: Example of a fine-tuned ReID merge candidate. The panel compares representative crops from two Person fragments before merging and reports the ReID similarity, temporal gap and geometric constraints used for the association decision.",
        "- ReID ambiguous: Ambiguous fine-tuned ReID merge candidate retained for diagnostic review rather than automatic claims.",
        "- Final comparison chart: Final frozen comparison of V1, V2 pseudo3D, compact and ReID-enhanced variants.",
        "",
        "## Collected Files",
        "",
    ]
    for row in rows:
        lines.append("- `%s`: %s" % (row.get("destination_path"), row.get("caption")))
    write_text(lines, path)


def destination_name(source: Path, index: int) -> str:
    """Return deterministic copied figure name."""
    return "%04d_%s%s" % (index, slugify(source.stem), source.suffix.lower())


def slugify(text: str) -> str:
    """Make a simple filename slug."""
    chars = []
    for char in text:
        if char.isalnum():
            chars.append(char.lower())
        else:
            chars.append("_")
    return "_".join([item for item in "".join(chars).split("_") if item])


def short_name(name: str) -> str:
    """Short label for charts."""
    mapping = {
        "v1_geometry_only": "V1",
        "v2_pseudo3d_fullcam_current": "V2",
        "v2_export_compact": "compact",
        "osnet_pretrained_diagnostic": "pre-ReID",
        "osnet_finetuned_threshold_080": "ft-080",
        "osnet_finetuned_combined_safe_080": "combined",
    }
    return mapping.get(name, name[:14])

