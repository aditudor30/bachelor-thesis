"""Build final freeze v2 manifests and source path summaries."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze_v2.final_checksum_utils import maybe_sha256, write_checksum_text
from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_config import output_root_from_config
from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import load_yaml, progress_iter, write_csv_rows, write_json
from deep_oc_sort_3d.final_freeze_v2.final_variant_registry import final_variant_specs


def build_final_outputs_manifest_from_config(config_path: Path, show_progress: bool = True) -> Dict[str, Any]:
    """Build final manifest JSON/CSV/checksums/source paths."""
    config = load_yaml(Path(config_path))
    output_root = output_root_from_config(config)
    package_cfg = config.get("packages", {})
    rows = source_manifest_rows(config)
    rows.extend(final_output_rows(output_root, show_progress=show_progress))
    threshold = float(package_cfg.get("large_file_mb_threshold", 200))
    checksum_large = bool(package_cfg.get("checksum_large_files", False))
    for row in progress_iter(rows, show_progress, "final freeze v2 manifest checksums", "artifact"):
        path = Path(str(row.get("path", "")))
        if bool(row.get("exists", False)) and path.is_file():
            sha = maybe_sha256(path, threshold, checksum_large)
            row["sha256"] = "" if sha is None else sha
        else:
            row["sha256"] = ""
    manifest = {"created_utc": datetime.utcnow().isoformat() + "Z", "output_root": str(output_root), "artifacts": rows}
    write_json(manifest, output_root / "manifests" / "final_outputs_manifest.json")
    write_csv_rows(rows, output_root / "manifests" / "final_outputs_manifest.csv")
    write_checksum_text([row for row in rows if row.get("sha256")], output_root / "manifests" / "checksums_sha256.txt")
    write_json({"paths": config.get("paths", {}), "variants": final_variant_specs(config)}, output_root / "manifests" / "source_paths_used.json")
    write_baseline_summaries(output_root, rows, config)
    return manifest


def source_manifest_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Manifest rows for source artifacts and roots."""
    rows = []
    for spec in final_variant_specs(config):
        for path_text in spec.get("source_paths", []):
            if not path_text:
                continue
            rows.append(artifact_row(Path(str(path_text)), "source_path", spec.get("variant_name"), spec.get("role"), spec.get("notes", "")))
        if spec.get("track1_path"):
            rows.append(artifact_row(Path(str(spec.get("track1_path"))), "track1", spec.get("variant_name"), spec.get("role"), "Track1 text for frozen variant."))
    return rows


def final_output_rows(output_root: Path, show_progress: bool = True) -> List[Dict[str, Any]]:
    """Manifest rows for generated final freeze files."""
    rows = []
    if not output_root.exists():
        return rows
    files = [path for path in sorted(output_root.rglob("*")) if path.is_file()]
    for path in progress_iter(files, show_progress, "final freeze v2 output manifest", "file"):
        rows.append(artifact_row(path, classify_output_artifact(path, output_root), "final_freeze_v2", "generated", "Generated Step 19 artifact."))
    return rows


def artifact_row(path: Path, artifact_type: str, variant: Any, role: Any, description: str) -> Dict[str, Any]:
    """Build one manifest row."""
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "file_size": path.stat().st_size if exists and path.is_file() else "",
        "sha256": "",
        "artifact_type": artifact_type,
        "variant": variant,
        "role": role,
        "description": description,
    }


def classify_output_artifact(path: Path, output_root: Path) -> str:
    """Classify generated artifact by output subfolder."""
    try:
        rel = path.relative_to(output_root)
    except ValueError:
        return "artifact"
    head = rel.parts[0] if rel.parts else "artifact"
    return head


def write_baseline_summaries(output_root: Path, rows: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    """Write per-baseline lightweight summaries."""
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant", "")), []).append(row)
    for spec in final_variant_specs(config):
        variant = str(spec.get("variant_name", ""))
        root = output_root / "baselines" / variant.replace("osnet_finetuned_", "reid_finetuned_").replace("osnet_pretrained_diagnostic", "reid_pretrained_diagnostic")
        root.mkdir(parents=True, exist_ok=True)
        write_json({"variant": spec, "artifacts": by_variant.get(variant, [])}, root / "baseline_summary.json")
