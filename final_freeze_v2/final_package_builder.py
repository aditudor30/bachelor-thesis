"""Build compact final freeze v2 packages."""

import shutil
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.final_freeze_v2.final_checksum_utils import checksum_rows_for_files, write_checksum_text
from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_config import output_root_from_config
from deep_oc_sort_3d.final_freeze_v2.final_freeze_v2_io import copy_file_if_exists, find_first_track1, load_yaml, progress_iter, write_csv_rows, write_json, write_text
from deep_oc_sort_3d.final_freeze_v2.final_variant_registry import final_variant_specs


def package_final_freeze_v2_from_config(config_path: Path, show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Build all final packages."""
    config = load_yaml(Path(config_path))
    output_root = output_root_from_config(config)
    packages_root = output_root / "packages"
    specs = package_specs(config, output_root)
    packages = []
    for spec in progress_iter(specs, show_progress, "package final freeze v2", "package"):
        packages.append(build_package(spec, packages_root, config, show_progress=show_progress, overwrite=overwrite))
    write_json({"packages": packages}, packages_root / "package_manifest.json")
    write_csv_rows(flatten_package_rows(packages), packages_root / "package_manifest.csv")
    return {"packages": packages}


def package_specs(config: Dict[str, Any], output_root: Path) -> List[Dict[str, Any]]:
    """Return package specs."""
    variants = {spec["variant_name"]: spec for spec in final_variant_specs(config)}
    return [
        {
            "name": "submission_safe_v1",
            "variant": "v1_geometry_only",
            "title": "Submission-safe V1 geometry-only baseline",
            "description": "Validated compact Track1 fallback.",
            "variant_spec": variants.get("v1_geometry_only", {}),
        },
        {
            "name": "mvp_3d_v2",
            "variant": "v2_pseudo3d_fullcam_current",
            "title": "V2 pseudo3D fullcam MVP",
            "description": "Main 3D provenance-backed MVP package.",
            "variant_spec": variants.get("v2_pseudo3d_fullcam_current", {}),
        },
        {
            "name": "compact_safe_v2",
            "variant": "v2_export_compact",
            "title": "V2 compact safe variant",
            "description": "Conservative compact export variant.",
            "variant_spec": variants.get("v2_export_compact", {}),
        },
        {
            "name": "reid_experimental_combined_safe_080",
            "variant": "osnet_finetuned_combined_safe_080",
            "title": "Experimental fine-tuned ReID combined_safe_080",
            "description": "ReID-enhanced experimental final variant.",
            "variant_spec": variants.get("osnet_finetuned_combined_safe_080", {}),
        },
        {
            "name": "thesis_assets",
            "variant": "all",
            "title": "Thesis assets",
            "description": "Tables, figures, reports and manifests for thesis writing.",
            "asset_package": True,
        },
    ]


def build_package(spec: Dict[str, Any], packages_root: Path, config: Dict[str, Any], show_progress: bool = True, overwrite: bool = False) -> Dict[str, Any]:
    """Build one compact package."""
    root = Path(packages_root) / str(spec.get("name", "package"))
    if overwrite and root.exists():
        shutil.rmtree(str(root))
    root.mkdir(parents=True, exist_ok=True)
    copied = []
    if bool(spec.get("asset_package", False)):
        entries = thesis_asset_entries(packages_root.parent)
    else:
        entries = variant_package_entries(spec, packages_root.parent)
    for entry in entries:
        source = Path(str(entry.get("source", "")))
        destination = root / str(entry.get("relative", source.name))
        copied_path = copy_file_if_exists(source, destination)
        copied.append(
            {
                "kind": entry.get("kind", ""),
                "source": str(source),
                "destination": "" if copied_path is None else str(copied_path),
                "exists": copied_path is not None,
            }
        )
    write_package_readme(spec, root)
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    package_cfg = config.get("packages", {})
    checksum_rows = checksum_rows_for_files(
        files,
        root,
        show_progress=show_progress,
        large_file_mb_threshold=float(package_cfg.get("large_file_mb_threshold", 200)),
        checksum_large_files=bool(package_cfg.get("checksum_large_files", False)),
    )
    write_csv_rows(checksum_rows, root / "checksums.csv")
    write_checksum_text(checksum_rows, root / "CHECKSUMS.sha256")
    write_json({"package": spec.get("name"), "files": copied, "checksums": checksum_rows}, root / "manifest.json")
    return {"name": spec.get("name"), "root": str(root), "files": copied, "num_files": len(files)}


def variant_package_entries(spec: Dict[str, Any], output_root: Path) -> List[Dict[str, Any]]:
    """Build package entries for one variant."""
    variant = spec.get("variant_spec", {})
    track1 = find_first_track1(Path(str(variant.get("track1_root", ""))))
    entries = [
        {"kind": "summary", "source": output_root / "tables" / "final_variant_comparison.csv", "relative": "tables/final_variant_comparison.csv"},
        {"kind": "report", "source": output_root / "reports" / "FINAL_PROJECT_FREEZE_REPORT.md", "relative": "reports/FINAL_PROJECT_FREEZE_REPORT.md"},
        {"kind": "reproducibility", "source": output_root / "reports" / "FINAL_REPRODUCIBILITY.md", "relative": "reports/FINAL_REPRODUCIBILITY.md"},
    ]
    if track1 is not None:
        entries.insert(0, {"kind": "track1", "source": track1, "relative": "track1.txt"})
    return entries


def thesis_asset_entries(output_root: Path) -> List[Dict[str, Any]]:
    """Collect compact thesis asset files."""
    entries = []
    for root_name in ["tables", "reports", "figures/captions", "manifests"]:
        root = output_root / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                entries.append({"kind": root_name, "source": path, "relative": str(path.relative_to(output_root))})
    return entries


def write_package_readme(spec: Dict[str, Any], root: Path) -> None:
    """Write package README."""
    lines = [
        "# %s" % spec.get("title", spec.get("name", "Package")),
        "",
        str(spec.get("description", "")),
        "",
        "This compact package contains references, reports and small artifacts only. Large intermediate detections, videos and H5 files are intentionally not copied.",
    ]
    write_text(lines, root / "README.md")


def flatten_package_rows(packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten package manifest rows."""
    rows = []
    for package in packages:
        for row in package.get("files", []):
            copied = dict(row)
            copied["package"] = package.get("name")
            rows.append(copied)
    return rows

