"""Run Step 15B: class priors and pseudo-3D design artifacts."""

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from deep_oc_sort_3d.audit3d.audit3d_io import ensure_clean_output_dir, write_csv, write_json, write_markdown
from deep_oc_sort_3d.audit3d.generic_3d_audit import read_generic_export_rows
from deep_oc_sort_3d.priors3d.class_priors_builder import (
    build_class_priors_report,
    build_final_class_priors,
    final_priors_to_rows,
)
from deep_oc_sort_3d.priors3d.dimension_prior_analysis import (
    build_dimension_comparison_report,
    compare_priors_to_generic_rows,
    comparison_to_rows,
)
from deep_oc_sort_3d.priors3d.priors_io import read_prior_csv, read_prior_json
from deep_oc_sort_3d.priors3d.pseudo3d_design_report import (
    build_pseudo3d_estimator_design,
    build_pseudo3d_validation_plan,
    build_roadmap_baseline_v2,
    build_step15b_summary,
    build_step15b_summary_markdown,
)
from deep_oc_sort_3d.priors3d.source_metadata_schema import (
    build_source_metadata_schema,
    build_source_metadata_schema_markdown,
)


def run_step15b(args: Any) -> Dict[str, Any]:
    """Run all Step 15B artifact generation."""
    config = _load_config(args.config)
    output_root = Path(config.get("step15b", {}).get("output_root", "output/3d_priors/baseline_v1_geometry_only"))
    progress = bool(args.progress if args.progress is not None else config.get("step15b", {}).get("progress", True))
    ensure_clean_output_dir(output_root, overwrite=args.overwrite)
    paths = config.get("paths", {})

    priors_summary = _build_priors(output_root, paths, config, progress)
    comparison_summary = _build_comparison(output_root, paths, config, priors_summary, progress)
    schema = _build_schema(output_root)
    _build_design_docs(output_root, config.get("pseudo3d_design", {}))
    summary = build_step15b_summary(priors_summary, comparison_summary, schema)
    write_json(summary, output_root / "summary" / "step15b_summary.json")
    write_markdown(build_step15b_summary_markdown(summary), output_root / "summary" / "step15b_summary.md")
    print("Wrote Step 15B outputs to %s" % output_root)
    return summary


def _build_priors(output_root: Path, paths: Dict[str, Any], config: Dict[str, Any], progress: bool) -> Dict[str, Any]:
    prior_json = read_prior_json(paths.get("class_priors_input_json", ""))
    prior_csv_rows = read_prior_csv(paths.get("class_priors_input_csv", ""), show_progress=progress)
    priors_summary = build_final_class_priors(
        prior_json=prior_json,
        prior_csv_rows=prior_csv_rows,
        classes=config.get("classes", []),
        config=config.get("priors", {}),
    )
    out = output_root / "priors"
    write_json(priors_summary, out / "class_dimension_priors_final.json")
    write_csv(final_priors_to_rows(priors_summary), out / "class_dimension_priors_final.csv")
    write_markdown(build_class_priors_report(priors_summary), out / "class_dimension_priors_report.md")
    return priors_summary


def _build_comparison(
    output_root: Path,
    paths: Dict[str, Any],
    config: Dict[str, Any],
    priors_summary: Dict[str, Any],
    progress: bool,
) -> Dict[str, Any]:
    generic_rows = read_generic_export_rows(paths.get("generic_export_root", ""), show_progress=progress)
    comparison = compare_priors_to_generic_rows(priors_summary, generic_rows, config.get("comparison", {}))
    out = output_root / "comparisons"
    write_json(comparison, out / "dimension_prior_vs_test_comparison.json")
    write_csv(comparison_to_rows(comparison), out / "dimension_prior_vs_test_comparison.csv")
    write_markdown(build_dimension_comparison_report(comparison), out / "dimension_prior_vs_test_comparison.md")
    return comparison


def _build_schema(output_root: Path) -> Dict[str, Any]:
    schema = build_source_metadata_schema()
    out = output_root / "metadata_schema"
    write_json(schema, out / "source_metadata_schema.json")
    write_markdown(build_source_metadata_schema_markdown(schema), out / "SOURCE_METADATA_SCHEMA.md")
    return schema


def _build_design_docs(output_root: Path, config: Dict[str, Any]) -> None:
    out = output_root / "design"
    write_markdown(build_pseudo3d_estimator_design(config), out / "PSEUDO3D_ESTIMATOR_DESIGN.md")
    write_markdown(build_pseudo3d_validation_plan(), out / "PSEUDO3D_VALIDATION_PLAN.md")
    write_markdown(build_roadmap_baseline_v2(), out / "ROADMAP_BASELINE_V2_PSEUDO3D.md")


def _load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 15B pseudo-3D design artifact generation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_step15b(args)


if __name__ == "__main__":
    main()

