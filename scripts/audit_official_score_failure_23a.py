"""Run the complete Step 23A official score failure audit."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.official_failure_audit.failure_audit_config import (
    load_failure_audit_config,
    output_root,
    progress_default,
    resolved_config,
)
from deep_oc_sort_3d.official_failure_audit.failure_diagnostics import write_failure_diagnostics
from deep_oc_sort_3d.official_failure_audit.failure_figures import write_failure_figures
from deep_oc_sort_3d.official_failure_audit.failure_io import prepare_output_root, write_yaml
from deep_oc_sort_3d.official_failure_audit.failure_report import write_failure_report
from deep_oc_sort_3d.official_failure_audit.gt_parser import parse_val_ground_truth
from deep_oc_sort_3d.official_failure_audit.gt_track1_exporter import export_gt_track1_like
from deep_oc_sort_3d.official_failure_audit.hypothesis_sweep import run_hypothesis_sweep, write_original_matching
from deep_oc_sort_3d.official_failure_audit.track1_validator import audit_prediction_rows
from deep_oc_sort_3d.official_failure_audit.val_prediction_source_finder import build_val_prediction_files


def main() -> None:
    args = _parser().parse_args()
    config = load_failure_audit_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    root = output_root(config)
    if not prepare_output_root(root, overwrite=args.overwrite, skip_existing=args.skip_existing):
        print("23A audit skipped existing output: %s" % root)
        return
    write_yaml(root / "configs" / "resolved_config.yaml", resolved_config(config))
    gt_rows, gt_summary = parse_val_ground_truth(config, root, progress=progress)
    export_gt_track1_like(gt_rows, root, config)
    pred_rows, source_summary = build_val_prediction_files(config, root, progress=progress)
    pred_audit = audit_prediction_rows(pred_rows, root, source_summary)
    original, original_details = write_original_matching(pred_rows, gt_rows, config, root)
    sweep = run_hypothesis_sweep(pred_rows, gt_rows, config, root, progress=progress)
    diagnosis = write_failure_diagnostics(original, sweep, source_summary, root)
    figures = {"status": "disabled", "files": []}
    if bool(config.get("figures", {}).get("enabled", True)):
        figures = write_failure_figures(pred_rows, gt_rows, original_details, sweep, root)
    prediction_summary = dict(source_summary)
    prediction_summary["validation"] = pred_audit
    write_failure_report(config, gt_summary, prediction_summary, original, sweep, diagnosis, figures, root)
    print("23A verdict: %s" % diagnosis.get("verdict"))
    print("23A best hypothesis: %s" % sweep.get("best", {}).get("hypothesis"))
    print("23A report: %s" % (root / "comparison" / "OFFICIAL_SCORE_FAILURE_AUDIT_23A_REPORT.md"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--all", action="store_true", help="Accepted for explicit all-in-one invocation.")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
