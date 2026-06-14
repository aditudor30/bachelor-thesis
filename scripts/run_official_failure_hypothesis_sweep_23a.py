"""Run Step 23A original matching and convention hypothesis sweep from prepared files."""

import argparse
import shutil
from pathlib import Path

from deep_oc_sort_3d.official_failure_audit.failure_audit_config import load_failure_audit_config, output_root, progress_default
from deep_oc_sort_3d.official_failure_audit.failure_diagnostics import write_failure_diagnostics
from deep_oc_sort_3d.official_failure_audit.failure_io import read_json
from deep_oc_sort_3d.official_failure_audit.hypothesis_sweep import run_hypothesis_sweep, write_original_matching
from deep_oc_sort_3d.official_failure_audit.track1_parser import read_track1_like


def main() -> None:
    args = _parser().parse_args()
    config = load_failure_audit_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    root = output_root(config)
    if args.skip_existing and (root / "matching_original" / "original_match_summary.json").is_file() and (root / "hypothesis_sweep" / "best_hypothesis.json").is_file():
        print("23A hypothesis sweep skipped existing completed outputs: %s" % root)
        return
    if args.skip_existing and ((root / "matching_original").exists() or (root / "hypothesis_sweep").exists()):
        raise FileExistsError("Incomplete Step 23A sweep output exists; use --overwrite to rebuild: %s" % root)
    _prepare_stage(root / "matching_original", args.overwrite, args.skip_existing)
    _prepare_stage(root / "hypothesis_sweep", args.overwrite, args.skip_existing)
    gt_rows = read_track1_like(root / "gt_track1_like" / "val_gt_track1_like.txt")
    source_summary = read_json(root / "val_predictions" / "selected_val_prediction_source.json")
    selected = str(source_summary.get("selected_variant", "v5"))
    pred_rows = read_track1_like(root / "val_predictions" / (selected + "_val_track1_like.txt"))
    original, _details = write_original_matching(pred_rows, gt_rows, config, root)
    sweep = run_hypothesis_sweep(pred_rows, gt_rows, config, root, progress=progress)
    diagnosis = write_failure_diagnostics(original, sweep, source_summary, root)
    print("23A verdict: %s" % diagnosis.get("verdict"))
    print("23A best hypothesis: %s" % sweep.get("best", {}).get("hypothesis"))


def _prepare_stage(path: Path, overwrite: bool, skip_existing: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(str(path))
    elif path.exists() and not skip_existing:
        raise FileExistsError("Stage output exists; use --overwrite or --skip-existing: %s" % path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
