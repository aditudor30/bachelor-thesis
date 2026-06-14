"""Find or reconstruct GT-independent validation Track1-like predictions for Step 23A."""

import argparse
import shutil
from pathlib import Path

from deep_oc_sort_3d.official_failure_audit.failure_audit_config import load_failure_audit_config, output_root, progress_default, resolved_config
from deep_oc_sort_3d.official_failure_audit.failure_io import write_yaml
from deep_oc_sort_3d.official_failure_audit.val_prediction_source_finder import build_val_prediction_files


def main() -> None:
    args = _parser().parse_args()
    config = load_failure_audit_config(Path(args.config))
    progress = progress_default(config) if args.progress is None else bool(args.progress)
    root = output_root(config)
    directory = root / "val_predictions"
    if directory.exists() and args.overwrite:
        shutil.rmtree(str(directory))
    elif directory.exists() and args.skip_existing:
        print("23A val predictions skipped existing output: %s" % directory)
        return
    elif directory.exists():
        raise FileExistsError("Val prediction output exists; use --overwrite or --skip-existing: %s" % directory)
    write_yaml(root / "configs" / "resolved_config.yaml", resolved_config(config))
    rows, summary = build_val_prediction_files(config, root, progress=progress)
    print("23A selected variant: %s" % summary.get("selected_variant"))
    print("23A selected rows: %d" % len(rows))
    print("23A source status: %s" % summary.get("status"))


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
