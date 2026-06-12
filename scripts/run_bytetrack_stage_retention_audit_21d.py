"""Run only the Step 21D stage-retention audit."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_audit.audit_config import load_audit_config, write_resolved_config
from deep_oc_sort_3d.bytetrack_audit.stage_retention_audit import run_stage_retention_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ByteTrack stage-retention audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    config = load_audit_config(args.config)
    write_resolved_config(config)
    result = run_stage_retention_audit(config, progress=args.progress)
    print("stage transitions: %d" % len(result.get("rows", [])))
    print("unit mismatch warnings: %d" % len(result.get("warnings", [])))


if __name__ == "__main__":
    main()

