"""Run only the Step 21D motion-filter audit."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_audit.audit_config import load_audit_config, write_resolved_config
from deep_oc_sort_3d.bytetrack_audit.motion_filter_audit import run_motion_filter_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ByteTrack motion-filter audit")
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
    result = run_motion_filter_audit(config, progress=args.progress)
    print("status: %s" % result.get("status"))
    for row in result.get("summary_rows", []):
        print("%s rejection_rate=%s" % (row.get("variant_name"), row.get("rejection_rate")))


if __name__ == "__main__":
    main()

