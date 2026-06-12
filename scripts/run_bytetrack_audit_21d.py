"""Run the complete Step 21D ByteTrack diagnostic audit."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_audit.audit_config import load_audit_config
from deep_oc_sort_3d.bytetrack_audit.audit_report import run_complete_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ByteTrack lifecycle and filtering audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--artifact-only", action="store_true")
    parser.add_argument("--instrumented-mini-rerun", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    result = run_complete_audit(
        config=load_audit_config(args.config),
        progress=args.progress,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        artifact_only=args.artifact_only,
        instrumented_mini_rerun=args.instrumented_mini_rerun,
        max_samples=args.max_samples,
    )
    print("verdict: %s" % result.get("verdict", {}).get("label"))
    print("recommended_step_21e: %s" % result.get("verdict", {}).get("recommended_step_21e"))


if __name__ == "__main__":
    main()

