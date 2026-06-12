"""Run only the Step 21D ByteTrack lifecycle audit."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.bytetrack_audit.audit_config import load_audit_config, write_resolved_config
from deep_oc_sort_3d.bytetrack_audit.lifecycle_audit import run_lifecycle_audit


def main() -> None:
    parser = _parser("Run ByteTrack lifecycle audit")
    args = parser.parse_args()
    config = load_audit_config(args.config)
    write_resolved_config(config)
    result = run_lifecycle_audit(
        config,
        progress=args.progress,
        artifact_only=args.artifact_only or not args.instrumented_mini_rerun,
        instrumented_mini_rerun=args.instrumented_mini_rerun,
    )
    print("lifecycle rows: %d" % len(result.get("rows", [])))
    print("warnings: %d" % len(result.get("warnings", [])))


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
    return parser


if __name__ == "__main__":
    main()

