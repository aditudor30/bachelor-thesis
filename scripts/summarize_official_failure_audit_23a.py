"""Print the high-signal summary of an existing Step 23A audit."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.official_failure_audit.failure_report import summarize_existing_root


def main() -> None:
    args = _parser().parse_args()
    values = summarize_existing_root(Path(args.root))
    summary = values.get("summary", {})
    verdict = values.get("verdict", {})
    original = summary.get("original_matching", {})
    print("verdict: %s" % verdict.get("verdict"))
    print("original_match_rate_at_2m: %s" % original.get("match_rate_at_2m"))
    print("best_hypothesis: %s" % verdict.get("best_hypothesis", {}).get("hypothesis"))
    print("recommended_v6_fix: %s" % verdict.get("recommended_v6_fix"))
    print("upload_recommendation: %s" % verdict.get("upload_recommendation"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser


if __name__ == "__main__":
    main()
