"""Summarize local tracking outputs and optional evaluation files."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def summarize_local_tracking(args: Any) -> None:
    """Print compact local tracking run summary."""
    summary_rows = _read_csv(args.tracking_root / "summaries" / "local_tracking_summary.csv")
    eval_jsons = sorted((args.tracking_root / "eval").rglob("*.json"))
    print("tracking_root: %s" % args.tracking_root)
    if summary_rows:
        print("files: %d" % len(summary_rows))
        print("total records: %d" % sum(_int(row.get("num_track_records")) for row in summary_rows))
        print("total observations: %d" % sum(_int(row.get("num_observations")) for row in summary_rows))
        print("errors: %d" % len([row for row in summary_rows if row.get("status") == "error"]))
        print("per subset: %s" % json.dumps(_per_subset(summary_rows), sort_keys=True))
    if eval_jsons:
        print("eval files: %d" % len(eval_jsons))
        for path in eval_jsons[:10]:
            data = json.loads(path.read_text(encoding="utf-8"))
            print(
                "  %s: tracks=%s records=%s purity=%s ids=%s frags=%s"
                % (
                    path,
                    data.get("num_tracks"),
                    data.get("num_records"),
                    data.get("purity_mean"),
                    data.get("id_switches_approx"),
                    data.get("fragmentations_approx"),
                )
            )
    if not summary_rows and not eval_jsons:
        print("No summary/eval files found.")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _per_subset(rows: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {}
    for row in rows:
        subset = row.get("subset", "")
        counts[subset] = counts.get(subset, 0) + _int(row.get("num_track_records"))
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize local tracking outputs.")
    parser.add_argument("--tracking-root", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize_local_tracking(args)


if __name__ == "__main__":
    main()
