"""Print the compact Step 21F upload readiness summary."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.freeze_upload.freeze_io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frozen Step 21F candidates")
    parser.add_argument("--root", type=Path, default=Path("output/frozen_upload_candidates"))
    args = parser.parse_args()
    verdict = read_json(args.root / "comparison" / "verdict.json")
    readiness = read_json(args.root / "comparison" / "upload_readiness.json")
    print("verdict: %s" % verdict.get("label"))
    print("official_winner: %s" % verdict.get("official_winner"))
    for name in ["v2_current", "v3_gap_aware_soft"]:
        row = readiness.get(name, {})
        print(
            "%s ready=%s rows=%s errors=%s zip=%s"
            % (name, row.get("ready"), row.get("line_count"), row.get("validation_errors"), row.get("zip_path"))
        )
    print("recommendation: %s" % verdict.get("recommended_action"))


if __name__ == "__main__":
    main()
