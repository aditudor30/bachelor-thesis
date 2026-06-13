"""Print the final Step 22E decision and frozen candidate summary."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.v51_geometry_calibration_refit.v51_io import read_json


def main() -> None:
    args = _parser().parse_args()
    root = Path(args.root)
    dataset = read_json(root / "calibration_dataset" / "match_rate_summary.json")
    corrections = read_json(root / "learned_corrections" / "selected_corrections.json")
    selected = read_json(root / "comparison" / "selected_variant.json")
    readiness = read_json(root / "frozen_candidate" / "comparison" / "upload_readiness.json")
    candidate = readiness.get("v51_geometry_calibrated_official", {})
    print("fit_source: %s" % corrections.get("fit_source"))
    for phase in ["fit_train", "internal_holdout", "official_val"]:
        item = dataset.get(phase, {})
        print("%s matches=%s match_rate=%s" % (phase, item.get("num_matches"), item.get("match_rate")))
    print("selected_variant: %s" % selected.get("selected_variant"))
    print("verdict: %s" % selected.get("verdict"))
    print("ready: %s" % candidate.get("ready"))
    print("rows: %s" % candidate.get("rows"))
    print("unique_tracks: %s" % candidate.get("unique_tracks"))
    print("track1: %s" % candidate.get("track1_path"))
    print("zip: %s" % candidate.get("zip_path"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    main()
