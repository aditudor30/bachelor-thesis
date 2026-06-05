"""CLI for printing a compact Step 16A Person ReID summary."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.person_reid.reid_utils import read_json


def main() -> None:
    """Print summary."""
    parser = argparse.ArgumentParser(description="Summarize Person ReID diagnostics.")
    parser.add_argument("--root", type=Path, default=Path("output/reid_person/baseline_v2_pseudo3d_fullcam"))
    args = parser.parse_args()
    summary = read_json(args.root / "report" / "PERSON_REID_STEP16A_SUMMARY.json") or {}
    crop = read_json(args.root / "summaries" / "crop_extraction_summary.json") or {}
    embedding = read_json(args.root / "summaries" / "crop_embedding_summary.json") or {}
    print("verdict: %s" % summary.get("verdict"))
    print("backend: %s status=%s weights_loaded=%s" % (embedding.get("backend"), embedding.get("status"), embedding.get("weights_loaded")))
    print("crop_records: %s success_rate=%s" % (crop.get("crop_records"), crop.get("crop_success_rate")))
    print("embeddings_generated: %s dim=%s" % (embedding.get("embeddings_generated"), embedding.get("embedding_dim")))
    sim = summary.get("similarity", {}) if isinstance(summary.get("similarity", {}), dict) else {}
    print("same_gt_mean: %s different_gt_mean: %s margin: %s" % (sim.get("same_gt_similarity_mean"), sim.get("different_gt_similarity_mean"), sim.get("separation_margin")))


if __name__ == "__main__":
    main()

