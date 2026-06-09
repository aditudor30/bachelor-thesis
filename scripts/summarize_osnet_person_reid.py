"""CLI for summarizing OSNet Person ReID fine-tuning outputs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.reid_training.osnet_finetune_report import summarize_osnet_finetune_output


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Summarize OSNet Person ReID fine-tuning outputs.")
    parser.add_argument("--output-root", default="output/reid_training/osnet_person_smartspaces_v1")
    return parser.parse_args()


def main() -> None:
    """Run summary from CLI."""
    args = parse_args()
    summary = summarize_osnet_finetune_output(Path(args.output_root))
    evaluation = summary.get("evaluation", {})
    verdict = (evaluation.get("verdict") or {}).get("verdict")
    print("status: %s" % summary.get("status"))
    print("output_root: %s" % summary.get("output_root"))
    print("verdict: %s" % verdict)
    print("best_train_loss: %s" % summary.get("best_train_loss"))
    print("best_val_top1: %s" % summary.get("best_val_top1"))


if __name__ == "__main__":
    main()
