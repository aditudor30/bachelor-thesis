"""CLI for OSNet Person ReID fine-tuning."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.reid_training.osnet_finetune_report import summarize_osnet_finetune_output
from deep_oc_sort_3d.reid_training.osnet_finetune_trainer import train_osnet_person_reid


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune OSNet on SmartSpaces Person ReID crops.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/osnet_person_smartspaces_finetune.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    """Run fine-tuning from CLI."""
    args = parse_args()
    overrides: Dict[str, Any] = {
        "output_root": args.output_root,
        "weights": args.weights,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    summary = train_osnet_person_reid(
        Path(args.config),
        overrides=overrides,
        progress=bool(args.progress),
        overwrite=bool(args.overwrite),
        resume=Path(args.resume) if args.resume else None,
    )
    summarize_osnet_finetune_output(Path(str(summary.get("output_root", ""))))
    print("status: %s" % summary.get("status"))
    print("output_root: %s" % summary.get("output_root"))
    print("num_train_identities: %s" % summary.get("num_train_identities"))
    print("best_val_top1: %s" % summary.get("best_val_top1"))


if __name__ == "__main__":
    main()
