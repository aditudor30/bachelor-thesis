"""CLI for evaluating OSNet Person ReID checkpoints."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.reid_training.osnet_finetune_config import (
    load_osnet_finetune_config,
    prepare_output_dirs,
    save_resolved_config,
    summarize_environment,
)
from deep_oc_sort_3d.reid_training.osnet_finetune_report import summarize_osnet_finetune_output
from deep_oc_sort_3d.reid_training.osnet_finetune_trainer import evaluate_pretrained_vs_finetuned


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned OSNet Person ReID.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/osnet_person_smartspaces_finetune.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    """Run evaluation from CLI."""
    args = parse_args()
    overrides: Dict[str, Any] = {
        "output_root": args.output_root,
        "weights": args.weights,
        "device": args.device,
    }
    config = load_osnet_finetune_config(Path(args.config), overrides=overrides)
    output_root = prepare_output_dirs(config, overwrite=bool(args.overwrite))
    save_resolved_config(config, output_root)
    summarize_environment(config, output_root)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output_root / "checkpoints" / "best_retrieval_top1.pth"
    summary = evaluate_pretrained_vs_finetuned(
        config,
        checkpoint_path=checkpoint,
        output_root=output_root,
        progress=bool(args.progress),
    )
    summarize_osnet_finetune_output(output_root)
    verdict = (summary.get("verdict") or {}).get("verdict")
    print("status: %s" % summary.get("status"))
    print("verdict: %s" % verdict)
    print("checkpoint_loaded: %s" % ((summary.get("checkpoint") or {}).get("loaded")))
    print("output_root: %s" % output_root)


if __name__ == "__main__":
    main()
