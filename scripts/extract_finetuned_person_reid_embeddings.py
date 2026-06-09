"""CLI for extracting fine-tuned Person ReID embeddings."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import (
    load_finetuned_association_config,
    prepare_output_root,
    save_resolved_config,
)
from deep_oc_sort_3d.reid_finetuned_association.finetuned_embedding_extractor import extract_finetuned_person_embeddings_from_config


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Extract fine-tuned OSNet Person ReID embeddings for V2 fragments.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/person_reid_finetuned_association.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    """Run embedding extraction."""
    args = parse_args()
    config = load_finetuned_association_config(Path(args.config))
    if args.dataset_root is not None:
        paths = dict(config.get("paths", {}))
        paths["dataset_root"] = str(args.dataset_root)
        config["paths"] = paths
    output_root = prepare_output_root(config, overwrite=bool(args.overwrite))
    save_resolved_config(config, Path(args.config), output_root)
    summary = extract_finetuned_person_embeddings_from_config(config, show_progress=bool(args.progress), overwrite=bool(args.overwrite))
    print("status: %s" % summary.get("status"))
    print("crop_embeddings: %s" % summary.get("crop_embeddings"))
    print("fragment_embeddings: %s" % ((summary.get("fragment_summary") or {}).get("num_valid_fragments")))


if __name__ == "__main__":
    main()
