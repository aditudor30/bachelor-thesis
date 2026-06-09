"""CLI for Step 18C fine-tuned Person ReID association sweep."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_sweep import run_finetuned_person_reid_association_sweep


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run fine-tuned Person ReID association sweep.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/person_reid_finetuned_association.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    """Run Step 18C."""
    args = parse_args()
    config_path = Path(args.config)
    if args.dataset_root is not None:
        from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import load_finetuned_association_config, write_yaml

        config = load_finetuned_association_config(config_path)
        paths = dict(config.get("paths", {}))
        paths["dataset_root"] = str(args.dataset_root)
        config["paths"] = paths
        config_path = Path("output/person_reid_finetuned_association_runtime_config.yaml")
        write_yaml(config, config_path)
    summary = run_finetuned_person_reid_association_sweep(
        config_path,
        progress=bool(args.progress),
        overwrite=bool(args.overwrite),
        run_embeddings=not bool(args.skip_embeddings),
        run_scoring=not bool(args.skip_scoring),
        run_sweep=not bool(args.skip_sweep),
    )
    selected = summary.get("selected_variant") or {}
    print("status: %s" % summary.get("status"))
    print("output_root: %s" % summary.get("output_root"))
    print("verdict: %s" % selected.get("verdict"))
    print("best_run: %s" % selected.get("best_run"))


if __name__ == "__main__":
    main()
