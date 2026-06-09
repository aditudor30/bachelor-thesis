"""CLI for comparing fine-tuned Person ReID association sweep outputs."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_figures import create_finetuned_association_figures
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import load_finetuned_association_config, output_root_from_config, prepare_output_root
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_metrics import compare_sweep_to_v2
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_report import write_finetuned_association_report
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_selector import select_finetuned_reid_variant
from deep_oc_sort_3d.reid_finetuned_association.finetuned_association_io import write_json


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Compare fine-tuned Person ReID association variants.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/person_reid_finetuned_association.yaml")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    """Run comparison only."""
    args = parse_args()
    config = load_finetuned_association_config(Path(args.config))
    output_root = prepare_output_root(config, overwrite=False)
    comparison = compare_sweep_to_v2(config, progress=bool(args.progress))
    selected = select_finetuned_reid_variant(comparison, config.get("selection", {}))
    write_json(selected, output_root / "comparison" / "selected_variant.json")
    write_finetuned_association_report(config, comparison, selected, output_root)
    create_finetuned_association_figures(output_root)
    print("output_root: %s" % output_root_from_config(config))
    print("verdict: %s" % selected.get("verdict"))
    print("best_run: %s" % selected.get("best_run"))


if __name__ == "__main__":
    main()
