"""Build SmartSpaces Person ReID crop dataset and diagnostics."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.reid_training.person_crop_dataset import build_person_reid_dataset_from_config
from deep_oc_sort_3d.reid_training.reid_dataset_config import load_person_reid_dataset_config, output_root_from_config
from deep_oc_sort_3d.reid_training.reid_dataset_diagnostics import build_reid_dataset_diagnostics_from_config
from deep_oc_sort_3d.reid_training.reid_dataset_figures import build_reid_dataset_figures_from_config
from deep_oc_sort_3d.reid_training.reid_dataset_io import write_json
from deep_oc_sort_3d.reid_training.reid_dataset_report import write_reid_dataset_report_from_config
from deep_oc_sort_3d.reid_training.reid_pair_triplet_builder import build_pairs_triplets_from_config


def build_person_reid_dataset(config_path: Path, overrides: Dict[str, Any], progress: bool, overwrite: bool) -> Dict[str, Any]:
    """Run crop extraction, pair/triplet generation, diagnostics, figures, and report."""
    config = load_person_reid_dataset_config(config_path, overrides)
    output_root = output_root_from_config(config)
    crop_summary = build_person_reid_dataset_from_config(config, show_progress=progress, overwrite=overwrite)
    pair_summary = build_pairs_triplets_from_config(config, overwrite=overwrite)
    diagnostics = build_reid_dataset_diagnostics_from_config(config)
    figures = build_reid_dataset_figures_from_config(config)
    report = write_reid_dataset_report_from_config(config)
    summary = {
        "config_path": str(config_path),
        "output_root": str(output_root),
        "crop_extraction": crop_summary,
        "pairs_triplets": pair_summary,
        "diagnostics": diagnostics,
        "figures": figures,
        "report": report,
        "verdict": diagnostics.get("verdict"),
    }
    write_json(summary, output_root / "diagnostics" / "run_summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Build SmartSpaces Person ReID dataset.")
    parser.add_argument("--config", default="deep_oc_sort_3d/configs/person_reid_dataset_v1.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--max-crops-per-identity", type=int, default=None)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.set_defaults(progress=True)
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    overrides: Dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "output_root": args.output_root,
        "frame_stride": args.frame_stride,
        "max_crops_per_identity": args.max_crops_per_identity,
        "overwrite": bool(args.overwrite),
        "skip_existing": bool(args.skip_existing),
    }
    summary = build_person_reid_dataset(Path(args.config), overrides, progress=bool(args.progress), overwrite=bool(args.overwrite))
    diagnostics = summary.get("diagnostics", {})
    print("output_root:", summary.get("output_root"))
    print("verdict:", summary.get("verdict"))
    print("total_crops:", diagnostics.get("total_crops"))
    print("train identities:", diagnostics.get("num_train_identities"))
    print("val identities:", diagnostics.get("num_val_identities"))


if __name__ == "__main__":
    main()

