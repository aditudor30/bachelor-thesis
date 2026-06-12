"""Build the Step 20A supervised Person fragment-pair dataset."""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deep_oc_sort_3d.learned_association.fragment_loader import load_person_fragments
from deep_oc_sort_3d.learned_association.gt_fragment_matcher import (
    match_fragments_to_gt_from_dataset,
)
from deep_oc_sort_3d.learned_association.pair_balancer import balance_pairs
from deep_oc_sort_3d.learned_association.pair_candidate_builder import build_candidate_pairs
from deep_oc_sort_3d.learned_association.pair_dataset_config import (
    apply_cli_overrides,
    load_pair_dataset_config,
    output_root_from_config,
    progress_enabled,
)
from deep_oc_sort_3d.learned_association.pair_dataset_diagnostics import build_dataset_diagnostics
from deep_oc_sort_3d.learned_association.pair_dataset_figures import generate_pair_dataset_figures
from deep_oc_sort_3d.learned_association.pair_dataset_io import (
    prepare_output_tree,
    progress_iter,
    write_csv_rows,
    write_json,
)
from deep_oc_sort_3d.learned_association.pair_dataset_report import write_pair_dataset_report
from deep_oc_sort_3d.learned_association.pair_feature_builder import build_pair_features


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="deep_oc_sort_3d/configs/person_association_pair_dataset_v1.yaml",
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-pairs-per-scene", type=int)
    parser.add_argument("--debug-limit-scenes", type=int)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    return parser.parse_args()


def main() -> None:
    """Build metadata, pair features, balanced splits and diagnostics."""
    args = parse_args()
    config = load_pair_dataset_config(Path(args.config))
    apply_cli_overrides(
        config,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        progress=args.progress,
        max_pairs_per_scene=args.max_pairs_per_scene,
    )
    output_root = output_root_from_config(config)
    verdict_path = output_root / "diagnostics" / "dataset_verdict.json"
    if args.skip_existing and verdict_path.is_file():
        print("Existing completed dataset kept: %s" % output_root)
        return
    prepare_output_tree(output_root, overwrite=args.overwrite)
    progress = progress_enabled(config)

    fragments, source_summary = load_person_fragments(
        config, debug_limit_scenes=args.debug_limit_scenes, progress=progress
    )
    write_json(output_root / "metadata" / "fragment_source_summary.json", source_summary)
    gt_rows = match_fragments_to_gt_from_dataset(fragments, config, progress=progress)
    write_fragment_outputs(output_root, fragments, gt_rows)

    candidate_pairs = build_candidate_pairs(fragments, config, progress=progress)
    train_camera_pairs = {
        "__".join(
            sorted(
                (
                    str(row.get("_fragment_a", {}).get("camera_id") or ""),
                    str(row.get("_fragment_b", {}).get("camera_id") or ""),
                )
            )
        )
        for row in candidate_pairs
        if row.get("split") == "train"
    }
    pairs = []  # type: List[Dict[str, Any]]
    for candidate in progress_iter(
        candidate_pairs, "feature computation", progress, len(candidate_pairs)
    ):
        pairs.append(build_pair_features(candidate, train_camera_pairs, config))
    write_pair_split_outputs(output_root / "metadata", "candidate_pairs", pairs)

    train_pairs = [row for row in pairs if row.get("split") == "train"]
    val_pairs = [row for row in pairs if row.get("split") == "val"]
    pair_settings = config.get("pair_generation", {})
    ratio = float(pair_settings.get("negative_to_positive_ratio_balanced", 3.0))
    seed = int(config.get("person_association_pair_dataset", {}).get("random_seed", 42))
    require_reid = bool(pair_settings.get("require_valid_reid_for_balanced", True))
    balanced_results = {}  # type: Dict[str, List[Dict[str, Any]]]
    for split_name, split_pairs, split_seed in progress_iter(
        [("train", train_pairs, seed), ("val", val_pairs, seed + 1)],
        "balancing",
        progress,
        2,
    ):
        balanced_results[split_name] = balance_pairs(
            split_pairs, ratio, split_seed, require_reid
        )
    train_balanced = balanced_results["train"]
    val_balanced = balanced_results["val"]
    write_pair_outputs(output_root, train_pairs, val_pairs, train_balanced, val_balanced)

    balanced = {"train": train_balanced, "val": val_balanced}
    verdict = build_dataset_diagnostics(
        fragments, pairs, balanced, output_root, source_summary=source_summary
    )
    figure_warnings = generate_pair_dataset_figures(
        pairs,
        output_root / "figures",
        enabled=bool(config.get("figures", {}).get("enabled", True)),
    )
    if figure_warnings:
        write_json(
            output_root / "diagnostics" / "figure_warnings.json",
            {"warnings": figure_warnings},
        )
    write_pair_dataset_report(output_root, source_summary, verdict, fragments, pairs)
    print_summary(output_root, verdict, fragments, train_pairs, val_pairs, train_balanced, val_balanced)


def write_fragment_outputs(
    output_root: Path,
    fragments: List[Dict[str, Any]],
    gt_rows: List[Dict[str, Any]],
) -> None:
    """Write fragment metadata and GT match provenance."""
    metadata = output_root / "metadata"
    write_csv_rows(metadata / "fragments_all.csv", fragments)
    write_csv_rows(metadata / "fragments_train.csv", [row for row in fragments if row.get("split") == "train"])
    write_csv_rows(metadata / "fragments_val.csv", [row for row in fragments if row.get("split") == "val"])
    write_csv_rows(metadata / "gt_fragment_matches.csv", gt_rows)


def write_pair_split_outputs(
    directory: Path, prefix: str, pairs: List[Dict[str, Any]]
) -> None:
    """Write all/train/val versions of candidate pair metadata."""
    write_csv_rows(directory / (prefix + "_all.csv"), pairs)
    write_csv_rows(directory / (prefix + "_train.csv"), [row for row in pairs if row.get("split") == "train"])
    write_csv_rows(directory / (prefix + "_val.csv"), [row for row in pairs if row.get("split") == "val"])


def write_pair_outputs(
    output_root: Path,
    train_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    train_balanced: List[Dict[str, Any]],
    val_balanced: List[Dict[str, Any]],
) -> None:
    """Write raw, balanced and hard-negative pair files."""
    pairs_dir = output_root / "pairs"
    write_csv_rows(pairs_dir / "train_pairs.csv", train_pairs)
    write_csv_rows(pairs_dir / "val_pairs.csv", val_pairs)
    write_csv_rows(pairs_dir / "train_pairs_balanced.csv", train_balanced)
    write_csv_rows(pairs_dir / "val_pairs_balanced.csv", val_balanced)
    write_csv_rows(
        pairs_dir / "hard_negative_pairs_train.csv",
        [row for row in train_pairs if int(row.get("hard_negative") or 0) == 1],
    )
    write_csv_rows(
        pairs_dir / "hard_negative_pairs_val.csv",
        [row for row in val_pairs if int(row.get("hard_negative") or 0) == 1],
    )


def print_summary(
    output_root: Path,
    verdict: Dict[str, Any],
    fragments: List[Dict[str, Any]],
    train_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    train_balanced: List[Dict[str, Any]],
    val_balanced: List[Dict[str, Any]],
) -> None:
    """Print a concise completion summary."""
    print("Output root: %s" % output_root)
    print("fragments: %d" % len(fragments))
    print("valid fragments: %d" % sum(bool(row.get("valid_for_pairs")) for row in fragments))
    print("train pairs: %d" % len(train_pairs))
    print("val pairs: %d" % len(val_pairs))
    print("balanced train pairs: %d" % len(train_balanced))
    print("balanced val pairs: %d" % len(val_balanced))
    print("verdict: %s" % verdict.get("verdict"))
    print("ready_for_step_20b: %s" % verdict.get("ready_for_step_20b"))


if __name__ == "__main__":
    main()
