"""Train and evaluate Step 20B Person association scorers."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.learned_association.pair_dataset_io import read_csv_rows, read_json, write_json
from deep_oc_sort_3d.learned_association.scorer_config import (
    apply_scorer_overrides,
    load_scorer_config,
    scorer_output_root,
    scorer_progress_enabled,
)
from deep_oc_sort_3d.learned_association.scorer_figures import generate_scorer_figures
from deep_oc_sort_3d.learned_association.scorer_io import (
    prepare_scorer_output,
    save_resolved_config,
)
from deep_oc_sort_3d.learned_association.scorer_report import write_scorer_report
from deep_oc_sort_3d.learned_association.scorer_trainer import train_all_scorers


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="deep_oc_sort_3d/configs/person_association_scorer_v1.yaml",
    )
    parser.add_argument("--pair-dataset-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    return parser.parse_args()


def main() -> None:
    """Train all enabled scorers and write validation diagnostics."""
    args = parse_args()
    config = load_scorer_config(Path(args.config))
    apply_scorer_overrides(
        config,
        pair_dataset_root=args.pair_dataset_root,
        output_root=args.output_root,
        device=args.device,
        epochs=args.epochs,
        progress=args.progress,
    )
    output_root = scorer_output_root(config)
    prepare_scorer_output(output_root, overwrite=args.overwrite)
    save_resolved_config(config, output_root)
    evaluation = train_all_scorers(
        config, output_root, progress=scorer_progress_enabled(config)
    )
    threshold_rows = read_csv_rows(output_root / "evaluation" / "threshold_sweep_all_models.csv")
    history = read_csv_rows(output_root / "evaluation" / "mlp_training_history.csv")
    importance = read_json(output_root / "evaluation" / "feature_importance.json", {}) or {}
    per_camera = read_csv_rows(output_root / "evaluation" / "per_camera_pair_metrics.csv")
    figure_warnings = generate_scorer_figures(
        evaluation.get("score_map", {}),
        evaluation.get("labels"),
        output_root / "figures",
        threshold_rows,
        history=history,
        feature_importance=importance,
        per_camera_rows=per_camera,
        enabled=bool(config.get("figures", {}).get("enabled", True)),
    )
    if figure_warnings:
        write_json(output_root / "evaluation" / "figure_warnings.json", {"warnings": figure_warnings})
    write_scorer_report(output_root, evaluation, config)
    print_summary(output_root, evaluation)


def print_summary(output_root: Path, evaluation: dict) -> None:
    """Print high-signal scorer results."""
    selection = evaluation.get("selection", {})
    verdict = evaluation.get("verdict", {})
    print("Output root: %s" % output_root)
    print("models evaluated: %s" % ", ".join(sorted(evaluation.get("model_evaluations", {}).keys())))
    print("selected model: %s" % selection.get("selected_model"))
    print("selected strict threshold: %s" % selection.get("selected_thresholds", {}).get("strict", {}).get("threshold"))
    print("verdict: %s" % verdict.get("verdict"))
    print("ready_for_step_20c: %s" % verdict.get("ready_for_step_20c"))


if __name__ == "__main__":
    main()
