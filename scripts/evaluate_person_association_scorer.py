"""Re-evaluate saved Step 20B scorers without retraining."""

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
from deep_oc_sort_3d.learned_association.scorer_report import write_scorer_report
from deep_oc_sort_3d.learned_association.scorer_trainer import evaluate_saved_scorers


def main() -> None:
    """Load saved models and regenerate validation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="deep_oc_sort_3d/configs/person_association_scorer_v1.yaml",
    )
    parser.add_argument("--model-root")
    parser.add_argument("--overwrite", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=None)
    args = parser.parse_args()
    config = load_scorer_config(Path(args.config))
    apply_scorer_overrides(config, output_root=args.model_root, progress=args.progress)
    output_root = scorer_output_root(config)
    if not output_root.is_dir():
        raise FileNotFoundError("Scorer model root not found: %s" % output_root)
    evaluation = evaluate_saved_scorers(
        config, output_root, progress=scorer_progress_enabled(config)
    )
    preprocessing = read_json(output_root / "data" / "feature_preprocessing_summary.json", {}) or {}
    evaluation["data_summary"] = preprocessing
    threshold_rows = read_csv_rows(output_root / "evaluation" / "threshold_sweep_all_models.csv")
    history = read_csv_rows(output_root / "evaluation" / "mlp_training_history.csv")
    importance = read_json(output_root / "evaluation" / "feature_importance.json", {}) or {}
    per_camera = read_csv_rows(output_root / "evaluation" / "per_camera_pair_metrics.csv")
    warnings = generate_scorer_figures(
        evaluation.get("score_map", {}),
        evaluation.get("labels"),
        output_root / "figures",
        threshold_rows,
        history=history,
        feature_importance=importance,
        per_camera_rows=per_camera,
        enabled=bool(config.get("figures", {}).get("enabled", True)),
    )
    if warnings:
        write_json(output_root / "evaluation" / "figure_warnings.json", {"warnings": warnings})
    write_scorer_report(output_root, evaluation, config)
    print("selected model: %s" % evaluation.get("selection", {}).get("selected_model"))
    print("verdict: %s" % evaluation.get("verdict", {}).get("verdict"))


if __name__ == "__main__":
    main()
