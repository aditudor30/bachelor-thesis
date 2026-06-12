"""Print a compact summary of Step 20B scorer results."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.learned_association.pair_dataset_io import read_csv_rows, read_json


def main() -> None:
    """Read scorer artifacts without training or evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    verdict = read_json(root / "evaluation" / "scorer_verdict.json", {}) or {}
    selection = read_json(root / "models" / "selected_model.json", {}) or {}
    preprocessing = read_json(root / "data" / "feature_preprocessing_summary.json", {}) or {}
    comparison = read_csv_rows(root / "evaluation" / "model_comparison.csv")
    print("root: %s" % root)
    print("verdict: %s" % verdict.get("verdict"))
    print("ready_for_step_20c: %s" % verdict.get("ready_for_step_20c"))
    print("selected_model: %s" % selection.get("selected_model"))
    print("selected_thresholds: %s" % selection.get("selected_thresholds"))
    print("train_pairs: %s" % preprocessing.get("num_train_pairs"))
    print("val_pairs: %s" % preprocessing.get("num_val_pairs"))
    print("input_dim: %s" % preprocessing.get("input_dim"))
    print("model comparison:")
    for row in comparison:
        print(
            "  %s pr_auc=%s roc_auc=%s precision=%s recall=%s fpr=%s hard_fpr=%s"
            % (
                row.get("model_name"),
                row.get("pr_auc"),
                row.get("roc_auc"),
                row.get("precision"),
                row.get("recall"),
                row.get("false_positive_rate"),
                row.get("hard_negative_false_positive_rate"),
            )
        )
    reasons = verdict.get("reasons") or []
    if reasons:
        print("reasons: %s" % ", ".join(str(item) for item in reasons))


if __name__ == "__main__":
    main()
