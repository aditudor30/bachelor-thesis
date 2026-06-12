"""Training orchestration for Step 20B Person association scorers."""

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deep_oc_sort_3d.learned_association.mlp_pairwise_scorer import (
    build_mlp_from_config,
    checkpoint_payload,
)
from deep_oc_sort_3d.learned_association.reid_only_baseline import (
    reid_model_payload,
    reid_scores_from_rows,
)
from deep_oc_sort_3d.learned_association.scorer_config import random_seed, scorer_output_root
from deep_oc_sort_3d.learned_association.scorer_data_loader import (
    labels_from_rows,
    load_pair_splits,
    metadata_from_rows,
    validate_scene_disjointness,
)
from deep_oc_sort_3d.learned_association.scorer_evaluator import (
    binary_classification_metrics,
    calibration_metrics,
    grouped_metrics,
    hard_negative_metrics,
    sigmoid_scores,
    threshold_sweep_metrics,
    top_error_rows,
)
from deep_oc_sort_3d.learned_association.scorer_feature_preprocessor import (
    build_preprocessor_from_config,
)
from deep_oc_sort_3d.learned_association.scorer_io import (
    load_pickle,
    read_csv_rows,
    save_pickle,
    write_csv_rows,
    write_json,
)
from deep_oc_sort_3d.learned_association.sklearn_scorers import (
    model_feature_importance,
    sklearn_probability_scores,
    train_sklearn_scorers,
)
from deep_oc_sort_3d.learned_association.threshold_selector import (
    scorer_verdict,
    select_model_conservatively,
    select_threshold_labels,
)
from deep_oc_sort_3d.learned_association.pair_dataset_io import progress_iter


def prepare_training_data(
    config: Dict[str, Any], output_root: Path, progress: bool = True
) -> Dict[str, Any]:
    """Load pair CSVs, fit preprocessing and persist matrices/provenance."""
    train_rows, val_rows = load_pair_splits(config, progress=progress)
    split_check = validate_scene_disjointness(train_rows, val_rows)
    if not split_check["scene_disjoint"]:
        raise ValueError("Train/val scene leakage detected: %s" % split_check["overlap_scenes"])
    preprocessor = build_preprocessor_from_config(config)
    train_x = preprocessor.fit_transform(train_rows)
    val_x = preprocessor.transform(val_rows)
    train_y = labels_from_rows(train_rows)
    val_y = labels_from_rows(val_rows)
    train_metadata = metadata_from_rows(train_rows)
    val_metadata = metadata_from_rows(val_rows)
    data_dir = output_root / "data"
    np.save(str(data_dir / "train_matrix.npy"), train_x)
    np.save(str(data_dir / "train_labels.npy"), train_y)
    np.save(str(data_dir / "val_matrix.npy"), val_x)
    np.save(str(data_dir / "val_labels.npy"), val_y)
    write_csv_rows(data_dir / "train_metadata.csv", train_metadata)
    write_csv_rows(data_dir / "val_metadata.csv", val_metadata)
    save_pickle(data_dir / "feature_scaler.pkl", preprocessor)
    summary = preprocessor.summary()
    summary.update(
        {
            "num_train_pairs": len(train_rows),
            "num_val_pairs": len(val_rows),
            "train_positive_count": int(np.sum(train_y == 1)),
            "val_positive_count": int(np.sum(val_y == 1)),
            "scene_split": split_check,
        }
    )
    write_json(data_dir / "selected_features.json", {"features": preprocessor.output_features})
    write_json(data_dir / "feature_preprocessing_summary.json", summary)
    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_x": train_x,
        "val_x": val_x,
        "train_y": train_y,
        "val_y": val_y,
        "train_metadata": train_metadata,
        "val_metadata": val_metadata,
        "preprocessor": preprocessor,
        "summary": summary,
    }


def train_all_scorers(
    config: Dict[str, Any], output_root: Path, progress: bool = True
) -> Dict[str, Any]:
    """Train enabled models and evaluate all of them on validation pairs."""
    seed_everything(random_seed(config))
    data = prepare_training_data(config, output_root, progress)
    reid_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    write_json(output_root / "models" / "reid_only_baseline.json", reid_model_payload(reid_thresholds))

    scores = {}  # type: Dict[str, np.ndarray]
    model_objects = {}  # type: Dict[str, Any]
    warnings = {}  # type: Dict[str, str]
    if bool(config.get("models", {}).get("train_reid_only_baseline", True)):
        scores["reid_only_baseline"] = reid_scores_from_rows(data["val_rows"])

    sklearn_models, sklearn_warnings = train_sklearn_scorers(
        data["train_x"], data["train_y"], config, output_root / "models"
    )
    warnings.update(sklearn_warnings)
    for model_name, model in sklearn_models.items():
        scores[model_name] = sklearn_probability_scores(model, data["val_x"])
        model_objects[model_name] = model

    history = []  # type: List[Dict[str, Any]]
    if bool(config.get("models", {}).get("train_mlp", True)):
        mlp_result = train_mlp(
            data["train_x"],
            data["train_y"],
            data["val_x"],
            data["val_y"],
            config,
            output_root,
            progress,
        )
        scores["mlp_pairwise_scorer"] = mlp_result["val_scores"]
        model_objects["mlp_pairwise_scorer"] = mlp_result["model"]
        history = mlp_result["history"]

    evaluation = evaluate_score_map(
        scores,
        data["val_y"],
        data["val_metadata"],
        config,
        output_root,
        progress=progress,
    )
    importance = {}
    for model_name, model in sklearn_models.items():
        importance[model_name] = model_feature_importance(
            model, data["preprocessor"].output_features
        )
    write_json(output_root / "evaluation" / "feature_importance.json", importance)
    if history:
        write_csv_rows(output_root / "evaluation" / "mlp_training_history.csv", history)
    write_json(output_root / "evaluation" / "training_warnings.json", warnings)
    evaluation["warnings"] = warnings
    evaluation["data_summary"] = data["summary"]
    return evaluation


def evaluate_saved_scorers(
    config: Dict[str, Any], output_root: Path, progress: bool = True
) -> Dict[str, Any]:
    """Reload saved matrices/models and regenerate validation evaluation."""
    train_x_path = output_root / "data" / "train_matrix.npy"
    val_x_path = output_root / "data" / "val_matrix.npy"
    val_y_path = output_root / "data" / "val_labels.npy"
    val_metadata_path = output_root / "data" / "val_metadata.csv"
    for path in (train_x_path, val_x_path, val_y_path, val_metadata_path):
        if not path.is_file():
            raise FileNotFoundError("Required trained-scorer artifact missing: %s" % path)
    val_x = np.load(str(val_x_path))
    val_y = np.load(str(val_y_path))
    val_metadata = read_csv_rows(val_metadata_path)
    score_map = {}  # type: Dict[str, np.ndarray]
    score_map["reid_only_baseline"] = reid_scores_from_rows(val_metadata)
    model_files = {
        "logistic_regression_l2": output_root / "models" / "logistic_regression_l2.pkl",
        "gradient_boosting": output_root / "models" / "gradient_boosting.pkl",
        "random_forest": output_root / "models" / "random_forest.pkl",
    }
    for model_name, path in progress_iter(
        list(model_files.items()), "loading saved scorers", progress, len(model_files)
    ):
        if path.is_file():
            score_map[model_name] = sklearn_probability_scores(load_pickle(path), val_x)

    mlp_path = output_root / "models" / "mlp_pairwise_scorer_best.pth"
    if mlp_path.is_file():
        import torch

        requested_device = str(config.get("mlp", {}).get("device", "cuda"))
        device = requested_device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        checkpoint = torch.load(str(mlp_path), map_location=device)
        model = build_mlp_from_config(int(checkpoint["input_dim"]), checkpoint.get("config", config))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        dummy_criterion = torch.nn.BCEWithLogitsLoss()
        scores, _ = predict_mlp(
            model,
            val_x,
            val_y,
            dummy_criterion,
            device,
            int(config.get("mlp", {}).get("batch_size", 256)),
        )
        score_map["mlp_pairwise_scorer"] = scores
    return evaluate_score_map(
        score_map, val_y, val_metadata, config, output_root, progress=progress
    )


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    config: Dict[str, Any],
    output_root: Path,
    progress: bool,
) -> Dict[str, Any]:
    """Train the main PyTorch MLP with early stopping."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    settings = config.get("mlp", {})
    requested_device = str(settings.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA unavailable; using CPU for the pairwise MLP")
        device = "cpu"
    else:
        device = requested_device
    model = build_mlp_from_config(train_x.shape[1], config).to(device)
    batch_size = int(settings.get("batch_size", 256))
    dataset = TensorDataset(
        torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
    )
    generator = torch.Generator()
    generator.manual_seed(random_seed(config))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 0.001)),
        weight_decay=float(settings.get("weight_decay", 0.0001)),
    )
    positive_count = float(np.sum(train_y == 1))
    negative_count = float(np.sum(train_y == 0))
    if bool(settings.get("auto_positive_class_weight", False)):
        pos_weight = negative_count / max(1.0, positive_count)
    else:
        pos_weight = float(settings.get("positive_class_weight", 1.0))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    epochs = int(settings.get("epochs", 50))
    patience = int(settings.get("early_stop_patience", 8))
    monitor = str(settings.get("monitor_metric", "val_pr_auc"))
    best_value = None  # type: Optional[float]
    stale_epochs = 0
    history = []  # type: List[Dict[str, Any]]
    best_path = output_root / "models" / "mlp_pairwise_scorer_best.pth"
    last_path = output_root / "models" / "mlp_pairwise_scorer_last.pth"
    for epoch in progress_iter(range(1, epochs + 1), "MLP training epochs", progress, epochs):
        model.train()
        loss_total = 0.0
        batches = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu().item())
            batches += 1
        train_loss = loss_total / float(max(1, batches))
        val_scores, val_loss = predict_mlp(model, val_x, val_y, criterion, device, batch_size)
        val_metrics = binary_classification_metrics(val_y, val_scores, 0.5)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": val_metrics.get("roc_auc"),
            "val_pr_auc": val_metrics.get("pr_auc"),
            "val_precision_050": val_metrics.get("precision"),
            "val_recall_050": val_metrics.get("recall"),
        }
        history.append(row)
        torch.save(
            checkpoint_payload(model, optimizer, epoch, train_x.shape[1], config, row),
            str(last_path),
        )
        monitored = row.get(monitor)
        if monitored is None:
            monitored = -val_loss if monitor != "val_loss" else val_loss
        maximize = monitor != "val_loss"
        improved = best_value is None or (
            float(monitored) > best_value if maximize else float(monitored) < best_value
        )
        if improved:
            best_value = float(monitored)
            stale_epochs = 0
            torch.save(
                checkpoint_payload(model, optimizer, epoch, train_x.shape[1], config, row),
                str(best_path),
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    checkpoint = torch.load(str(best_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_scores, _ = predict_mlp(model, val_x, val_y, criterion, device, batch_size)
    return {"model": model, "val_scores": final_scores, "history": history, "device": device}


def predict_mlp(
    model: Any,
    matrix: np.ndarray,
    labels: np.ndarray,
    criterion: Any,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    """Predict scores and mean loss without retaining gradients."""
    import torch

    model.eval()
    all_logits = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            end = min(len(matrix), start + batch_size)
            features = torch.from_numpy(matrix[start:end]).float().to(device)
            target = torch.from_numpy(labels[start:end]).float().to(device)
            logits = model(features)
            losses.append(float(criterion(logits, target).detach().cpu().item()))
            all_logits.append(logits.detach().cpu().numpy())
    joined = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,), dtype=np.float32)
    return sigmoid_scores(joined), float(np.mean(losses)) if losses else 0.0


def evaluate_score_map(
    score_map: Dict[str, np.ndarray],
    labels: np.ndarray,
    metadata: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    output_root: Path,
    progress: bool = True,
) -> Dict[str, Any]:
    """Evaluate scores, select thresholds/model and write core artifacts."""
    evaluation_config = config.get("evaluation", {})
    default_thresholds = configured_thresholds(evaluation_config, reid=False)
    evaluations = {}  # type: Dict[str, Dict[str, Any]]
    all_sweep_rows = []  # type: List[Dict[str, Any]]
    comparison_rows = []  # type: List[Dict[str, Any]]
    confusion = {}
    calibration = {}
    per_scene_rows = []
    per_camera_rows = []
    score_items = list(score_map.items())
    for model_name, scores in progress_iter(
        score_items, "evaluation and threshold sweep", progress, len(score_items)
    ):
        thresholds = (
            configured_thresholds(evaluation_config, reid=True)
            if model_name == "reid_only_baseline"
            else default_thresholds
        )
        sweep = threshold_sweep_metrics(model_name, labels, scores, thresholds)
        selected = select_threshold_labels(sweep, evaluation_config)
        strict_threshold = float(selected.get("strict", {}).get("threshold", 0.5))
        overall = binary_classification_metrics(labels, scores, strict_threshold)
        hard = hard_negative_metrics(labels, scores, metadata, strict_threshold)
        result = {
            "model_name": model_name,
            "overall_metrics": overall,
            "selected_thresholds": selected,
            "hard_negative_metrics": hard,
        }
        evaluations[model_name] = result
        all_sweep_rows.extend(sweep)
        comparison = dict(overall)
        comparison["model_name"] = model_name
        comparison.update(hard)
        comparison_rows.append(comparison)
        confusion[model_name] = {
            key: overall.get(key) for key in ("tp", "fp", "tn", "fn", "threshold")
        }
        calibration[model_name] = calibration_metrics(labels, scores)
        per_scene_rows.extend(
            grouped_metrics(labels, scores, metadata, "scene_name", strict_threshold, model_name)
        )
        per_camera_rows.extend(
            grouped_metrics(labels, scores, metadata, "camera_pair", strict_threshold, model_name)
        )
        metrics_filename = (
            "reid_only_metrics.json"
            if model_name == "reid_only_baseline"
            else model_name + "_metrics.json"
        )
        write_json(output_root / "evaluation" / metrics_filename, result)

    selection = select_model_conservatively(evaluations, config.get("selection", {}))
    selected_name = selection.get("selected_model")
    artifact_names = {
        "reid_only_baseline": "reid_only_baseline.json",
        "logistic_regression_l2": "logistic_regression_l2.pkl",
        "gradient_boosting": "gradient_boosting.pkl",
        "random_forest": "random_forest.pkl",
        "mlp_pairwise_scorer": "mlp_pairwise_scorer_best.pth",
    }
    if selected_name in artifact_names:
        selection["model_artifact"] = str(output_root / "models" / artifact_names[str(selected_name)])
    verdict = scorer_verdict(evaluations, selection, config)
    if selected_name in score_map:
        false_positives, false_negatives = top_error_rows(
            labels, score_map[str(selected_name)], metadata
        )
        write_csv_rows(output_root / "evaluation" / "top_false_positives.csv", false_positives)
        write_csv_rows(output_root / "evaluation" / "top_false_negatives.csv", false_negatives)
    write_csv_rows(output_root / "evaluation" / "model_comparison.csv", comparison_rows)
    write_csv_rows(output_root / "evaluation" / "threshold_sweep_all_models.csv", all_sweep_rows)
    write_csv_rows(output_root / "evaluation" / "per_scene_metrics.csv", per_scene_rows)
    write_csv_rows(output_root / "evaluation" / "per_camera_pair_metrics.csv", per_camera_rows)
    write_json(output_root / "evaluation" / "hard_negative_metrics.json", {
        name: result["hard_negative_metrics"] for name, result in evaluations.items()
    })
    write_csv_rows(
        output_root / "evaluation" / "hard_negative_metrics.csv",
        [
            dict({"model_name": name}, **result["hard_negative_metrics"])
            for name, result in evaluations.items()
        ],
    )
    write_json(output_root / "evaluation" / "calibration_metrics.json", calibration)
    write_json(output_root / "evaluation" / "confusion_matrices.json", confusion)
    write_json(output_root / "evaluation" / "selected_thresholds.json", {
        name: result["selected_thresholds"] for name, result in evaluations.items()
    })
    write_json(output_root / "models" / "selected_model.json", selection)
    write_json(output_root / "evaluation" / "scorer_verdict.json", verdict)
    return {
        "model_evaluations": evaluations,
        "selection": selection,
        "verdict": verdict,
        "score_map": score_map,
        "labels": labels,
        "metadata": metadata,
    }


def configured_thresholds(
    evaluation_config: Dict[str, Any], reid: bool = False
) -> List[float]:
    """Merge report thresholds with a dense grid used for selection."""
    fixed = [float(value) for value in evaluation_config.get("thresholds", [])]
    if reid:
        minimum = float(evaluation_config.get("reid_threshold_search_min", 0.60))
        maximum = float(evaluation_config.get("reid_threshold_search_max", 0.90))
    else:
        minimum = float(evaluation_config.get("threshold_search_min", 0.50))
        maximum = float(evaluation_config.get("threshold_search_max", 0.95))
    step = float(evaluation_config.get("threshold_search_step", 0.01))
    if step <= 0.0:
        raise ValueError("threshold_search_step must be positive")
    dense = []
    value = minimum
    while value <= maximum + 1e-9:
        dense.append(round(value, 6))
        value += step
    return sorted(set(fixed + dense))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
