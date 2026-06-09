"""Training and evaluation helpers for OSNet Person ReID fine-tuning."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deep_oc_sort_3d.reid_training.balanced_identity_sampler import BalancedIdentitySampler
from deep_oc_sort_3d.reid_training.osnet_finetune_config import (
    load_osnet_finetune_config,
    output_root_from_config,
    prepare_output_dirs,
    save_resolved_config,
    summarize_environment,
)
from deep_oc_sort_3d.reid_training.osnet_model_factory import (
    build_osnet_model,
    forward_with_features,
    model_state_payload,
    resolve_device,
    set_backbone_trainable,
)
from deep_oc_sort_3d.reid_training.person_reid_torch_dataset import SmartSpacesPersonReIDTorchDataset
from deep_oc_sort_3d.reid_training.reid_dataset_io import write_json
from deep_oc_sort_3d.reid_training.reid_embedding_extractor import extract_embeddings_for_csv, load_embedding_matrix
from deep_oc_sort_3d.reid_training.reid_losses import total_reid_loss
from deep_oc_sort_3d.reid_training.reid_retrieval_eval import evaluate_retrieval
from deep_oc_sort_3d.reid_training.reid_similarity_diagnostics import (
    compute_similarity_diagnostics,
    finetuning_verdict,
    metric_deltas,
)


LOG_FIELDS = [
    "epoch",
    "train_loss",
    "ce_loss",
    "triplet_loss",
    "val_top1_accuracy",
    "val_top5_accuracy",
    "val_mAP",
    "learning_rate",
]


def train_osnet_person_reid(
    config_path: Path,
    overrides: Optional[Dict[str, Any]] = None,
    progress: bool = True,
    overwrite: bool = False,
    resume: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fine-tune OSNet on the prepared SmartSpaces Person ReID crops."""
    import torch
    config = load_osnet_finetune_config(Path(config_path), overrides)
    output_root = prepare_output_dirs(config, overwrite=overwrite)
    save_resolved_config(config, output_root)
    environment = summarize_environment(config, output_root)
    train_dataset, val_dataset = create_train_val_datasets(config)
    if len(train_dataset) <= 0:
        raise ValueError("Training ReID dataset is empty. Check Step 18A metadata paths.")
    if len(val_dataset) <= 0:
        raise ValueError("Validation ReID dataset is empty. Check Step 18A metadata paths.")

    device = resolve_device(config)
    num_classes = len(train_dataset.identity_to_label)
    model = build_osnet_model(config, num_classes=num_classes, device=device)
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    loader = create_train_loader(train_dataset, config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = _create_amp_scaler(config, device)
    start_epoch = 1
    if resume is not None:
        start_epoch = load_training_checkpoint(model, optimizer, Path(resume), device) + 1

    log_csv = output_root / "logs" / "training_log.csv"
    if not log_csv.exists() or overwrite:
        _write_log_header(log_csv)

    best_loss = None
    best_top1 = None
    epochs = int(training_cfg.get("epochs", 20))
    freeze_epochs = int(training_cfg.get("freeze_backbone_epochs", 2))
    eval_every = int(training_cfg.get("eval_every_epochs", 1))
    summaries: List[Dict[str, Any]] = []
    for epoch in range(start_epoch, epochs + 1):
        set_backbone_trainable(model, trainable=epoch > freeze_epochs)
        train_metrics = train_one_epoch(model, loader, optimizer, scaler, config, device, progress=progress, epoch=epoch)
        if scheduler is not None:
            scheduler.step()
        val_metrics: Dict[str, Any] = {}
        if eval_every > 0 and epoch % eval_every == 0:
            val_metrics = evaluate_model_retrieval(
                model,
                config,
                output_root,
                device,
                prefix="epoch_%03d_val" % epoch,
                max_crops=int(data_cfg.get("max_val_crops", 20000)),
                progress=progress,
            )
        row = _log_row(epoch, train_metrics, val_metrics, optimizer)
        _append_log_row(log_csv, row)
        summaries.append(row)
        current_loss = float(train_metrics.get("total_loss", 0.0))
        checkpoint_metrics = dict(row)
        checkpoint_metrics["num_classes"] = int(num_classes)
        checkpoint_metrics["environment"] = environment
        save_checkpoint(model, optimizer, epoch, checkpoint_metrics, config, output_root / "checkpoints" / ("epoch_%03d.pth" % epoch))
        save_checkpoint(model, optimizer, epoch, checkpoint_metrics, config, output_root / "checkpoints" / "last.pth")
        if best_loss is None or current_loss < float(best_loss):
            best_loss = current_loss
            save_checkpoint(model, optimizer, epoch, checkpoint_metrics, config, output_root / "checkpoints" / "best_val_loss.pth")
        top1 = val_metrics.get("top1_accuracy")
        if top1 is not None and (best_top1 is None or float(top1) > float(best_top1)):
            best_top1 = float(top1)
            save_checkpoint(model, optimizer, epoch, checkpoint_metrics, config, output_root / "checkpoints" / "best_retrieval_top1.pth")

    final_checkpoint = output_root / "checkpoints" / "best_retrieval_top1.pth"
    if not final_checkpoint.exists():
        final_checkpoint = output_root / "checkpoints" / "last.pth"
    final_eval = evaluate_pretrained_vs_finetuned(
        config,
        checkpoint_path=final_checkpoint,
        output_root=output_root,
        progress=progress,
    )
    summary = {
        "status": "ok",
        "output_root": str(output_root),
        "num_train_crops": len(train_dataset),
        "num_val_crops": len(val_dataset),
        "num_train_identities": int(num_classes),
        "epochs": epochs,
        "best_train_loss_proxy": best_loss,
        "best_val_top1": best_top1,
        "final_evaluation": final_eval,
    }
    write_json(summary, output_root / "reports" / "training_summary.json")
    return summary


def create_train_val_datasets(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Create train and validation crop datasets."""
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_csv = Path(str(data_cfg.get("train_metadata_csv", "")))
    val_csv = Path(str(data_cfg.get("val_metadata_csv", "")))
    input_size = (int(model_cfg.get("input_height", 256)), int(model_cfg.get("input_width", 128)))
    train_dataset = SmartSpacesPersonReIDTorchDataset(
        csv_path=train_csv,
        min_crops_per_identity=int(data_cfg.get("min_crops_per_identity", 5)),
        max_crops_per_identity=_optional_int(data_cfg.get("max_train_crops_per_identity")),
        input_size=input_size,
        training=True,
        normalize=True,
    )
    val_dataset = SmartSpacesPersonReIDTorchDataset(
        csv_path=val_csv,
        min_crops_per_identity=int(data_cfg.get("min_crops_per_identity", 5)),
        max_crops_per_identity=_optional_int(data_cfg.get("max_val_crops_per_identity")),
        input_size=input_size,
        training=False,
        normalize=True,
    )
    return train_dataset, val_dataset


def create_train_loader(dataset: Any, config: Dict[str, Any]) -> Any:
    """Create a balanced identity training loader."""
    from torch.utils.data import DataLoader

    data_cfg = config.get("data", {})
    sampler_cfg = config.get("sampler", {})
    batch_size = int(data_cfg.get("batch_size", 64))
    labels = [int(row.get("label", -1)) for row in dataset.rows]
    sampler = None
    shuffle = True
    if bool(sampler_cfg.get("use_balanced_identity_sampler", True)):
        sampler = BalancedIdentitySampler(
            labels,
            identities_per_batch=int(sampler_cfg.get("identities_per_batch", 16)),
            images_per_identity=int(sampler_cfg.get("images_per_identity", 4)),
            seed=int(sampler_cfg.get("seed", 42)),
        )
        shuffle = False
        expected_batch = int(sampler_cfg.get("identities_per_batch", 16)) * int(sampler_cfg.get("images_per_identity", 4))
        batch_size = int(data_cfg.get("batch_size", expected_batch))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 4)),
        drop_last=True,
    )


def create_optimizer(model: Any, config: Dict[str, Any]) -> Any:
    """Create optimizer for trainable parameters."""
    import torch

    training_cfg = config.get("training", {})
    params = [param for param in model.parameters() if param.requires_grad]
    lr = float(training_cfg.get("learning_rate", 0.0003))
    weight_decay = float(training_cfg.get("weight_decay", 0.0005))
    name = str(training_cfg.get("optimizer", "adamw")).lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer: Any, config: Dict[str, Any]) -> Any:
    """Create optional LR scheduler."""
    import torch

    training_cfg = config.get("training", {})
    name = str(training_cfg.get("scheduler", "cosine")).lower()
    if name == "none":
        return None
    epochs = int(training_cfg.get("epochs", 20))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(training_cfg.get("step_size", 10)), gamma=float(training_cfg.get("gamma", 0.1)))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))


def train_one_epoch(model: Any, loader: Any, optimizer: Any, scaler: Any, config: Dict[str, Any], device: str, progress: bool, epoch: int) -> Dict[str, Any]:
    """Run one training epoch."""
    import torch

    from deep_oc_sort_3d.reid_training.reid_dataset_io import progress_iter

    model.train()
    totals: Dict[str, float] = {"total_loss": 0.0, "ce_loss": 0.0, "triplet_loss": 0.0}
    batches = 0
    use_amp = scaler is not None
    iterator = progress_iter(loader, progress, "train ReID epoch %d" % int(epoch), "batch")
    for batch in iterator:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = forward_with_features(model, images)
                loss, components = total_reid_loss(outputs, labels, config)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = forward_with_features(model, images)
            loss, components = total_reid_loss(outputs, labels, config)
            loss.backward()
            optimizer.step()
        totals["total_loss"] += float(components.get("total_loss", 0.0))
        totals["ce_loss"] += float(components.get("ce_loss", 0.0))
        totals["triplet_loss"] += float(components.get("triplet_loss", 0.0))
        batches += 1
    if batches <= 0:
        raise ValueError("No training batches were produced. Check batch size and sampler settings.")
    return {key: value / float(batches) for key, value in totals.items()}


def evaluate_model_retrieval(model: Any, config: Dict[str, Any], output_root: Path, device: str, prefix: str, max_crops: Optional[int], progress: bool) -> Dict[str, Any]:
    """Extract validation embeddings and compute retrieval metrics."""
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    npy_path = output_root / "embeddings" / ("%s_embeddings.npy" % prefix)
    csv_path = output_root / "embeddings" / ("%s_metadata.csv" % prefix)
    extract_embeddings_for_csv(
        model,
        Path(str(data_cfg.get("val_metadata_csv", ""))),
        npy_path,
        csv_path,
        config,
        device,
        max_crops=max_crops,
        show_progress=progress,
    )
    embeddings, metadata = load_embedding_matrix(npy_path, csv_path)
    return evaluate_retrieval(
        embeddings,
        metadata,
        topk=[int(value) for value in eval_cfg.get("topk", [1, 5, 10])],
        query_chunk_size=int(eval_cfg.get("retrieval_chunk_size", 512)),
        max_map_queries=_optional_int(eval_cfg.get("max_map_queries", 5000)),
    )


def evaluate_pretrained_vs_finetuned(
    config: Dict[str, Any],
    checkpoint_path: Optional[Path],
    output_root: Optional[Path] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Evaluate local pretrained OSNet against a fine-tuned checkpoint."""
    import torch

    if output_root is None:
        output_root = prepare_output_dirs(config, overwrite=False)
    else:
        output_root = Path(output_root)
        for name in ["configs", "checkpoints", "logs", "embeddings", "evaluation", "figures", "reports"]:
            (output_root / name).mkdir(parents=True, exist_ok=True)
    device = resolve_device(config)
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    max_val_crops = _optional_int(data_cfg.get("max_val_crops", 20000))
    pretrained_model = build_osnet_model(config, num_classes=1, device=device)
    pretrained_summary = _extract_and_score_model(
        pretrained_model,
        config,
        output_root,
        device,
        prefix="pretrained_val",
        max_crops=max_val_crops,
        progress=progress,
    )
    num_classes = _checkpoint_num_classes(checkpoint_path) if checkpoint_path is not None else 1
    finetuned_model = build_osnet_model(config, num_classes=max(1, int(num_classes)), device=device)
    checkpoint_status: Dict[str, Any] = {"checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "", "loaded": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(str(checkpoint_path), map_location=device)
        payload = state.get("model_state_dict", state) if isinstance(state, dict) else state
        finetuned_model.load_state_dict(payload, strict=False)
        checkpoint_status["loaded"] = True
    finetuned_summary = _extract_and_score_model(
        finetuned_model,
        config,
        output_root,
        device,
        prefix="finetuned_val",
        max_crops=max_val_crops,
        progress=progress,
    )
    deltas = metric_deltas(
        pretrained_summary["retrieval"],
        finetuned_summary["retrieval"],
        pretrained_summary["similarity"],
        finetuned_summary["similarity"],
    )
    verdict = finetuning_verdict(pretrained_summary["retrieval"], finetuned_summary["retrieval"], deltas)
    summary = {
        "status": "ok",
        "checkpoint": checkpoint_status,
        "pretrained": pretrained_summary,
        "finetuned": finetuned_summary,
        "deltas": deltas,
        "verdict": verdict,
        "max_val_crops": max_val_crops,
        "topk": eval_cfg.get("topk", [1, 5, 10]),
    }
    write_json(summary, output_root / "evaluation" / "retrieval_metrics.json")
    write_json(summary, output_root / "reports" / "osnet_finetune_evaluation_summary.json")
    return summary


def save_checkpoint(model: Any, optimizer: Any, epoch: int, metrics: Dict[str, Any], config: Dict[str, Any], path: Path) -> None:
    """Save a checkpoint payload."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state_payload(model, optimizer, epoch, metrics, config), str(path))


def load_training_checkpoint(model: Any, optimizer: Any, checkpoint_path: Path, device: str) -> int:
    """Load a training checkpoint and return checkpoint epoch."""
    import torch

    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return int(state.get("epoch", 0))


def _extract_and_score_model(model: Any, config: Dict[str, Any], output_root: Path, device: str, prefix: str, max_crops: Optional[int], progress: bool) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    npy_path = output_root / "embeddings" / ("%s_embeddings.npy" % prefix)
    csv_path = output_root / "embeddings" / ("%s_metadata.csv" % prefix)
    extraction = extract_embeddings_for_csv(
        model,
        Path(str(data_cfg.get("val_metadata_csv", ""))),
        npy_path,
        csv_path,
        config,
        device,
        max_crops=max_crops,
        show_progress=progress,
    )
    embeddings, metadata = load_embedding_matrix(npy_path, csv_path)
    retrieval = evaluate_retrieval(
        embeddings,
        metadata,
        topk=[int(value) for value in eval_cfg.get("topk", [1, 5, 10])],
        query_chunk_size=int(eval_cfg.get("retrieval_chunk_size", 512)),
        max_map_queries=_optional_int(eval_cfg.get("max_map_queries", 5000)),
    )
    similarity = compute_similarity_diagnostics(
        embeddings,
        metadata,
        max_pairs=int(eval_cfg.get("max_similarity_pairs", 200000)),
        thresholds=[float(value) for value in eval_cfg.get("thresholds", [0.75, 0.80, 0.85])],
        seed=int(eval_cfg.get("seed", 42)),
    )
    return {"extraction": extraction, "retrieval": retrieval, "similarity": similarity}


def _create_amp_scaler(config: Dict[str, Any], device: str) -> Any:
    training_cfg = config.get("training", {})
    if not bool(training_cfg.get("use_amp", True)):
        return None
    if str(device) != "cuda":
        return None
    import torch

    return torch.cuda.amp.GradScaler()


def _write_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()


def _append_log_row(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


def _log_row(epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], optimizer: Any) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train_loss": train_metrics.get("total_loss"),
        "ce_loss": train_metrics.get("ce_loss"),
        "triplet_loss": train_metrics.get("triplet_loss"),
        "val_top1_accuracy": val_metrics.get("top1_accuracy"),
        "val_top5_accuracy": val_metrics.get("top5_accuracy"),
        "val_mAP": val_metrics.get("mAP"),
        "learning_rate": optimizer.param_groups[0].get("lr") if optimizer is not None and optimizer.param_groups else None,
    }


def _checkpoint_num_classes(checkpoint_path: Optional[Path]) -> int:
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return 1
    import torch

    state = torch.load(str(checkpoint_path), map_location="cpu")
    payload = state.get("model_state_dict", state) if isinstance(state, dict) else {}
    for key, value in payload.items():
        if str(key).endswith("classifier.weight"):
            return int(value.shape[0])
    metrics = state.get("metrics", {}) if isinstance(state, dict) else {}
    return int(metrics.get("num_classes", 1))


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)
