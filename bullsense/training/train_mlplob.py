import sys
import os
import json
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Mapping, Optional
import torch
import torch.nn as nn
import numpy as np

# PYTHONPATH / ROOT
ROOT_DIR = Path(__file__).resolve().parents[1]  # bullsense/
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# PROJECT IMPORTS
from bullsense.config.base_config import ExperimentConfig
from bullsense.config.config_loader import load_config
from bullsense.data.dataset import create_dataloaders
from bullsense.data.metadata import DatasetMetadata
from bullsense.training.optim import build_optimizer
from bullsense.training.trainer import Trainer
from bullsense.tracking import MLflowLogger
from bullsense.utils.metrics import (
    compute_metrics,
    compute_regression_metrics,
    print_metrics,
    print_regression_metrics,
)
from bullsense.utils.visualization import plot_training_history, plot_confusion_matrix
from bullsense.model.registry import build_model as build_registered_model


# Helpers
def setup_device() -> torch.device:
    """Pick CUDA if available; print short info."""
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {d}")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    return d

def build_model(
    input_dim: int,
    seq_len: int,
    num_classes: int,
    cfg,
) -> nn.Module:
    """Build the configured model from a sweep/config namespace."""
    return build_registered_model(
        model_type=str(getattr(cfg, "model_type", "tlob")),
        input_dim=input_dim,
        seq_len=seq_len,
        num_classes=num_classes,
        cfg=cfg,
    )

def _build_trainer_config(experiment: ExperimentConfig, device: torch.device) -> SimpleNamespace:
    """Translate ExperimentConfig.training into Trainer-friendly attributes."""
    training = experiment.training
    model_cfg = experiment.model
    cfg = SimpleNamespace(
        BATCH_SIZE=training.batch_size,
        LEARNING_RATE=training.learning_rate,
        DROPOUT=model_cfg.dropout,
        WEIGHT_DECAY=training.weight_decay,
        NUM_CLASSES=model_cfg.num_classes,
        EPOCHS=training.epochs,
        EARLY_STOPPING_PATIENCE=training.early_stopping_patience,
        GRAD_CLIP=training.grad_clip,
        USE_FOCAL_LOSS=training.use_focal_loss,
        FOCAL_GAMMA=training.focal_gamma,
        FOCAL_ALPHA=training.focal_alpha,
        USE_AUTO_FOCAL_ALPHA=training.use_auto_focal_alpha,
        USE_CLASS_WEIGHTS=training.use_class_weights,
        LABEL_SMOOTHING=training.label_smoothing,
        MAX_GRAD_NORM=training.max_grad_norm,
        DEVICE=str(device),
    )
    return cfg


def _class_names_for(num_classes: int) -> list[str]:
    if num_classes == 5:
        return ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]
    if num_classes == 3:
        return ["STABLE", "UP", "DOWN"]
    return [f"CLASS_{i}" for i in range(num_classes)]


def _tagify(name: str) -> str:
    return name.lower().replace(" ", "_")


def _run_config_snapshot(
    exp_cfg: ExperimentConfig,
    *,
    model_type: str,
    task: str,
    target_shift_norm: str,
) -> dict[str, Any]:
    """Return the config payload persisted with checkpoints and reports."""
    model_cfg = exp_cfg.model
    training_cfg = exp_cfg.training
    label_cfg = exp_cfg.label
    return {
        "model_type": model_type,
        "task": task,
        "dropout": model_cfg.dropout,
        "lr": training_cfg.learning_rate,
        "weight_decay": training_cfg.weight_decay,
        "batch_size": training_cfg.batch_size,
        "epochs": training_cfg.epochs,
        "use_focal": training_cfg.use_focal_loss,
        "focal_gamma": training_cfg.focal_gamma,
        "focal_alpha": training_cfg.focal_alpha,
        "use_auto_focal_alpha": training_cfg.use_auto_focal_alpha,
        "use_class_weights": training_cfg.use_class_weights,
        "d_model": model_cfg.d_model,
        "t_layers": model_cfg.t_layers,
        "n_heads": model_cfg.n_heads,
        "hidden_dim": model_cfg.hidden_dim,
        "num_layers": model_cfg.num_layers,
        "is_sin_emb": model_cfg.is_sin_emb,
        "order_type_idx": model_cfg.order_type_idx,
        "use_bilinear_norm": model_cfg.use_bilinear_norm,
        "bilinear_norm_type": model_cfg.bilinear_norm_type,
        "temporal_readout": model_cfg.temporal_readout,
        "input_shift_norm": model_cfg.input_shift_norm,
        "dish_init": model_cfg.dish_init,
        "dish_activate": model_cfg.dish_activate,
        "use_target_shift_for_price": model_cfg.use_target_shift_for_price,
        "target_shift_norm": target_shift_norm,
        "revin_affine": model_cfg.revin_affine,
        "revin_detach_stats": model_cfg.revin_detach_stats,
        "regression_loss": training_cfg.regression_loss,
        "reg_target": label_cfg.reg_target,
        "reg_label_basis": label_cfg.reg_label_basis,
        "reg_normalization": label_cfg.reg_normalization,
        "reg_vol_window": label_cfg.reg_vol_window,
        "reg_vol_source": label_cfg.reg_vol_source,
        "reg_horizon_ticks": label_cfg.reg_horizon_ticks,
        "reg_clip_value": label_cfg.reg_clip_value,
        "lob_normalizer_style": training_cfg.lob_normalizer_style,
        "huber_delta": training_cfg.huber_delta,
        "weighted_huber_alpha": training_cfg.weighted_huber_alpha,
        "weighted_huber_power": training_cfg.weighted_huber_power,
        "weighted_huber_scale": training_cfg.weighted_huber_scale,
    }


def train_once(
    data_dir: Optional[str] = None,
    # Parameters made Optional to allow config passthrough
    dropout: Optional[float] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    use_focal: Optional[bool] = None,
    focal_gamma: Optional[float] = None,
    focal_alpha: Optional[list] = None,
    use_class_weights: Optional[bool] = None,
    run_name: Optional[str] = None,
    model_type: Optional[str] = None,
    d_model: Optional[int] = None,
    t_layers: Optional[int] = None,
    n_heads: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    input_shift_norm: Optional[str] = None,
    
    # --- Config controls ---
    experiment: Optional[ExperimentConfig] = None,
    config_path: Optional[Path] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    Runs a single training job (train/val/test) and returns metrics.
    Arguments defaulting to None will use the value from the loaded configuration.
    """
    device = setup_device()

    if experiment is not None:
        exp_cfg = experiment.model_copy(deep=True)
    else:
        exp_cfg = load_config(path=config_path, overrides=config_overrides)

    paths_cfg = exp_cfg.paths
    if data_dir is not None:
        paths_cfg.data_dir = Path(data_dir)
    data_dir_path = Path(paths_cfg.data_dir)

    training_cfg = exp_cfg.training
    model_cfg = exp_cfg.model
    task = str(getattr(exp_cfg, "task", "classification")).lower()

    # Update training hyperparameters ONLY if arguments are provided
    if batch_size is not None:
        training_cfg.batch_size = batch_size
    if lr is not None:
        training_cfg.learning_rate = lr
    if weight_decay is not None:
        training_cfg.weight_decay = weight_decay
    if epochs is not None:
        training_cfg.epochs = epochs
    
    # Loss function overrides
    if use_focal is not None:
        training_cfg.use_focal_loss = use_focal
    if focal_gamma is not None:
        training_cfg.focal_gamma = focal_gamma
    if focal_alpha is not None:
        training_cfg.focal_alpha = focal_alpha
    if use_class_weights is not None:
        training_cfg.use_class_weights = use_class_weights

    # Re-evaluate derived logic for focal loss vs class weights
    # (Prioritize Focal Loss settings if enabled)
    if training_cfg.use_focal_loss:
        # If focal enabled, disable standard class weights to avoid double weighting
        training_cfg.use_class_weights = False
        # Auto alpha logic
        if training_cfg.focal_alpha is None:
            training_cfg.use_auto_focal_alpha = True
        else:
            training_cfg.use_auto_focal_alpha = False
    else:
        # If focal disabled, ensure alpha is None so Trainer doesn't get confused
        training_cfg.focal_alpha = None
        
    # Model overrides
    if model_type is not None:
        model_cfg.model_type = model_type
    if dropout is not None:
        model_cfg.dropout = dropout
    if d_model is not None:
        model_cfg.d_model = d_model
    if t_layers is not None:
        model_cfg.t_layers = t_layers
    if n_heads is not None:
        model_cfg.n_heads = n_heads
    if hidden_dim is not None:
        model_cfg.hidden_dim = hidden_dim
    if num_layers is not None:
        model_cfg.num_layers = num_layers
    if input_shift_norm is not None:
        model_cfg.input_shift_norm = input_shift_norm

    # Sync locals with possibly updated config for usage below
    dropout = model_cfg.dropout
    d_model = model_cfg.d_model
    t_layers = model_cfg.t_layers
    n_heads = model_cfg.n_heads
    hidden_dim = model_cfg.hidden_dim
    num_layers = model_cfg.num_layers
    model_type = str(model_cfg.model_type).lower()
    output_dim = 1 if task == "regression" else int(model_cfg.num_classes)

    # Use values from CONFIG (which might have been updated above)
    batch_size = training_cfg.batch_size
    lr = training_cfg.learning_rate
    weight_decay = training_cfg.weight_decay
    epochs = training_cfg.epochs
    use_focal = training_cfg.use_focal_loss
    focal_gamma = training_cfg.focal_gamma
    focal_alpha = training_cfg.focal_alpha
    use_class_weights = training_cfg.use_class_weights

    # ----------------- Dataloaders -----------------
    print("-> Loading dataloaders (Train, Val, Test)...")
    num_workers = int(os.environ.get("NUM_WORKERS", "0"))
    # create_dataloaders returns (train, val, test)
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=str(data_dir_path),
        use_normalizer=training_cfg.use_lob_normalizer,
        task=task,
        normalizer_style=training_cfg.lob_normalizer_style,
    )

    # ----------------- Metadata -----------------
    meta_path = data_dir_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at: {meta_path}")
    metadata = DatasetMetadata.load_json(meta_path)
    seq_len = metadata.seq_len
    feat_dim = metadata.feature_dim

    def _smoothed_inverse_class_weights(y_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        y_train = np.load(y_path, mmap_mode="r")
        y_train_np = np.asarray(y_train)
        if not np.issubdtype(y_train_np.dtype, np.integer):
            y_train_np = y_train_np.astype(np.int64)

        class_counts = torch.from_numpy(
            np.bincount(
                y_train_np,
                minlength=int(model_cfg.num_classes),
            )
        )
        smoothing_factor = 0.5
        weights = 1.0 / (class_counts.float() ** smoothing_factor)
        weights[torch.isinf(weights)] = 0.0
        weights = weights / weights.sum() * model_cfg.num_classes
        return class_counts, weights

    # Class Weights / Focal Alpha Calculation
    class_weights_tensor: Optional[torch.Tensor] = None
    if task != "regression" and use_class_weights:
        print("-> Computing class weights from training labels...")
        class_counts, class_weights_tensor = _smoothed_inverse_class_weights(
            data_dir_path / "y_train.npy"
        )
        print(f"   Class counts: {class_counts.tolist()}")
        print(f"   Class weights: {[round(float(x), 6) for x in class_weights_tensor]}")

    elif task != "regression" and use_focal:
        if training_cfg.focal_alpha is not None:
            print(f"-> Using manual focal_alpha from config {training_cfg.focal_alpha}")
            class_weights_tensor = torch.tensor(training_cfg.focal_alpha).float()
        elif training_cfg.use_auto_focal_alpha:
            print("-> Computing auto focal_alpha from training labels...")
            class_counts, class_weights_tensor = _smoothed_inverse_class_weights(
                data_dir_path / "y_train.npy"
            )
            computed_alpha = [float(x) for x in class_weights_tensor.tolist()]
            training_cfg.focal_alpha = computed_alpha
            focal_alpha = computed_alpha
            print(f"   Class counts: {class_counts.tolist()}")
            print(f"   Focal alpha: {[round(x, 6) for x in computed_alpha]}")

    # ----------------- Model Build -----------------
    model_params = SimpleNamespace(
        model_type=model_type,
        dropout=dropout,
        d_model=d_model,
        t_layers=t_layers,
        n_heads=n_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        is_sin_emb=model_cfg.is_sin_emb,
        order_type_idx=model_cfg.order_type_idx,
        use_bilinear_norm=model_cfg.use_bilinear_norm,
        bilinear_norm_type=model_cfg.bilinear_norm_type,
        temporal_readout=model_cfg.temporal_readout,
        attn_dropout=model_cfg.attn_dropout,
        temporal_causal=model_cfg.temporal_causal,
        gradient_checkpointing=model_cfg.gradient_checkpointing,
        input_shift_norm=model_cfg.input_shift_norm,
        dish_init=model_cfg.dish_init,
        dish_activate=model_cfg.dish_activate,
        revin_affine=model_cfg.revin_affine,
        revin_detach_stats=model_cfg.revin_detach_stats,
        use_target_shift_for_price=model_cfg.use_target_shift_for_price,
        dain_mode=model_cfg.dain_mode,
        dain_mean_lr=model_cfg.dain_mean_lr,
        dain_gate_lr=model_cfg.dain_gate_lr,
        dain_scale_lr=model_cfg.dain_scale_lr,
        grouped_dain_groups=model_cfg.grouped_dain_groups,
    )

    model = build_model(
        input_dim=int(feat_dim),
        seq_len=int(seq_len),
        num_classes=output_dim,
        cfg=model_params,
    ).to(device)

    target_shift_norm = "none"
    if (
        task == "regression"
        and exp_cfg.label.reg_label_basis == "price"
        and exp_cfg.label.reg_normalization == "none"
        and model_cfg.use_target_shift_for_price
        and model_cfg.input_shift_norm in {"dish", "revin"}
    ):
        target_shift_norm = model_cfg.input_shift_norm

    # Model bilgisi
    if hasattr(model, "info"):
        info = model.info()
        print(f"\nModel: {info.get('name','unknown')}")
        print(f"  Parameters: {info.get('total_params','?'):,}")
        print(f"  type: {model_type}")
    else:
        print(f"\nModel: {model.__class__.__name__}")

    # ----------------- Run metadata -----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = model_type
    loss_tag = "focal" if use_focal else "ce"
    output_dir = Path(exp_cfg.paths.output_dir)
    run_dir = output_dir / f"{base_dir}_{loss_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if run_name is not None:
        tag = run_name
    else:
        tag = f"{timestamp}_D{d_model}_L{t_layers}_H{n_heads}"

    tracking_cfg = getattr(exp_cfg, "tracking", None)
    mlflow_run_name = (
        run_name
        or (getattr(tracking_cfg, "run_name", None) if tracking_cfg is not None else None)
        or tag
    )
    tracking_tags = {
        "experiment": exp_cfg.name,
        "model_type": model_type,
        "task": task,
        "loss": loss_tag,
        "timestamp": timestamp,
        "device": str(device),
        "output_tag": tag,
    }
    if exp_cfg.data.symbol:
        tracking_tags["data.symbol"] = exp_cfg.data.symbol
    if exp_cfg.data.obid is not None:
        tracking_tags["data.obid"] = str(exp_cfg.data.obid)

    trainer_cfg = _build_trainer_config(exp_cfg, device)

    results: dict[str, Any] | None = None

    with MLflowLogger.from_config(
        tracking_cfg,
        run_name=mlflow_run_name,
        tags=tracking_tags,
    ) as mlflow_logger:
        if mlflow_logger.active:
            mlflow_logger.log_params(
                {
                    "model": {
                        "type": model_type,
                        "task": task,
                        "dropout": dropout,
                        "d_model": d_model,
                        "t_layers": t_layers,
                        "n_heads": n_heads,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "is_sin_emb": model_cfg.is_sin_emb,
                        "order_type_idx": model_cfg.order_type_idx,
                        "use_bilinear_norm": model_cfg.use_bilinear_norm,
                        "bilinear_norm_type": model_cfg.bilinear_norm_type,
                        "temporal_readout": model_cfg.temporal_readout,
                        "input_shift_norm": model_cfg.input_shift_norm,
                        "dish_init": model_cfg.dish_init,
                        "dish_activate": model_cfg.dish_activate,
                        "use_target_shift_for_price": model_cfg.use_target_shift_for_price,
                        "target_shift_norm": target_shift_norm,
                        "revin_affine": model_cfg.revin_affine,
                        "revin_detach_stats": model_cfg.revin_detach_stats,
                        "dain_mode": model_cfg.dain_mode,
                        "dain_mean_lr": model_cfg.dain_mean_lr,
                        "dain_gate_lr": model_cfg.dain_gate_lr,
                        "dain_scale_lr": model_cfg.dain_scale_lr,
                        "grouped_dain_groups": model_cfg.grouped_dain_groups,
                    },
                    "training": {
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "epochs": epochs,
                        "use_focal": use_focal,
                        "focal_gamma": focal_gamma,
                        "focal_alpha": focal_alpha,
                        "use_class_weights": use_class_weights,
                        "regression_loss": training_cfg.regression_loss,
                        "lob_normalizer_style": training_cfg.lob_normalizer_style,
                        "huber_delta": training_cfg.huber_delta,
                        "weighted_huber_alpha": training_cfg.weighted_huber_alpha,
                        "weighted_huber_power": training_cfg.weighted_huber_power,
                        "weighted_huber_scale": training_cfg.weighted_huber_scale,
                    },
                    "data": {
                        "data_dir": str(data_dir_path),
                        "seq_len": int(seq_len),
                        "feature_dim": int(feat_dim),
                    },
                }
            )

        # Optimizer + (opsiyonel) scheduler
        # build_optimizer gives DAIN sublayers their own LR groups (wd=0);
        # for every other model it is a plain single-group AdamW.
        optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
        scheduler = None

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=epochs,
            grad_clip=training_cfg.grad_clip,
            use_amp=True,
            is_focal=use_focal,
            focal_gamma=focal_gamma,
            class_weights=class_weights_tensor,
            task=task,
            regression_loss=training_cfg.regression_loss,
            huber_delta=training_cfg.huber_delta,
            weighted_huber_alpha=training_cfg.weighted_huber_alpha,
            weighted_huber_power=training_cfg.weighted_huber_power,
            weighted_huber_scale=training_cfg.weighted_huber_scale,
            target_shift_norm=target_shift_norm,
            seq_len=int(seq_len),
            reg_projection_target=exp_cfg.label.reg_target,
            lob_normalizer=getattr(train_loader.dataset, "normalizer", None),
            dish_init=model_cfg.dish_init,
            dish_activate=model_cfg.dish_activate,
            revin_affine=model_cfg.revin_affine,
            revin_detach_stats=model_cfg.revin_detach_stats,
            es_patience=training_cfg.early_stopping_patience,
            es_min_delta=1e-4,
        )
        history = trainer.fit(train_loader, val_loader)

        # ----------------- Evaluate -----------------
        print(f"\n{'='*60}")
        print(" EVALUATION ON TEST SET")
        print(f"{'='*60}")
        test_out = trainer.evaluate(test_loader)
        if task == "regression":
            class_names = []
            class_tags = []
            metrics = compute_regression_metrics(test_out["labels"], test_out["predictions"])
            print_regression_metrics(metrics)
        else:
            class_names = _class_names_for(int(model_cfg.num_classes))
            class_tags = [_tagify(n) for n in class_names]
            labels_eval = list(range(len(class_names)))
            metrics = compute_metrics(
                test_out["labels"],
                test_out["predictions"],
                labels=labels_eval,
            )
            metrics["class_names"] = class_names
            print_metrics(metrics, class_names=class_names)

        if mlflow_logger.active:
            step = len(history.get("train_loss", []))
            if task == "regression":
                metric_payload = {
                    "metrics/test_mae": metrics["mae"],
                    "metrics/test_rmse": metrics["rmse"],
                    "metrics/test_directional_accuracy": metrics["directional_accuracy"],
                    "metrics/test_correlation": metrics["correlation"],
                    "training/best_val_loss": float(getattr(trainer, "best_val_loss", np.nan)),
                }
            else:
                metric_payload = {
                    "metrics/test_accuracy": metrics["accuracy"],
                    "metrics/test_balanced_accuracy": metrics.get("balanced_accuracy"),
                    "metrics/test_f1_weighted": metrics["f1_weighted"],
                    "metrics/test_f1_macro": metrics["f1_macro"],
                    "training/best_val_accuracy": float(getattr(trainer, "best_val_acc", np.nan)),
                    "training/best_val_loss": float(getattr(trainer, "best_val_loss", np.nan)),
                }
            mlflow_logger.log_metrics(metric_payload, step=step)
            if task != "regression":
                for idx, class_tag in enumerate(class_tags):
                    mlflow_logger.log_metric(f"metrics/f1_{class_tag}", metrics["f1_per_class"][idx], step=step)

        # ----------------- Saving / Plots -----------------
        try:
            plot_training_history(history, save_path=run_dir / f"curves_{tag}.png")
        except Exception as e:
            print(f"[warn] plot_training_history failed: {e}")

        if task != "regression":
            try:
                plot_confusion_matrix(
                    metrics["confusion_matrix"],
                    target_names=class_names,
                    save_path=run_dir / f"cm_{tag}.png",
                )
            except Exception as e:
                print(f"[warn] plot_confusion_matrix failed: {e}")

        run_config = _run_config_snapshot(
            exp_cfg,
            model_type=model_type,
            task=task,
            target_shift_norm=target_shift_norm,
        )
        save_obj = {
            "state_dict": model.state_dict(),
            "target_shift_norm_state_dict": (
                trainer.target_shift_norm.state_dict()
                if getattr(trainer, "target_shift_norm", None) is not None
                else None
            ),
            "metadata": metadata.to_dict(),
            "config": run_config,
        }
        model_path = run_dir / f"model_{tag}.pt"
        torch.save(save_obj, model_path)

        history_file = run_dir / f"history_{tag}.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        results = {
            "timestamp": timestamp,
            "run_name": tag,
            "data_dir": str(data_dir_path),
            "output_dir": str(run_dir),
            "model_type": model_type,
            "task": task,
            "config": run_config,
            "metrics": (
                {
                    "test_mae": float(metrics["mae"]),
                    "test_rmse": float(metrics["rmse"]),
                    "test_mse": float(metrics["mse"]),
                    "test_directional_accuracy": float(metrics["directional_accuracy"]),
                    "test_correlation": float(metrics["correlation"]),
                }
                if task == "regression"
                else {
                    "test_accuracy": float(metrics["accuracy"]),
                    "test_balanced_accuracy": float(metrics.get("balanced_accuracy", np.nan)),
                    "test_f1_weighted": float(metrics["f1_weighted"]),
                    "test_f1_macro": float(metrics["f1_macro"]),
                    "per_class_f1": {
                        class_tags[i]: float(metrics["f1_per_class"][i])
                        for i in range(len(class_tags))
                    },
                    "per_class_precision": {
                        class_tags[i]: float(metrics["precision_per_class"][i])
                        for i in range(len(class_tags))
                    },
                }
            ),
            "training": {
                "epochs_trained": len(history.get("train_loss", [])),
                "best_val_acc": float(getattr(trainer, "best_val_acc", np.nan)),
                "best_val_loss": float(getattr(trainer, "best_val_loss", np.nan)),
                "final_train_loss": float(
                    history.get("train_loss", [np.nan])[-1] if history.get("train_loss") else np.nan
                ),
                "final_val_loss": float(
                    history.get("val_loss", [np.nan])[-1] if history.get("val_loss") else np.nan
                ),
            },
        }
        results_file = run_dir / f"results_{tag}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if mlflow_logger.active:
            mlflow_logger.log_artifact(results_file, artifact_path="reports")
            mlflow_logger.log_artifact(history_file, artifact_path="reports")
            mlflow_logger.log_artifact(model_path, artifact_path="models")
            curve_path = run_dir / f"curves_{tag}.png"
            cm_path = run_dir / f"cm_{tag}.png"
            if curve_path.exists():
                mlflow_logger.log_artifact(curve_path, artifact_path="plots")
            if cm_path.exists():
                mlflow_logger.log_artifact(cm_path, artifact_path="plots")

    print(f"\nSaved:")
    print(f"  Model:   {run_dir / f'model_{tag}.pt'}")
    print(f"  Results: {run_dir / f'results_{tag}.json'}")
    print(f"  Plots:   {run_dir / f'curves_{tag}.png'}, {run_dir / f'cm_{tag}.png'}")

    return results if results is not None else {}
