"""
Generic Trainer for MLP-LOB / TLOB
- Default: CrossEntropy + accuracy-based early stopping
- Optional: Focal Loss (precision-oriented) if is_focal=True
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from bullsense.model.layers.shift_norm import ScalarDishTS, ScalarRevIN


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    gamma > 0 -> easy örnekleri bastırır, zor örnekleri öne çıkarır.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], target: [B]
        ce_loss = F.cross_entropy(
            logits,
            target,
            reduction="none"
        )  # [B]

        pt = torch.exp(-ce_loss)  # p_t dogru sinifin probabilitysini hesapla
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # [B]
        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            focal_loss = focal_loss * alpha.gather(0, target)

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class WeightedHuberLoss(nn.Module):
    def __init__(
        self,
        delta: float = 1.0,
        alpha: float = 1.0,
        power: float = 1.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.power = float(power)
        self.scale = float(scale)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        abs_err = torch.abs(err)
        delta = torch.tensor(self.delta, device=pred.device, dtype=pred.dtype)
        huber = torch.where(
            abs_err <= delta,
            0.5 * err.pow(2),
            delta * (abs_err - 0.5 * delta),
        )
        scale = torch.tensor(max(self.scale, 1e-9), device=pred.device, dtype=pred.dtype)
        weights = 1.0 + self.alpha * (abs_err / scale).pow(self.power)
        return (huber * weights).mean()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        num_epochs: int = 50,
        grad_clip: Optional[float] = 1.0,
        use_amp: bool = True,
        
        is_focal: bool = False,           
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        task: str = "classification",
        regression_loss: str = "huber",
        huber_delta: float = 1.0,
        weighted_huber_alpha: float = 1.0,
        weighted_huber_power: float = 1.0,
        weighted_huber_scale: float = 1.0,
        target_shift_norm: str = "none",
        seq_len: int | None = None,
        reg_projection_target: str = "mid",
        lob_normalizer: Any | None = None,
        dish_init: str = "standard",
        dish_activate: bool = True,
        revin_affine: bool = True,
        revin_detach_stats: bool = True,
        # --- early stopping ---
        es_patience: int = 10,
        es_min_delta: float = 0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.task = task
        self.reg_projection_target = (reg_projection_target or "mid").lower()
        self.lob_normalizer = lob_normalizer

        if class_weights is not None:
            class_weights = class_weights.to(device)

        if task == "regression":
            loss_key = regression_loss.lower()
            if loss_key in {"huber", "smoothl1"}:
                self.criterion = nn.HuberLoss(delta=huber_delta)
            elif loss_key in {"mse", "l2"}:
                self.criterion = nn.MSELoss()
            elif loss_key in {"mae", "l1"}:
                self.criterion = nn.L1Loss()
            elif loss_key in {"weighted_huber", "whuber"}:
                self.criterion = WeightedHuberLoss(
                    delta=huber_delta,
                    alpha=weighted_huber_alpha,
                    power=weighted_huber_power,
                    scale=weighted_huber_scale,
                )
            else:
                raise ValueError(f"Unknown regression loss: {regression_loss}")
        elif is_focal:
            self.criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.target_shift_norm = None
        target_shift_key = (target_shift_norm or "none").lower()
        if task == "regression" and target_shift_key == "dish":
            if seq_len is None:
                raise ValueError("seq_len is required for target_shift_norm='dish'.")
            self.target_shift_norm = ScalarDishTS(
                seq_len=seq_len,
                dish_init=dish_init,
                activate=dish_activate,
            ).to(device)
        elif task == "regression" and target_shift_key == "revin":
            self.target_shift_norm = ScalarRevIN(
                affine=revin_affine,
                detach_stats=revin_detach_stats,
            ).to(device)
        elif target_shift_key not in {"none", "", "off", "false"}:
            raise ValueError(f"Unknown target_shift_norm: {target_shift_norm}")
        if self.target_shift_norm is not None:
            self.optimizer.add_param_group({"params": self.target_shift_norm.parameters()})

        # AMP scaler
        self.scaler = GradScaler(enabled=use_amp)

        # early stopping
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta

    @staticmethod
    def _compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
        preds = logits.argmax(dim=1)
        correct = (preds == target).sum().item()
        total = target.numel()
        return correct / total

    @staticmethod
    def _compute_directional_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_dir = torch.sign(pred)
        target_dir = torch.sign(target)
        return (pred_dir == target_dir).float().mean().item()

    def _prepare_outputs_for_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.task == "regression":
            return output.squeeze(-1), target.float()
        return output, target

    def _undo_lob_normalization_for_prices(
        self,
        ask_price: torch.Tensor,
        ask_size: torch.Tensor,
        bid_price: torch.Tensor,
        bid_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalizer = self.lob_normalizer
        if normalizer is None:
            return ask_price, ask_size, bid_price, bid_size

        price_mean = getattr(normalizer, "price_mean", 0.0)
        price_std = getattr(normalizer, "price_std", 1.0)
        ask_price = ask_price * price_std + price_mean
        bid_price = bid_price * price_std + price_mean

        vol_mean = getattr(normalizer, "vol_mean", 0.0)
        vol_std = getattr(normalizer, "vol_std", 1.0)
        ask_size = torch.expm1(ask_size * vol_std + vol_mean)
        bid_size = torch.expm1(bid_size * vol_std + vol_mean)
        return ask_price, ask_size, bid_price, bid_size

    def _extract_scalar_price_history(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] < 4:
            raise ValueError(
                f"Need input shape (B, L, >=4) for scalar price history, got {tuple(x.shape)}"
            )

        ask_price = x[:, :, 0]
        ask_size = x[:, :, 1]
        bid_price = x[:, :, 2]
        bid_size = x[:, :, 3]
        ask_price, ask_size, bid_price, bid_size = self._undo_lob_normalization_for_prices(
            ask_price, ask_size, bid_price, bid_size
        )

        mid = (ask_price + bid_price) / 2.0
        if self.reg_projection_target == "mid":
            return mid.unsqueeze(-1)

        micro = (bid_price * ask_size + ask_price * bid_size) / (ask_size + bid_size + 1e-9)
        if self.reg_projection_target == "micro":
            return micro.unsqueeze(-1)
        if self.reg_projection_target == "avg":
            return ((mid + micro) / 2.0).unsqueeze(-1)
        raise ValueError(f"Unknown reg_projection_target: {self.reg_projection_target}")

    def _prepare_target_shift(self, x: torch.Tensor) -> None:
        if self.target_shift_norm is not None:
            self.target_shift_norm.precompute(self._extract_scalar_price_history(x))

    def _normalize_regression_target(self, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if self.target_shift_norm is not None:
            return self.target_shift_norm.normalize_target(target)
        return target

    def _inverse_regression_prediction(self, pred: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        if self.target_shift_norm is not None:
            return self.target_shift_norm.inverse_target(pred)
        return pred

    def _run_one_epoch(
        self,
        dataloader,
        train: bool = True,
    ) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        all_labels = []
        all_preds = []

        loop = tqdm(dataloader, disable=False)
        for batch in loop:
            # Hem dict ("x","y") hem tuple (x,y) destekle
            if isinstance(batch, dict):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

            bs = y.size(0)
            total_samples += bs

            if train:
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    self._prepare_target_shift(x)
                    logits = self.model(x)
                    loss_pred, loss_target = self._prepare_outputs_for_loss(logits, y)
                    if self.task == "regression":
                        loss_target = self._normalize_regression_target(loss_target)
                    loss = self.criterion(loss_pred, loss_target)

                self.scaler.scale(loss).backward()

                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with torch.no_grad():
                    self._prepare_target_shift(x)
                    logits = self.model(x)
                    loss_pred, loss_target = self._prepare_outputs_for_loss(logits, y)
                    if self.task == "regression":
                        loss_target = self._normalize_regression_target(loss_target)
                    loss = self.criterion(loss_pred, loss_target)

            total_loss += loss.item() * bs
            if self.task == "regression":
                metric_pred = self._inverse_regression_prediction(loss_pred.detach())
                metric_target = y.detach().float()
                total_acc += self._compute_directional_accuracy(metric_pred, metric_target) * bs
                all_labels.append(metric_target.cpu())
                all_preds.append(metric_pred.cpu())
            else:
                total_acc += self._compute_accuracy(logits, y) * bs
                all_labels.append(y.detach().cpu())
                all_preds.append(logits.detach().cpu().argmax(dim=1))

            avg_loss = total_loss / total_samples
            avg_acc = total_acc / total_samples

            mode = "train" if train else "valid"
            metric_name = "dir_acc" if self.task == "regression" else "acc"
            loop.set_description(f"{mode} | loss: {avg_loss:.4f} | {metric_name}: {avg_acc:.4f}")

        out = {
            "loss": total_loss / total_samples,
            "accuracy": total_acc / total_samples,
        }
        if all_labels:
            out["labels"] = torch.cat(all_labels).numpy()
            out["preds"] = torch.cat(all_preds).numpy()
        return out
    """
        def fit(self, train_loader, val_loader):
        best_state = copy.deepcopy(self.model.state_dict())
        best_val_acc = -np.inf
        epochs_no_improve = 0

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            train_metrics = self._run_one_epoch(train_loader, train=True)
            val_metrics = self._run_one_epoch(val_loader, train=False)

            train_loss = train_metrics["loss"]
            train_acc = train_metrics["accuracy"]
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            print(
                f"Epoch {epoch} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            # ---- LR scheduler (ReduceLROnPlateau gibi) ----
            if self.scheduler is not None:
                try:
                    # çoğu zaman val_loss veya val_acc ile çalışır
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

            if val_acc > best_val_acc + self.es_min_delta:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                print(f"  >> New best model! val_acc = {val_acc:.4f}")
            else:
                epochs_no_improve += 1
                print(
                    f"  >> No improvement in val_acc for "
                    f"{epochs_no_improve}/{self.es_patience} epochs."
                )

            if epochs_no_improve >= self.es_patience:
                print("Early stopping triggered.")
                break

        # En iyi accuracy'li modeli geri yükle
        self.model.load_state_dict(best_state)
        print(f"Training finished. Best val_acc = {best_val_acc:.4f}")
        return self.model
    
    """
    def fit(self, train_loader, val_loader):
        best_state = copy.deepcopy(self.model.state_dict())

        best_val_loss = np.inf
        best_val_acc = 0.0
        best_bal_acc = 0.0
        epochs_no_improve = 0
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_bal_acc": [],
        }

        def _balanced_acc(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            # balanced accuracy = mean recall over classes
            uniq = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
            recalls = []
            for c in uniq:
                mask = y_true == c
                if mask.sum() == 0:
                    continue
                recalls.append((y_pred[mask] == c).mean())
            return float(np.mean(recalls)) if recalls else 0.0

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            train_metrics = self._run_one_epoch(train_loader, train=True)
            val_metrics = self._run_one_epoch(val_loader, train=False)

            train_loss = train_metrics["loss"]
            train_acc = train_metrics["accuracy"]
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            val_bal_acc = (
                val_acc
                if self.task == "regression"
                else _balanced_acc(val_metrics.get("labels", []), val_metrics.get("preds", [])) if "labels" in val_metrics else val_acc
            )

            history["train_loss"].append(float(train_loss))
            history["train_acc"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            history["val_bal_acc"].append(float(val_bal_acc))

            if self.task == "regression":
                print(
                    f"Epoch {epoch} | train_loss={train_loss:.4f}, train_dir_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f}, val_dir_acc={val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch} | "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_bal_acc={val_bal_acc:.4f}"
                )

            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

            if self.task == "regression":
                improved = val_loss < (best_val_loss - self.es_min_delta)
            else:
                improved = False
                if val_bal_acc > (best_bal_acc + self.es_min_delta):
                    improved = True
                elif abs(val_bal_acc - best_bal_acc) <= self.es_min_delta and val_loss < (best_val_loss - self.es_min_delta):
                    improved = True

            if improved:
                best_bal_acc = val_bal_acc
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                if self.task == "regression":
                    print(f"  >> New best model! (val_loss={best_val_loss:.4f})")
                else:
                    print(f"  >> New best model! (val_bal_acc={best_bal_acc:.4f}, val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  >> No improvement ({epochs_no_improve}/{self.es_patience})")

            if epochs_no_improve >= self.es_patience:
                print("Early stopping triggered.")
                break

        self.model.load_state_dict(best_state)
        print("Training finished.")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Acc at Best: {best_val_acc:.4f}")
        print(f"Balanced Acc at Best: {best_bal_acc:.4f}")

        self.best_val_loss = best_val_loss
        self.best_val_acc = best_val_acc
        self.best_val_bal_acc = best_bal_acc
        self.history = history

        return history
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=False, desc="test"):
                if isinstance(batch, dict):
                    x = batch["x"].to(self.device)
                    y = batch["y"].to(self.device)
                else:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)

                logits = self.model(x)
                loss_pred, loss_target = self._prepare_outputs_for_loss(logits, y)
                self._prepare_target_shift(x)
                if self.task == "regression":
                    loss_target = self._normalize_regression_target(loss_target)
                loss = self.criterion(loss_pred, loss_target)

                bs = y.size(0)
                total_loss += loss.item() * bs
                total_samples += bs

                if self.task == "regression":
                    preds_list.append(self._inverse_regression_prediction(loss_pred).cpu())
                    labels_list.append(y.float().cpu())
                else:
                    preds_list.append(logits.argmax(dim=1).cpu())
                    labels_list.append(y.cpu())

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        if preds_list:
            predictions = torch.cat(preds_list).numpy()
            labels = torch.cat(labels_list).numpy()
        else:
            predictions = np.array([])
            labels = np.array([])

        return {
            "loss": avg_loss,
            "predictions": predictions,
            "labels": labels,
        }

    
