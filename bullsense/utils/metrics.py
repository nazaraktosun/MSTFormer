# bullsense/utils/metrics.py

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
import numpy as np


def compute_metrics(y_true, y_pred, labels=None):
    """
    Compute metrics for arbitrary class counts. If labels is None, labels are
    inferred from the union of y_true and y_pred (sorted).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(np.unique(y_true).tolist() + np.unique(y_pred).tolist()))
    labels = list(labels)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )

    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": f1_per_class.tolist(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mse": float("nan"),
            "directional_accuracy": float("nan"),
            "correlation": float("nan"),
            "n": 0,
        }

    err = y_pred - y_true
    mse = float(np.mean(err**2))
    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "directional_accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "correlation": corr,
        "n": int(y_true.size),
    }


def print_regression_metrics(metrics):
    print(f"\n{'='*60}")
    print(" TEST REGRESSION METRICS")
    print(f"{'='*60}")
    print(f"  MAE:                 {metrics['mae']:.6f}")
    print(f"  RMSE:                {metrics['rmse']:.6f}")
    print(f"  Directional Acc:     {metrics['directional_accuracy']:.4f}")
    print(f"  Correlation:         {metrics['correlation']:.4f}")
    print(f"  N:                   {metrics['n']}")
    print(f"{'='*60}\n")


def print_metrics(metrics, class_names=None):
    """Pretty print metrics with dynamic class names."""
    labels = metrics.get("labels")
    f1_per_class = metrics.get("f1_per_class", [])

    if class_names is None:
        if "class_names" in metrics:
            class_names = metrics["class_names"]
        elif labels is not None:
            class_names = [f"CLASS_{lbl}" for lbl in labels]
        else:
            class_names = [f"CLASS_{i}" for i in range(len(f1_per_class))]

    print(f"\n{'='*60}")
    print(" TEST METRICS")
    print(f"{'='*60}")
    print(f"\n  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:      {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted:   {metrics['f1_weighted']:.4f}")

    print(f"\n  Per-Class Metrics:")
    for i, cls_name in enumerate(class_names):
        f1 = f1_per_class[i]
        prec = metrics["precision_per_class"][i]
        rec = metrics["recall_per_class"][i]
        sup = metrics["support_per_class"][i]

        status = "good" if f1 > 0.5 else ("eh" if f1 > 0.3 else "bad")
        print(f"    {cls_name:>12}: F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  (n={sup}) {status}")

    print(f"\n{'='*60}\n")
