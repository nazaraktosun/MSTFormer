"""
Dataset class for sequence modeling
"""
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


_PRICE_RE = re.compile(r"^p[ab]\d+$", re.IGNORECASE)
_VOL_RE = re.compile(r"^q[ab]\d+$", re.IGNORECASE)


def _detect_feature_indices(feature_names):
    """Bucket each feature column into (price, vol, other-feature).

    price/vol use pooled stats (one mean/std across all levels).
    "feature" gets per-column stats — distributions differ across
    derived features (mid_price ~68, spread ~0.01, imbalance ~0).
    """
    price_idx = []
    vol_idx = []
    feature_idx = []
    for i, name in enumerate(feature_names):
        if _PRICE_RE.match(name):
            price_idx.append(i)
        elif _VOL_RE.match(name):
            vol_idx.append(i)
        else:
            feature_idx.append(i)
    return price_idx, vol_idx, feature_idx


# Backwards compatibility — same signature as before, just drops the new bucket.
def _detect_lob_indices(feature_names):
    price_idx, vol_idx, _ = _detect_feature_indices(feature_names)
    return price_idx, vol_idx


class LOBNormalizer:
    def __init__(self, price_idx=None, vol_idx=None, feature_idx=None, style: str = "bullsense"):
        self.price_mean = 0
        self.price_std = 1
        self.vol_mean = 0
        self.vol_std = 1
        self.style = style
        self.price_idx = [] if price_idx is None else list(price_idx)
        self.vol_idx = [] if vol_idx is None else list(vol_idx)
        self.feature_idx = [] if feature_idx is None else list(feature_idx)
        # Per-column stats for the "feature" bucket (shape = [len(feature_idx)])
        self.feature_means = None
        self.feature_stds = None

    def fit(self, X_train):
        """Fit normalization stats on X_train only."""
        eps = 1e-8

        if self.style == "full_zscore":
            means = X_train.mean(axis=(0, 1)).astype(np.float32, copy=False)
            stds = (X_train.std(axis=(0, 1)) + eps).astype(np.float32, copy=False)
            self.feature_idx = list(range(X_train.shape[2]))
            self.feature_means = means
            self.feature_stds = stds
            self.price_idx = []
            self.vol_idx = []
            self.price_mean, self.price_std = 0.0, 1.0
            self.vol_mean, self.vol_std = 0.0, 1.0
            print(f"Stats -> full per-column z-score cols: {len(self.feature_idx)}")
            return

        def _safe_slice(idx_list):
            if not idx_list:
                return None
            return X_train[:, :, idx_list]

        prices = _safe_slice(self.price_idx)
        if prices is not None:
            self.price_mean = float(np.mean(prices))
            self.price_std = float(np.std(prices) + eps)
        else:
            self.price_mean, self.price_std = 0.0, 1.0

        vols = _safe_slice(self.vol_idx)
        if vols is not None:
            if self.style == "bullsense":
                vols = np.log1p(vols)
            self.vol_mean = float(np.mean(vols))
            self.vol_std = float(np.std(vols) + eps)
        else:
            self.vol_mean, self.vol_std = 0.0, 1.0

        feats = _safe_slice(self.feature_idx)
        if feats is not None:
            feats = feats.astype(np.float32, copy=False)
            # Per-column: collapse across windows (axis 0) and time (axis 1)
            self.feature_means = feats.mean(axis=(0, 1))
            self.feature_stds = feats.std(axis=(0, 1)) + eps
        else:
            self.feature_means = None
            self.feature_stds = None

        n_feats = 0 if self.feature_means is None else len(self.feature_means)
        print(
            f"Stats -> price mean: {self.price_mean:.4f}, "
            f"vol mean ({'log1p' if self.style == 'bullsense' else 'raw'}): {self.vol_mean:.4f}, "
            f"feature cols z-scored: {n_feats}"
        )

    def normalize(self, x_tensor):
        """Apply group-wise normalization to a [T, F] sample tensor."""
        x_norm = x_tensor.clone()

        if self.style != "features_only" and self.price_idx:
            idx = torch.tensor(self.price_idx, device=x_tensor.device)
            x_norm[:, idx] = (x_tensor[:, idx] - self.price_mean) / self.price_std

        if self.style != "features_only" and self.vol_idx:
            idx = torch.tensor(self.vol_idx, device=x_tensor.device)
            vols = torch.log1p(x_tensor[:, idx]) if self.style == "bullsense" else x_tensor[:, idx]
            x_norm[:, idx] = (vols - self.vol_mean) / self.vol_std

        if self.feature_idx and self.feature_means is not None:
            idx = torch.tensor(self.feature_idx, device=x_tensor.device)
            means = torch.as_tensor(
                self.feature_means, device=x_tensor.device, dtype=x_tensor.dtype
            )
            stds = torch.as_tensor(
                self.feature_stds, device=x_tensor.device, dtype=x_tensor.dtype
            )
            x_norm[:, idx] = (x_tensor[:, idx] - means) / stds

        return x_norm
    

class OrderBookDataset(Dataset):
    """Dataset for order book sequences"""
    
    def __init__(self, X, y, normalizer=None, task: str = "classification"): 
        X = np.asarray(X, dtype=np.float32)
        self.task = task
        if task == "regression":
            y = np.asarray(y, dtype=np.float32)
        else:
            y = np.asarray(y, dtype=np.int64)
        
        if X.ndim == 2:
            X = X[:, None, :]
        
        assert X.ndim == 3, f"X must be [N,T,F], got shape {X.shape}"
        if task != "regression":
            assert y.min() >= 0, f"Labels must be >= 0, got min={y.min()}"
        
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.normalizer = normalizer
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]
        
        if self.normalizer:
            x_sample = self.normalizer.normalize(x_sample)
            
        return x_sample, y_sample


def load_data(data_dir='data/processed', mmap=True):
    """Load preprocessed data with optional memory mapping"""
    data_dir = Path(data_dir)
    mm = "r" if mmap else None
    
    X_train = np.load(data_dir / 'X_train.npy', mmap_mode=mm)
    y_train = np.load(data_dir / 'y_train.npy', mmap_mode=mm)
    X_val = np.load(data_dir / 'X_val.npy', mmap_mode=mm)
    y_val = np.load(data_dir / 'y_val.npy', mmap_mode=mm)
    X_test = np.load(data_dir / 'X_test.npy', mmap_mode=mm)
    y_test = np.load(data_dir / 'y_test.npy', mmap_mode=mm)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_dataloaders(
    batch_size=128,
    num_workers=4,
    data_dir='data/processed',
    use_normalizer: bool = True,
    task: str | None = None,
    normalizer_style: str = "bullsense",
):
    """Create train/val/test dataloaders."""
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(data_dir)

    feature_names = None
    price_idx = None
    vol_idx = None
    feature_idx: list[int] = []

    meta_path = Path(data_dir) / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            feature_names = meta.get("feature_names")
            if task is None and meta.get("label", {}).get("strategy") == "regression":
                task = "regression"
        except Exception:
            feature_names = None
    task = task or "classification"

    if feature_names:
        price_idx, vol_idx, feature_idx = _detect_feature_indices(feature_names)
        if not price_idx and not vol_idx and not feature_idx:
            print("[WARN] metadata.json yüklendi ama tanınabilir kolon bulunamadı; normalizasyon atlanacak.")
    else:
        print("[INFO] metadata.json bulunamadı; eski davranış (0::2 fiyat, 1::2 hacim) geçerli.")
        price_idx = list(range(0, X_train.shape[2], 2))
        vol_idx = list(range(1, X_train.shape[2], 2))
        feature_idx = []

    normalizer = None
    if use_normalizer and (price_idx or vol_idx or feature_idx):
        normalizer = LOBNormalizer(
            price_idx=price_idx,
            vol_idx=vol_idx,
            feature_idx=feature_idx,
            style=normalizer_style,
        )
        normalizer.fit(X_train)
    
    # Normalizer'ı datasetlere gönderiyoruz
    train_dataset = OrderBookDataset(X_train, y_train, normalizer=normalizer, task=task)
    val_dataset = OrderBookDataset(X_val, y_val, normalizer=normalizer, task=task)
    test_dataset = OrderBookDataset(X_test, y_test, normalizer=normalizer, task=task)
    
    common_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None
    }
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, drop_last=True, 
                              **common_kwargs)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            drop_last=False, **common_kwargs)
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             drop_last=False, 
                             **common_kwargs)
    
    return train_loader, val_loader, test_loader
