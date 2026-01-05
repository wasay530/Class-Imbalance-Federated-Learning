
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

# Keep the same shape and API as wear_dataset
@dataclass
class WearClientData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: List[str]

def _factorize_labels(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
    codes, uniques = pd.factorize(series.astype(str), sort=True)
    return codes.astype(int), [str(u) for u in uniques]

def _sliding_windows(X: np.ndarray, y: np.ndarray, window_size: int, window_stride: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return np.zeros((0, window_size, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    xs, ys = [], []
    for start in range(0, len(X) - window_size + 1, window_stride):
        end = start + window_size
        xw = X[start:end]
        yw = y[start:end]

        # Majority vote for label (ties -> smallest label id)
        vals, counts = np.unique(yw, return_counts=True)
        lab = vals[np.argmax(counts)]

        # Replace NaNs in window with per-column nanmedian
        if np.all(np.isnan(xw)):
            xw = np.zeros_like(xw)
        col_med = np.nanmedian(xw, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        xw = np.where(np.isnan(xw), col_med, xw)

        xs.append(xw)
        ys.append(lab)

    if not xs:
        return np.zeros((0, window_size, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    Xw = np.stack(xs).astype(np.float32)
    yw = np.asarray(ys, dtype=np.int64)
    return Xw, yw

def _standardize_train_apply(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=(0,1), keepdims=True)
    sd = X_train.std(axis=(0,1), keepdims=True) + 1e-8
    return (X_train - mu)/sd, (X_test - mu)/sd

def _chrono_per_class_split(y: np.ndarray, test_pct: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    test_idx = []
    for c in np.unique(y):
        cls_idx = idx[y == c]
        n_test = max(1, int(len(cls_idx) * test_pct))
        test_idx.append(cls_idx[:n_test])  # earliest indices for test
    test_idx = np.concatenate(test_idx) if len(test_idx) else np.array([], dtype=int)
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_idx] = False
    train_idx = idx[train_mask]
    return train_idx, test_idx

def make_train_val_split(y: np.ndarray, val_pct_of_train: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    val_idx = []
    for c in np.unique(y):
        cls_idx = idx[y == c]
        n_val = max(1, int(len(cls_idx) * val_pct_of_train))
        val_idx.append(cls_idx[:n_val])
    val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=int)
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    train_idx = idx[train_mask]
    return train_idx, val_idx

def _read_wisdm_raw(raw_path: str) -> pd.DataFrame:
    rows = []
    with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("user") or line.startswith("#"):
                continue
            if line.endswith(';'):
                line = line[:-1]
            parts = line.split(',')
            if len(parts) != 6:
                continue
            user, act, ts, x, y, z = parts
            try:
                rows.append((int(user), str(act), int(float(ts)), float(x), float(y), float(z)))
            except Exception:
                # Skip malformed rows
                continue
    df = pd.DataFrame(rows, columns=["user","activity","timestamp","x","y","z"])
    df.sort_values(["user","timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_wisdm_clients(data_dir: str,
                       window_size: int,
                       window_stride: int,
                       features: Iterable[str],
                       label_col: str,
                       test_pct: float = 0.2,
                       val_pct_of_train: float = 0.1) -> Dict[str, WearClientData]:
    # Resolve raw file
    raw_path = data_dir
    if os.path.isdir(data_dir):
        # try common filename inside directory
        candidates = [
            os.path.join(data_dir, "WISDM_ar_v1.1_raw.txt"),
            os.path.join(data_dir, "wisdm_ar_v1.1_raw.txt"),
            *[os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".txt") and "wisdm" in f.lower() and "raw" in f.lower()]
        ]
        for c in candidates:
            if os.path.isfile(c):
                raw_path = c
                break

    df = _read_wisdm_raw(raw_path)
    # Factorize activity labels globally to keep consistent mapping across clients
    y_all, classes = _factorize_labels(df["activity"])
    df = df.assign(label_id=y_all)

    clients: Dict[str, WearClientData] = {}
    for user_id, g in df.groupby("user", sort=True):
        # Extract sequences in chronological order
        X = g[["x","y","z"]].to_numpy(dtype=np.float32)
        y = g["label_id"].to_numpy(dtype=np.int64)

        # Create windows
        Xw, yw = _sliding_windows(X, y, window_size, window_stride)
        if len(yw) == 0:
            continue

        # Chronological per-class split to TEST, remainder TRAIN
        train_idx, test_idx = _chrono_per_class_split(yw, test_pct=test_pct)

        X_train, y_train = Xw[train_idx], yw[train_idx]
        X_test,  y_test  = Xw[test_idx],  yw[test_idx]

        # Standardize using TRAIN stats (per-client)
        if X_train.size > 0:
            X_train, X_test = _standardize_train_apply(X_train, X_test)

        cid = f"user_{int(user_id):02d}"
        clients[cid] = WearClientData(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            classes=classes
        )

        # Debug print resembling WEAR loader
        tr_dist = np.bincount(y_train, minlength=len(classes)) if y_train.size else np.zeros(len(classes), dtype=int)
        te_dist = np.bincount(y_test,  minlength=len(classes)) if y_test.size  else np.zeros(len(classes), dtype=int)
        print(f"[WISDM split] {cid}: train={tr_dist.tolist()}  test={te_dist.tolist()}")

    return clients
