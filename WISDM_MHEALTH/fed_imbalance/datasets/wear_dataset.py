import os, pandas as pd, numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

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

def _sliding_windows(X: np.ndarray, y: np.ndarray, window_size:int, stride:int) -> Tuple[np.ndarray, np.ndarray]:
    T, C = X.shape
    xs, ys = [], []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        xw = X[start:end]
        yw = y[start:end]

        # Majority vote
        lab = np.bincount(yw).argmax()

        # NaN handling
        if np.all(np.isnan(xw)):
            xw = np.zeros_like(xw)
        col_med = np.nanmedian(xw, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        xw = np.where(np.isnan(xw), col_med, xw)

        xs.append(xw)
        ys.append(lab)
    if not xs:
        return (np.zeros((0, window_size, X.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)

def make_train_val_split(X_tr: np.ndarray, y_tr: np.ndarray, val_pct_of_train: float = 0.1):
    if X_tr.size == 0:
        return X_tr, y_tr, np.zeros((0,) + X_tr.shape[1:], dtype=X_tr.dtype), np.zeros((0,), dtype=y_tr.dtype)
    idx_all = np.arange(len(y_tr))
    val_idx, new_train_idx = [], []
    for c in np.unique(y_tr):
        cls_idx = idx_all[y_tr == c]
        n = len(cls_idx)
        n_val = max(1, int(round(n * val_pct_of_train)))
        if n_val >= n:
            n_val = max(1, n - 1)
        val_idx.extend(cls_idx[-n_val:])
        new_train_idx.extend(cls_idx[:-n_val])
    val_idx = np.array(val_idx, dtype=int)
    new_train_idx = np.array(new_train_idx, dtype=int)
    return X_tr[new_train_idx], y_tr[new_train_idx], X_tr[val_idx], y_tr[val_idx]

def _per_label_split(X: np.ndarray, y: np.ndarray, test_pct: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx_all = np.arange(len(y))
    test_idx, train_idx = [], []
    for c in np.unique(y):
        cls_idx = idx_all[y == c]
        if cls_idx.size == 0:
            continue
        cutoff = int(cls_idx.size * test_pct)
        cutoff = max(cutoff, 1)  # ensure at least 1 sample goes to test if exists
        test_idx.extend(cls_idx[:cutoff])
        train_idx.extend(cls_idx[cutoff:])
    return X[np.array(train_idx)], y[np.array(train_idx)], X[np.array(test_idx)], y[np.array(test_idx)]

def _standardize_train_apply(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X_train.size == 0:
        return X_train, X_test
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (X_train - mean) / std, (X_test - mean) / std

def _cap_null_and_enforce_min_per_class(
    X: np.ndarray, y: np.ndarray, null_idx: int, max_null_ratio: float = 0.30, min_per_class: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        return X, y

    # Cap NULL ratio
    if null_idx is not None:
        total = y.shape[0]
        if total > 0:
            mask = np.ones(total, dtype=bool)
            null_mask = (y == null_idx)
            n_null = null_mask.sum()
            max_null = int(total * max_null_ratio)
            if n_null > max_null:
                # keep max_null of NULL at earliest indices and drop the rest
                keep_idx = np.flatnonzero(null_mask)[:max_null]
                drop_idx = np.setdiff1d(np.flatnonzero(null_mask), keep_idx, assume_unique=False)
                mask[drop_idx] = False
                X, y = X[mask], y[mask]

    # Enforce min per class
    binc = np.bincount(y, minlength=y.max() + 1 if y.size else 1)
    valid = np.where(binc >= min_per_class)[0]
    if valid.size == 0:  # if everything is too rare, return as-is
        return X, y
    keep = np.isin(y, valid)
    return X[keep], y[keep]

def load_wear_clients(
    data_dir: str,
    features: List[str],
    label_col: str,
    window_size: int,
    window_stride: int,
    test_pct: float = 0.2,
    cap_null_in_train: bool = True,
    max_null_ratio: float = 0.30,
    min_per_class: int = 3,
) -> Dict[str, WearClientData]:
    clients: Dict[str, WearClientData] = {}

    # 1) Collect all files
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    # 2) First pass: load + sort by timestamp + collect labels for global factorization
    all_labels = []
    raw = {}
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f), low_memory=False)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        cols = [c for c in features if c in df.columns]
        if len(cols) != len(features):
            raise ValueError(f"File {f} missing features. Found {cols}, expected {features}")

        # numeric features with NaN allowed (filled later at windowing)
        X = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in cols], axis=1).values

        # label column
        lab = df[label_col].fillna("NULL").astype(str)
        raw[f] = (X, lab)
        all_labels.extend(list(lab.unique()))

    # 3) Global factorization so class ids are consistent across clients
    _, classes = _factorize_labels(pd.Series(all_labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    null_idx = label_to_idx.get("NULL", None)

    # 4) Second pass: subject-wise split → windowing → standardize
    for f, (X, y_str) in raw.items():
        # Encode labels using global map
        y = np.array([label_to_idx[s] for s in y_str.to_numpy(dtype=str)], dtype=np.int64)

        # FedFitTech per-label split (first 20% to TEST)
        X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = _per_label_split(X, y, test_pct=test_pct)

        # Windowing on TRAIN/TEST separately
        X_train, y_train = _sliding_windows(X_tr_raw, y_tr_raw, window_size, window_stride)
        X_test, y_test   = _sliding_windows(X_te_raw, y_te_raw, window_size, window_stride)

        # Stability guards (TRAIN only)
        if cap_null_in_train and null_idx is not None and X_train.size > 0:
            X_train, y_train = _cap_null_and_enforce_min_per_class(
                X_train, y_train, null_idx=null_idx,
                max_null_ratio=max_null_ratio, min_per_class=min_per_class
            )

        # Standardize by train statistics
        if X_train.size > 0:
            X_train, X_test = _standardize_train_apply(X_train, X_test)

        # Save
        cid = os.path.splitext(f)[0]  # "sbj_XX" file base
        clients[cid] = WearClientData(X_train, y_train, X_test, y_test, classes)

        # Debug (kept as print; redirect to your logger if preferred)
        tr_dist = np.bincount(y_train, minlength=len(classes)) if y_train.size else np.zeros(len(classes), dtype=int)
        te_dist = np.bincount(y_test, minlength=len(classes)) if y_test.size else np.zeros(len(classes), dtype=int)
        print(f"[FedFitTech split] {cid}: train={tr_dist.tolist()}  test={te_dist.tolist()}")

    return clients
