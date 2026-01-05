
import os, glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

@dataclass
class WearClientData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: List[str]

def _factorize_labels(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
    # keep deterministic order
    codes, uniques = pd.factorize(series.astype(int), sort=True)
    return codes.astype(int), [str(u) for u in uniques]

def _sliding_windows(X: np.ndarray, y: np.ndarray, window_size: int, window_stride: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return np.zeros((0, window_size, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xs, ys = [], []
    for start in range(0, len(X) - window_size + 1, window_stride):
        end = start + window_size
        xw = X[start:end]
        yw = y[start:end]
        vals, counts = np.unique(yw, return_counts=True)
        lab = vals[np.argmax(counts)]
        if np.all(np.isnan(xw)):
            xw = np.zeros_like(xw)
        col_med = np.nanmedian(xw, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        xw = np.where(np.isnan(xw), col_med, xw)
        xs.append(xw); ys.append(lab)
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
        test_idx.append(cls_idx[:n_test])
    test_idx = np.concatenate(test_idx) if len(test_idx) else np.array([], dtype=int)
    train_mask = np.ones(len(y), dtype=bool); train_mask[test_idx] = False
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
    train_mask = np.ones(len(y), dtype=bool); train_mask[val_idx] = False
    train_idx = idx[train_mask]
    return train_idx, val_idx

def _read_mhealth_folder(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "mHealth_subject*.log")))
    if not files and path.lower().endswith(".log") and os.path.isfile(path):
        files = [path]
    rows = []
    for p in files:
        # subject id from filename
        m = os.path.basename(p)
        sid = None
        for token in m.split('_'):
            if token.startswith("subject"):
                try:
                    sid = int(token.replace("subject", "").split('.')[0])
                except Exception:
                    pass
        sid = sid if sid is not None else -1
        # read with flexible whitespace
        df = pd.read_csv(p, sep=r'\s+', header=None, engine='python')
        if df.shape[1] < 24:
            df = pd.read_csv(p, sep=r'\t+', header=None, engine='python')
        if df.shape[1] < 24:
            raise ValueError(f"Unexpected column count in {p}: {df.shape[1]} (expected 24)")
        df["user"] = sid
        rows.append(df)
    big = pd.concat(rows, ignore_index=True)
    # 23 signals + label
    sig_cols = [f"feat_{i}" for i in range(1, 24)]
    big.columns = sig_cols + ["label", "user"]
    # dtypes
    for c in sig_cols:
        big[c] = pd.to_numeric(big[c], errors='coerce')
    big["label"] = big["label"].astype(int)
    big.sort_values(["user"], inplace=True)
    big.reset_index(drop=True, inplace=True)
    return big

def _read_mhealth_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize column names
    rename = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename)
    lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower)

    if "activity" not in df.columns or "subject" not in df.columns:
        raise ValueError("CSV must include 'Activity' and 'subject' columns.")

    # choose feature columns: all numeric columns except 'activity' and 'subject'
    non_feat = {"activity", "subject"}
    feature_cols = [c for c in df.columns if c not in non_feat and np.issubdtype(df[c].dtype, np.number)]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in CSV.")

    # enforce dtypes
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df["label"] = df["activity"].astype(int)
    # derive numeric user id from 'subject' strings like 'subject1'
    def _to_uid(s):
        import re
        m = re.findall(r'\d+', str(s))
        return int(m[-1]) if m else -1
    df["user"] = df["subject"].apply(_to_uid)

    # reorder: features..., label, user
    cols = feature_cols + ["label", "user"]
    df = df[cols].copy()
    df.sort_values(["user"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, feature_cols

def load_mhealth_clients(data_dir: str,
                         window_size: int,
                         window_stride: int,
                         features: Optional[Iterable[str]] = None,
                         label_col: str = "label",
                         test_pct: float = 0.2,
                         val_pct_of_train: float = 0.1) -> Dict[str, WearClientData]:
    using_csv = (data_dir.lower().endswith(".csv") and os.path.isfile(data_dir))
    if using_csv:
        df, csv_feats = _read_mhealth_csv(data_dir)
        # If features not provided, use CSV's detected feature columns
        feature_cols = list(features) if features is not None else csv_feats
    else:
        folder = data_dir if os.path.isdir(data_dir) else os.path.dirname(data_dir) or "."
        df = _read_mhealth_folder(folder)
        feature_cols = list(features) if features is not None else [f"feat_{i}" for i in range(1, 24)]

    # Factorize labels globally
    y_all, classes = _factorize_labels(df[label_col])
    df = df.assign(label_id=y_all)

    clients: Dict[str, WearClientData] = {}
    for user_id, g in df.groupby("user", sort=True):
        X = g[feature_cols].to_numpy(dtype=np.float32)
        y = g["label_id"].to_numpy(dtype=np.int64)

        Xw, yw = _sliding_windows(X, y, window_size, window_stride)
        if len(yw) == 0:
            continue

        train_idx, test_idx = _chrono_per_class_split(yw, test_pct=test_pct)

        X_train, y_train = Xw[train_idx], yw[train_idx]
        X_test,  y_test  = Xw[test_idx],  yw[test_idx]

        if X_train.size > 0:
            X_train, X_test = _standardize_train_apply(X_train, X_test)

        cid = f"user_{int(user_id):02d}"
        clients[cid] = WearClientData(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            classes=classes
        )

        tr_dist = np.bincount(y_train, minlength=len(classes)) if y_train.size else np.zeros(len(classes), dtype=int)
        te_dist = np.bincount(y_test,  minlength=len(classes)) if y_test.size  else np.zeros(len(classes), dtype=int)
        src = "CSV" if using_csv else "LOG"
        print(f"[MHEALTH {src} split] {cid}: train={tr_dist.tolist()}  test={te_dist.tolist()}")

    return clients
