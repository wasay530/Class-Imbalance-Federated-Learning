import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(y_true, y_pred, null_index=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0}

    if null_index is not None:
        mask = (y_true != null_index)
        # If everything is null, avoid empty slice
        if mask.sum() == 0:
            return {"accuracy": 0.0, "f1_macro": 0.0}
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true.size else 0.0
    return {"accuracy": acc, "f1_macro": f1m}
