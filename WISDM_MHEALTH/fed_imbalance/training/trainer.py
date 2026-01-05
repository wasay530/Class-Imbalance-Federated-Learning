import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from fed_imbalance.utils.metrics import compute_metrics


def make_loader(X, y, batch_size, shuffle):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_local(model: nn.Module, optimizer, criterion, 
                X, y, batch_size: int, epochs: int, device: str, 
                normalize_logits: bool = False):
    # Train locally on one client.
    if X is None or len(X) == 0:
        return {"loss": None, "accuracy": 0.0, "f1_macro": 0.0}

    model.to(device)
    model.train()
    loader = make_loader(X, y, batch_size, shuffle=True)

    total_loss, n = 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            bs = yb.size(0)
            total_loss += float(loss.detach()) * bs
            n += bs

    # final metrics on train set
    m = _evaluate_metrics(model, X, y, batch_size, device, normalize_logits)
    return {
        "loss": float(total_loss / max(1, n)),
        "accuracy": float(m["accuracy"]),
        "f1_macro": float(m["f1_macro"])
    }


@torch.no_grad()
def evaluate(model: nn.Module, X, y, batch_size: int, device: str, normalize_logits: bool = False):
    # Evaluate model and return predictions + probabilities.
    model.to(device)
    model.eval()
    if X is None or len(X) == 0:
        return np.array([]), np.array([]), np.empty((0, 0))

    loader = make_loader(X, y, batch_size, shuffle=False)
    y_true, y_pred, y_prob = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def _evaluate_metrics(model, X, y, batch_size, device, normalize_logits=False, null_index=None):
    y_true, y_pred, _ = evaluate(model, X, y, batch_size, device, normalize_logits)
    return compute_metrics(y_true, y_pred, null_index=null_index)
