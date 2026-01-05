import os, time, numpy as np, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

from fed_imbalance.training.trainer import train_local, evaluate
from fed_imbalance.training.federated import get_weights, set_weights, fedavg
from fed_imbalance.losses.ldam import LDAMLoss, compute_drw_weights, TinyHARWithNormalizedLogits
from fed_imbalance.datasets.wear_dataset import make_train_val_split
from fed_imbalance.losses.focal import focal_loss
from fed_imbalance.losses.ratio import RatioLoss
from fed_imbalance.utils.metrics import compute_metrics


def _ensure_dir(p): 
    os.makedirs(p, exist_ok=True)


def _create_client_plots(cid, method_name, y_true, y_pred, y_prob, classes, num_classes, plot_dir):
    if y_true.size == 0:
        return
    _ensure_dir(plot_dir)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, values_format=".0f")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"client_{cid}_confusion.svg"))
    plt.close(fig)

    # ROC curve (micro-avg)
    try:
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
        row_sums = y_prob.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        y_prob /= row_sums
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", alpha=0.6)
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"client_{cid}_roc.svg"))
        plt.close(fig)
    except Exception as e:
        print(f"[warn] ROC plot failed for client {cid}: {e}")


# def _global_class_counts(clients, num_classes):
#     counts = np.zeros(num_classes, dtype=np.int64)
#     for data in clients.values():
#         if getattr(data, "y_train", None) is not None and len(data.y_train) > 0:
#             c = np.bincount(data.y_train, minlength=num_classes)
#             counts[:len(c)] += c
#     return counts

def _local_class_counts(y_train, num_classes: int) -> np.ndarray:
    # 1️ Extract label array properly
    if y_train is None or len(y_train) == 0:
        return np.ones(num_classes, dtype=np.int64)

    # Some clients may store dicts like {'labels': ..., 'X': ...} or similar
    if isinstance(y_train, dict):
        # Try common keys
        for key in ["labels", "y", "target", "targets"]:
            if key in y_train:
                y_train = y_train[key]
                break
        else:
            raise ValueError(f"Unsupported dict format for y_train: {y_train.keys()}")

    # 2️ Convert to flat numpy int64 array
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.detach().cpu().numpy()
    elif isinstance(y_train, (list, tuple)):
        # flatten lists of tensors or ints
        flat = []
        for item in y_train:
            if isinstance(item, torch.Tensor):
                flat.append(item.detach().cpu().numpy().ravel())
            elif isinstance(item, (list, tuple, np.ndarray)):
                flat.append(np.array(item).ravel())
            elif isinstance(item, (int, float)):
                flat.append(np.array([item], dtype=np.int64))
        if len(flat) == 0:
            return np.ones(num_classes, dtype=np.int64)
        y_train = np.concatenate(flat)
    else:
        y_train = np.array(y_train)

    # ensure numeric and 1D
    y_train = np.array(y_train, dtype=np.int64).ravel()

    # 3️ Count per class and stabilize zeros
    c = np.bincount(y_train, minlength=num_classes)
    c[c == 0] = 1
    return c


def _build_local_ldam(y_train, num_classes: int, cfg: dict, round_idx: int):
    total_rounds = cfg.get("num_rounds", 1)
    in_drw = (round_idx + 1) >= cfg.get("drw_start_round", total_rounds // 2)

    local_counts = _local_class_counts(y_train, num_classes)

    loss = LDAMLoss(
        local_counts.tolist(),
        max_m=cfg.get("ldam_max_m", 0.5),
        s=cfg.get("ldam_s", 30.0),
    )

    if in_drw:
        drw_w = compute_drw_weights(local_counts, beta=cfg.get("drw_beta", 0.9999))
        if hasattr(loss, "set_drw_weights"):
            loss.set_drw_weights(drw_w)

    phase = {"phase": "LocalLDAM+DRW" if in_drw else "LocalLDAM", "drw_active": in_drw}
    return loss, phase

# def _build_ldam(clients, num_classes, cfg, round_idx):
#     total_rounds = cfg["num_rounds"]
#     in_drw = (round_idx + 1) >= cfg.get("drw_start_round", total_rounds // 2)
#     global_counts = _global_class_counts(clients, num_classes)
#     loss = LDAMLoss(global_counts.tolist(),
#                     max_m=cfg.get("ldam_max_m", 0.5),
#                     s=cfg.get("ldam_s", 30.0))
#     if in_drw:
#         drw_w = compute_drw_weights(global_counts, beta=cfg.get("drw_beta", 0.9999))
#         if hasattr(loss, "set_drw_weights"):
#             loss.set_drw_weights(drw_w)
#     phase = {"phase": "LDAM+DRW" if in_drw else "LDAM", "drw_active": in_drw}
#     return loss, phase


def run_method(name: str, clients, model_ctor, num_classes: int, cfg: dict, out_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_ts = time.strftime("%Y%m%d-%H%M%S")
    method_dir = os.path.join(out_dir, start_ts, name)
    log_dir, model_dir, eval_dir, plot_dir = [os.path.join(method_dir, d) for d in ["logs","models","eval","plots"]]
    for d in (log_dir, model_dir, eval_dir, plot_dir): _ensure_dir(d)

    in_channels = len(cfg["features"])
    batch_size = cfg.get("batch_size", 64)
    local_epochs = cfg.get("local_epochs", 1)
    lr = cfg.get("lr", 1e-3)
    num_rounds = cfg.get("num_rounds", 5)

    # init global model
    gmodel = TinyHARWithNormalizedLogits(in_channels, num_classes) if name == "FedLDAM" else model_ctor()
    gmodel = gmodel.to(device)
    gweights = get_weights(gmodel)

    client_ids = list(clients.keys())
    per_client_logs = {cid: [] for cid in client_ids}

    for rnd in range(num_rounds):
        ckpts = []
        print(f"[round {rnd+1}/{num_rounds}]")

        for cid in client_ids:
            data = clients[cid]
            if data.X_train is None or len(data.X_train) == 0:
                print(f"[warn] skip {cid}: no train samples")
                continue

            model = TinyHARWithNormalizedLogits(in_channels, num_classes).to(device) if name == "FedLDAM" else model_ctor().to(device)
            set_weights(model, gweights)

            # choose loss
            if name == "FedLDAM":
                loss_fn, phase = _build_local_ldam(data.y_train, num_classes, cfg, rnd)
                criterion = lambda logits, targets: loss_fn(logits, targets)
                use_norm = True
            elif name == "FedFocal":
                gamma = cfg.get("focal_gamma", 2.0)
                criterion = lambda logits, targets: focal_loss(logits, targets, gamma=gamma)
                phase, use_norm = {"phase": "Focal"}, False
            elif name == "FedRatio":
                criterion = RatioLoss(
                    alpha=cfg.get("ratio_alpha", 1.0),
                    beta=cfg.get("ratio_beta", 0.0)
                )
                phase, use_norm = {"phase": "Ratio"}, False
            else:
                ce = nn.CrossEntropyLoss()
                criterion = lambda logits, targets: ce(logits, targets)
                phase, use_norm = {"phase": name}, False

            opt = optim.Adam(model.parameters(), lr=lr)
            met = train_local(model, opt, criterion, data.X_train, data.y_train,
                            batch_size, local_epochs, device, normalize_logits=use_norm)

            row = {
                "client": cid, "round": rnd+1,
                "phase": phase["phase"],
                "loss": met.get("loss"),
                "accuracy": met.get("accuracy", 0.0),
                "f1_macro": met.get("f1_macro", 0.0)
            }
            per_client_logs[cid].append(row)

            # append row immediately to CSV
            log_path = os.path.join(log_dir, f"metrics_client_{cid}.csv")
            df = pd.DataFrame([row])
            if os.path.exists(log_path):
                df.to_csv(log_path, mode="a", header=False, index=False)
            else:
                df.to_csv(log_path, mode="w", header=True, index=False)

            ckpts.append((get_weights(model), len(data.y_train)))

        if ckpts:
            gweights = fedavg(ckpts)
            set_weights(gmodel, gweights)

        torch.save(gmodel.state_dict(), os.path.join(model_dir, f"global_round_{rnd+1}.pt"))

    # final eval on test
    rows = []
    for cid in client_ids:
        data = clients[cid]
        y_true, y_pred, y_prob = evaluate(gmodel, data.X_test, data.y_test, batch_size, device, normalize_logits=(name=="FedLDAM"))
        null_idx = data.classes.index("NULL") if "NULL" in data.classes else None
        m = compute_metrics(y_true, y_pred, null_index=null_idx)
        rows.append({"client": cid, **m})
        _create_client_plots(cid, name, y_true, y_pred, y_prob, data.classes, num_classes, plot_dir)

    # save logs
    log_path = os.path.join(log_dir, f"metrics_client_{cid}.csv")
    df = pd.DataFrame(per_client_logs[cid])
    if os.path.exists(log_path):
        # append without header
        df.tail(1).to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.tail(1).to_csv(log_path, mode="w", header=True, index=False)

    pd.DataFrame(rows).to_csv(os.path.join(eval_dir, "summary.csv"), index=False)
    return method_dir
