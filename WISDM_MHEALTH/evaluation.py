
import argparse
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, accuracy_score, f1_score,
    classification_report,
)

from fed_imbalance.models.tinyhar import TinyHAR
from fed_imbalance.losses.ldam import TinyHARWithNormalizedLogits
from fed_imbalance.datasets.wear_dataset import load_wear_clients
from fed_imbalance.datasets.wisdm_dataset import load_wisdm_clients
from fed_imbalance.datasets.mhealth_dataset import load_mhealth_clients

def _numeric_from_id(s):
    if isinstance(s, str):
        import re as _re
        m = _re.findall(r'\d+', s)
        if m:
            return int(m[-1])
        return abs(hash(s)) % (10**6)
    try:
        return int(s)
    except Exception:
        return 0

def _infer_in_channels(clients):
    for c in clients.values():
        for a in (c.X_test, c.X_train):
            if a is not None and hasattr(a, "shape") and a.size > 0 and len(a.shape) == 3:
                return int(a.shape[2])
    raise RuntimeError("Unable to infer in_channels from clients (no data found).")

def _stack_global_test(clients):
    X_all, y_all, subj_ids = [], [], []
    for sid, client in clients.items():
        if getattr(client, "X_test", None) is None or len(client.X_test) == 0:
            continue
        X_all.append(client.X_test)
        y_all.append(client.y_test)
        subj_ids.extend([sid] * len(client.y_test))
    if not X_all:
        raise RuntimeError("No test samples found across clients.")
    return np.concatenate(X_all), np.concatenate(y_all), np.array(subj_ids)

def load_model(model_path, num_classes, in_channels, normalize_logits, device="cpu"):
    if normalize_logits:
        model = TinyHARWithNormalizedLogits(in_channels=in_channels, num_classes=num_classes)
    else:
        model = TinyHAR(in_channels=in_channels, num_classes=num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

@torch.no_grad()
def evaluate(model, X, y, device="cpu", batch_size=64):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred, y_prob = [], [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        probs = torch.softmax(model(Xb), dim=1)
        y_true.append(yb.cpu().numpy())
        y_pred.append(torch.argmax(probs, dim=1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)

def compute_class_f1(y_true, y_pred, labels, save_path="class_f1_scores.csv"):
    report = classification_report(
        y_true, y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True, zero_division=0,
    )
    print("\n=== Per-Class F1 Scores ===")
    for lbl in labels:
        print(f"{lbl}: {report[lbl]['f1-score']:.4f}")
    print("\n=== Overall Summary ===")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    pd.DataFrame(report).transpose().to_csv(save_path)
    print(f"Saved class-wise F1 scores to {save_path}")

def plot_confusion(y_true, y_pred, labels, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    plt.xticks(rotation=45, ha="right")
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path); print(f"Saved {save_path}")

def plot_roc(y_true, y_prob, labels, save_path="roc_curve.png"):
    n = len(labels); y_true_bin = np.eye(n)[y_true]
    plt.figure(figsize=(10, 8))
    for i, lbl in enumerate(labels):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{lbl} (AUC={auc(fpr, tpr):.2f})")
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    from sklearn.metrics import auc
    plt.plot(fpr, tpr, "k--", label=f"Micro (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves", fontsize=14)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")

def plot_subject_f1(y_true, y_pred, subj_ids, save_path="subject_f1_scores.png"):
    nums = np.array([_numeric_from_id(s) for s in subj_ids])
    subjects = np.array(sorted(np.unique(nums)))
    scores = []
    for s in subjects:
        m = nums == s
        scores.append(f1_score(y_true[m], y_pred[m], average="macro") if m.sum() else 0.0)
    scores = np.array(scores); mean_f1 = scores.mean() if len(scores) else 0.0
    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(len(subjects))+1, scores, alpha=0.7)
    for b, sc in zip(bars, scores):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{sc:.3f}", ha="center", va="bottom", fontsize=9)
    plt.axhline(mean_f1, linestyle="--", label=f"Mean F1: {mean_f1:.3f}")
    plt.xlabel("Subject #"); plt.ylabel("F1 Score (macro)")
    plt.title("Final Test F1 Score by Subject")
    plt.xticks(np.arange(len(subjects))+1, [str(int(s)) for s in subjects])
    plt.ylim(0, 1.05); plt.legend(); plt.tight_layout(); plt.savefig(save_path)
    print(f"Saved {save_path}")

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = args.dataset.upper()
    if ds == "WISDM":
        clients = load_wisdm_clients(
            data_dir=args.data_dir,
            window_size=args.window_size, window_stride=args.window_stride,
            features=["x","y","z"], label_col="activity",
            test_pct=args.test_pct, val_pct_of_train=args.val_pct_of_train
        )
    elif ds == "MHEALTH":
        clients = load_mhealth_clients(
            data_dir=args.data_dir,
            window_size=args.window_size, window_stride=args.window_stride,
            features=args.features if args.features else None,
            label_col=args.label_col,
            test_pct=args.test_pct, val_pct_of_train=args.val_pct_of_train
        )
    else:
        clients = load_wear_clients(
            data_dir=args.data_dir, features=args.features, label_col=args.label_col,
            window_size=args.window_size, window_stride=args.window_stride,
            test_pct=args.test_pct, val_pct_of_train=args.val_pct_of_train
        )
    if not clients:
        raise SystemExit(f"No clients found under {args.data_dir}")
    any_client = next(iter(clients.values()))
    classes = list(any_client.classes)
    num_classes = len(classes)
    in_channels = _infer_in_channels(clients)
    model = load_model(args.model_path, num_classes, in_channels, args.normalize_logits, device)
    X_all, y_all, subj_ids = _stack_global_test(clients)
    y_true, y_pred, y_prob = evaluate(model, X_all, y_all, device=device, batch_size=args.batch_size)
    print(f"Overall Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Overall Test F1-macro: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("Per-Subject Test Accuracy:")
    uniq = sorted(list(set(subj_ids)), key=_numeric_from_id)
    for s in uniq:
        m = subj_ids == s
        print(f"  {s}: {accuracy_score(y_true[m], y_pred[m]):.4f}")
    prefix = ds
    compute_class_f1(y_true, y_pred, classes, save_path=f"{prefix}_class_f1_scores.csv")
    plot_confusion(y_true, y_pred, classes, save_path=f"{prefix}_confusion_matrix.png")
    plot_roc(y_true, y_prob, classes, save_path=f"{prefix}_roc_curve.png")
    plot_subject_f1(y_true, y_pred, subj_ids, save_path=f"{prefix}_subject_f1_scores.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="MHEALTH", choices=["MHEALTH","WISDM","WEAR"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--window_stride", type=int, default=50)
    p.add_argument("--test_pct", type=float, default=0.2)
    p.add_argument("--val_pct_of_train", type=float, default=0.1)
    p.add_argument("--normalize_logits", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--features", nargs="+", default=[
        "right_arm_acc_x","right_arm_acc_y","right_arm_acc_z",
        "right_leg_acc_x","right_leg_acc_y","right_leg_acc_z",
        "left_leg_acc_x","left_leg_acc_y","left_leg_acc_z",
        "left_arm_acc_x","left_arm_acc_y","left_arm_acc_z"
    ])
    p.add_argument("--label_col", type=str, default="label")
    main(p.parse_args())
