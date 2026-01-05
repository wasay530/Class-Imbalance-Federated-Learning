# eval_clients.py
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, accuracy_score, f1_score
)

from fedbench.models.tinyhar import TinyHAR
from fedbench.losses.ldam import TinyHARWithNormalizedLogits
from fedbench.datasets.wear_dataset import load_wear_clients, WearClientData
from sklearn.metrics import classification_report

def compute_class_f1(y_true, y_pred, labels, save_path="class_f1_scores.csv"):
    """
    Compute F1-score for each class + overall summary.
    """
    report = classification_report(
        y_true, y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    # Print nicely
    print("\n=== Per-Class F1 Scores ===")
    for lbl in labels:
        f1 = report[lbl]["f1-score"]
        print(f"{lbl}: {f1:.4f}")

    print("\n=== Overall Summary ===")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_path)
    print(f"Saved class-wise F1 scores to {save_path}")

def load_model(model_path, num_classes, normalize_logits, device="cpu"):
    if normalize_logits:
        model = TinyHARWithNormalizedLogits(in_channels=12, num_classes=num_classes)
    else:
        model = TinyHAR(in_channels=12, num_classes=num_classes)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
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
        logits = model(Xb)
        probs = torch.softmax(logits, dim=1)

        y_true.append(yb.cpu().numpy())
        y_pred.append(torch.argmax(probs, dim=1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    return (np.concatenate(y_true),
            np.concatenate(y_pred),
            np.concatenate(y_prob))


def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    plt.xticks(rotation=45, ha="right")
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")


def plot_roc(y_true, y_prob, labels):
    num_classes = len(labels)
    y_true_bin = np.eye(num_classes)[y_true]

    plt.figure(figsize=(10, 8))
    for i, lbl in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lbl} (AUC={auc_val:.2f})")

    # Micro-average
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, "k--", label=f"Micro (AUC={auc_val:.2f})")

    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves", fontsize=14)
    # Put legend outside box
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("roc_curve.png", bbox_inches="tight")
    print("Saved roc_curve.png")

def plot_subject_f1(y_true, y_pred, subj_ids, save_path="subject_f1_scores.png"):
    # Convert subject IDs to numeric (assumes names like 'sbj_0', 'sbj_1', ...)
    subj_numeric = []
    for s in subj_ids:
        if isinstance(s, str):
            # Remove prefix and convert to int
            if s.startswith("sbj_"):
                subj_numeric.append(int(s.replace("sbj_", "")))
            else:
                subj_numeric.append(int(s))
        else:
            subj_numeric.append(int(s))
    subj_numeric = np.array(subj_numeric)

    # Ensure subjects are sorted
    subjects = np.array(sorted(np.unique(subj_numeric)))
    scores = []
    for subj in subjects:
        mask = subj_numeric == subj
        if mask.sum() == 0:
            scores.append(0.0)
            continue
        f1 = f1_score(y_true[mask], y_pred[mask], average="macro")
        scores.append(f1)

    scores = np.array(scores)
    mean_f1 = scores.mean()

    plt.figure(figsize=(12, 6))
    bars = plt.bar(subjects + 1, scores, color="skyblue", alpha=0.7)

    # Annotate bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{score:.3f}",
                 ha="center", va="bottom", fontsize=9)

    # Mean line
    plt.axhline(mean_f1, color="red", linestyle="--", label=f"Mean F1: {mean_f1:.3f}")

    plt.xlabel("Subject ID")
    plt.ylabel("F1 Score")
    plt.title("Final Test F1 Score by Subject")
    plt.xticks(subjects + 1, [str(s+1) for s in subjects])  # show 1,2,3,...
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "/home/users/sardara/LDAM/WEAR_DATASET/wear_dataset/inertial_50hz"
    features = [
        "right_arm_acc_x","right_arm_acc_y","right_arm_acc_z",
        "right_leg_acc_x","right_leg_acc_y","right_leg_acc_z",
        "left_leg_acc_x","left_leg_acc_y","left_leg_acc_z",
        "left_arm_acc_x","left_arm_acc_y","left_arm_acc_z"
    ]
    label_col = "label"
    window_size = 100
    window_stride = 50
    num_classes = 19  # 18 activities + NULL

    # Load clients
    clients = load_wear_clients(
        data_dir=data_dir,
        features=features,
        label_col=label_col,
        window_size=window_size,
        window_stride=window_stride
    )

    if not clients:
        raise SystemExit(f"No CSV files found under {data_dir}")
    any_client = next(iter(clients.values()))
    classes = any_client.classes

    # Load model
    model = load_model(args.model_path, num_classes, args.normalize_logits, device)

    # Gather global test set
    X_all, y_all, subj_ids = [], [], []
    for subj_id, client in clients.items():
        if len(client.X_test) == 0:
            continue
        X_all.append(client.X_test)
        y_all.append(client.y_test)
        subj_ids.extend([subj_id] * len(client.y_test))

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    subj_ids = np.array(subj_ids)

    # Evaluate
    y_true, y_pred, y_prob = evaluate(model, X_all, y_all, device=device)

    # Class-level F1 scores
    compute_class_f1(y_true, y_pred, classes)

    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"Overall Test Accuracy: {overall_acc:.4f}")

    # Per-subject accuracy
    subj_acc = {}
    for subj in np.unique(subj_ids):
        mask = subj_ids == subj
        subj_acc[subj] = accuracy_score(y_true[mask], y_pred[mask])
    print("Per-Subject Test Accuracy:")
    for k, v in subj_acc.items():
        print(f"  {k}: {v:.4f}")

    # Confusion matrix & ROC
    plot_confusion(y_true, y_pred, classes)
    plot_roc(y_true, y_prob, classes)
    plot_subject_f1(y_true, y_pred, subj_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .pt model checkpoint")
    parser.add_argument("--normalize_logits", action="store_true",
                        help="Use TinyHARWithNormalizedLogits (for LDAM models)")
    args = parser.parse_args()
    main(args)