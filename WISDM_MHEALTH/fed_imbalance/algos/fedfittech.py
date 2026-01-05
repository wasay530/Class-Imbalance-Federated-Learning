import os, time, numpy as np, torch, torch.nn as nn, torch.optim as optim
import pandas as pd
from collections import deque

from fed_imbalance.training.trainer import train_local, evaluate
from fed_imbalance.training.federated import get_weights, set_weights, fedavg
from fed_imbalance.datasets.wear_dataset import make_train_val_split
from fed_imbalance.utils.metrics import compute_metrics
from fed_imbalance.algos.common import _ensure_dir, _create_client_plots

class EarlyStopper:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = 0.0
        self.counter = 0
        self.scores = deque(maxlen=patience)

    def __call__(self, score: float) -> bool:
        self.scores.append(score)
        if len(self.scores) < self.patience:
            return False # Not enough scores to decide

        # Check if the score has stabilized
        avg_score = sum(self.scores) / self.patience
        if (self.scores[0] - avg_score) < self.min_delta:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.patience:
            return True # Stop
        
        if score > self.best_score:
            self.best_score = score

        return False

def run_fedfittech(clients, model_ctor, num_classes: int, cfg: dict, out_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_ts = time.strftime("%Y%m%d-%H%M%S")
    method_dir = os.path.join(out_dir, start_ts, "FedFitTech")
    log_dir, model_dir, eval_dir, plot_dir = [os.path.join(method_dir, d) for d in ["logs","models","eval","plots"]]
    for d in (log_dir, model_dir, eval_dir, plot_dir): _ensure_dir(d)

    in_channels = len(cfg["features"])
    batch_size = cfg.get("batch_size", 32)
    local_epochs = cfg.get("local_epochs", 1)
    lr = cfg.get("lr", 1e-3)
    num_rounds = cfg.get("num_rounds", 100)

    gmodel = model_ctor().to(device)
    gweights = get_weights(gmodel)

    client_ids = list(clients.keys())
    per_client_logs = {cid: [] for cid in client_ids}
    stoppers = {cid: EarlyStopper(patience=5, min_delta=0.01) for cid in client_ids}
    active_clients = set(client_ids)

    for rnd in range(num_rounds):
        if not active_clients:
            print("All clients stopped. Halting.")
            break

        ckpts = []
        print(f"[round {rnd+1}/{num_rounds}] Active clients: {len(active_clients)}/{len(client_ids)}")

        for cid in list(active_clients):
            data = clients[cid]
            if data.X_train is None or len(data.X_train) == 0:
                continue

            model = model_ctor().to(device)
            set_weights(model, gweights)

            X_train_client, y_train_client, X_val_client, y_val_client = make_train_val_split(data.X_train, data.y_train, val_pct_of_train=0.1)

            opt = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            met = train_local(model, opt, criterion, X_train_client, y_train_client, batch_size, local_epochs, device)

            # Evaluate on validation set for early stopping
            val_metrics = evaluate(model, X_val_client, y_val_client, batch_size, device)
            val_f1 = compute_metrics(val_metrics[0], val_metrics[1])["f1_macro"]

            row = {
                "client": cid, "round": rnd+1,
                "loss": met.get("loss"),
                "accuracy": met.get("accuracy", 0.0),
                "f1_macro": met.get("f1_macro", 0.0),
                "val_f1": val_f1
            }
            per_client_logs[cid].append(row)

            log_path = os.path.join(log_dir, f"metrics_client_{cid}.csv")
            df = pd.DataFrame([row])
            df.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

            if stoppers[cid](val_f1):
                print(f"Client {cid} stopping early at round {rnd+1}")
                active_clients.remove(cid)
            else:
                ckpts.append((get_weights(model), len(y_train_client)))

        if ckpts:
            gweights = fedavg(ckpts)
            set_weights(gmodel, gweights)

        torch.save(gmodel.state_dict(), os.path.join(model_dir, f"global_round_{rnd+1}.pt"))

    rows = []
    for cid in client_ids:
        data = clients[cid]
        y_true, y_pred, y_prob = evaluate(gmodel, data.X_test, data.y_test, batch_size, device)
        null_idx = data.classes.index("NULL") if "NULL" in data.classes else None
        m = compute_metrics(y_true, y_pred, null_index=null_idx)
        rows.append({"client": cid, **m})
        _create_client_plots(cid, "FedFitTech", y_true, y_pred, y_prob, data.classes, num_classes, plot_dir)

    pd.DataFrame(rows).to_csv(os.path.join(eval_dir, "summary.csv"), index=False)
    return method_dir