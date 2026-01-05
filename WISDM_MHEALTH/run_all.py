
import argparse, yaml, os
from fed_imbalance.datasets.wear_dataset import load_wear_clients
from fed_imbalance.datasets.wisdm_dataset import load_wisdm_clients
from fed_imbalance.datasets.mhealth_dataset import load_mhealth_clients
from fed_imbalance.models.tinyhar import TinyHAR
from fed_imbalance.algos.common import run_method
from fed_imbalance.algos.fedfittech import run_fedfittech
from fed_imbalance.utils.seed import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    dataset = cfg.get("dataset", "WEAR").upper()
    data_dir = cfg["data_dir"]
    features = cfg.get("features", None)
    label_col = cfg.get("label_col", "label")
    window_size = cfg["window_size"]
    window_stride = cfg["window_stride"]
    methods = cfg["methods"]
    out_dir = cfg["out_dir"]
    num_classes = cfg["num_classes"]

    if dataset == "WISDM":
        clients = load_wisdm_clients(
            data_dir=data_dir,
            window_size=window_size,
            window_stride=window_stride,
            features=features,              # ignored by loader
            label_col="activity",
            test_pct=cfg.get("test_pct", 0.2),
            val_pct_of_train=cfg.get("val_pct_of_train", 0.1)
        )
    elif dataset == "MHEALTH":
        # If a CSV is used, let loader infer the feature columns (do not force defaults)
        if features is None and data_dir.lower().endswith(".csv"):
            pass  # let loader infer
        clients = load_mhealth_clients(
            data_dir=data_dir,
            window_size=window_size,
            window_stride=window_stride,
            features=features,
            label_col=label_col,
            test_pct=cfg.get("test_pct", 0.2),
            val_pct_of_train=cfg.get("val_pct_of_train", 0.1)
        )
    else:
        # WEAR: features must be explicit in config
        clients = load_wear_clients(
            data_dir=data_dir,
            features=features,
            label_col=label_col,
            window_size=window_size,
            window_stride=window_stride,
            test_pct=cfg.get("test_pct", 0.2),
            val_pct_of_train=cfg.get("val_pct_of_train", 0.1)
        )

    # Model constructor (kept parity)
    in_ch = next(iter(clients.values())).X_train.shape[2]
    model_ctor = lambda: TinyHAR(in_channels=in_ch, num_classes=num_classes)

    ran = []
    for m in methods:
        if m == "FedFitTech":
            d = run_fedfittech(clients, model_ctor, num_classes, cfg, out_dir)
        else:
            d = run_method(m, clients, model_ctor, num_classes, cfg, out_dir)
        print(f"Completed {m}: outputs at {d}")
        ran.append((m, d))

    print("All done.")
    for m, d in ran:
        print(m, "-", d)

if __name__ == "__main__":
    main()
