import argparse, yaml, os, time
from fedbench.datasets.wear_dataset import load_wear_clients
from fedbench.models.tinyhar import TinyHAR
from fedbench.algos.common import run_method
from fedbench.algos.fedfittech import run_fedfittech
from fedbench.utils.seed import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    out_dir = cfg["out_dir"]
    data_dir = cfg["data_dir"]
    features = cfg["features"]
    label_col = cfg["label_col"]
    window_size = cfg["window_size"]
    window_stride = cfg["window_stride"]
    methods = cfg["methods"]

    # load clients
    clients = load_wear_clients(data_dir, features, label_col, window_size, window_stride)
    if not clients:
        raise SystemExit(f"No CSV files found under {data_dir}")
    any_client = next(iter(clients.values()))
    in_channels = len(features)
    num_classes_dataset = len(any_client.classes)
    if "num_classes" in cfg and cfg["num_classes"] != num_classes_dataset:
        # print(f"[warn] config.num_classes={cfg["num_classes"]} ")
        print(f"!= dataset={num_classes_dataset}; using dataset value.")
    num_classes = num_classes_dataset

    def model_ctor():
        return TinyHAR(in_channels, num_classes)

    ran = []
    for m in methods:
        if m == "FedFitTech":
            d = run_fedfittech(clients, model_ctor, num_classes, cfg, out_dir)
        else:
            d = run_method(m, clients, model_ctor, num_classes, cfg, out_dir)

        print(f"Completed {m}: outputs at {d}")
        ran.append((m,d))
    print("All done.")
    for m,d in ran:
        print(m, "→", d)

if __name__ == "__main__":
    main()