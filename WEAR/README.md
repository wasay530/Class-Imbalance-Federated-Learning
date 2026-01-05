# Federated HAR: Multi-Method Runner (TinyHAR on WEAR)

This package follows your LDAM-style code structure but runs **six techniques in one command** on the same data/splits/model/hparams from the FedFitTech baseline:

1. **FedAvg**
2. **FedFitTech** (baseline settings with TinyHAR)
3. **FedLDAM** (federated LDAM loss locally)
4. **FedRAMP** (staged/feedback reweighting + Balanced Softmax + LDAM margins-style idea)
5. **FedRatio** (AAAI'21 Ratio Loss in FL)
6. **FedFocal** (Fed-Focal Loss)

**No validation**: only **train** and **test**. For every method, we:
- log **per-client, per-epoch**: loss, accuracy, F1 (macro)
- save **global model** checkpoints
- **evaluate on test** after training
- save **per-client** plots: accuracy/F1 bar chart, **ROC curves** (one-vs-rest), and **confusion matrices**

> TinyHAR is used as the model (as in FedFitTech).

## Quick Start

```bash
# (optional) create venv & install
pip install -r requirements.txt

# run all methods specified in config.yaml
python run_all.py --config config.yaml
```

## Data

Place your WEAR dataset under `./data/WEAR`. Expected structure (one csv per subject is fine):

```
data/WEAR/
  subject_00.csv
  subject_01.csv
  ...
```

Each CSV must contain sensor feature columns (default in `config.yaml`) and a `label` column (string or int).

We do a **chronological** split per subject into **train/test** (80/20 by timestamp order). If your files already contain a `timestamp` column, it's used for ordering; otherwise file order is used.

## Outputs

Everything goes to `./experiments/<timestamp>/<MethodName>/`:
- `logs/metrics_client_<id>.csv` – per-epoch metrics for each client
- `models/global_last.pt` – final global model
- `eval/test_metrics_global.csv` – aggregated global test
- `plots/client_<id>_bar.svg`, `client_<id>_roc.svg`, `client_<id>_cm.svg`

## Notes

- This is a **simple federated loop** implemented without Flower simulation to reduce dependencies.
- FedFitTech settings are mirrored (TinyHAR, sensible LR/WD/BS); adjust in `config.yaml`.
- LDAM uses label-count adapted margins; Focal uses `gamma`; Ratio loss uses α/β.
- If you don't have WEAR locally yet, you can stub with a few CSVs using the same column names to verify the pipeline.
