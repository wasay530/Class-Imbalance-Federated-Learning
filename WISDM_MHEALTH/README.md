# Federated HAR for Class Imbalance Problem: WISDM & MHEALTH

This code follows your LDAM-style code structure but runs **five techniques** on the same data/splits/model/hparams from the FedFitTech baseline:

1. **FedAvg**
2. **FedFitTech** (baseline settings for FedFitTech)
3. **FedLDAM** (federated LDAM loss)
4. **FedRatio** (Ratio Loss in FL)
5. **FedFocal** (Fed-Focal Loss)

> TinyHAR is used as the model (as in FedFitTech).

## Methods (Baseline Studies)

1. **Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss** (https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf)
2. **FedFitTech: A Baseline in Federated Learning for Fitness Tracking** (https://www.arxiv.org/pdf/2506.16840)
3. **Fed-Focal Loss for imbalanced data classification in FL** (https://arxiv.org/pdf/2011.06283)
4. **Addressing Class Imbalance in Federated Learning (Fed Ratio)** (https://ojs.aaai.org/index.php/AAAI/article/view/17219)

## Quick Start

```bash
# (optional) create venv & install
pip install -r requirements.txt

# run all methods specified in config.yaml
python run_all.py --config config.yaml
```

## Data

We do a split per subject into **train/test** (80/20 by timestamp order). If your files already contain a `timestamp` column, it's used for ordering; otherwise file order is used.

## Notes

- FedFitTech settings are mirrored (TinyHAR, sensible LR/WD/BS); adjust in `config.yaml`.
- LDAM uses label-count adapted margins; Focal uses `gamma`; Ratio loss uses α/β.
