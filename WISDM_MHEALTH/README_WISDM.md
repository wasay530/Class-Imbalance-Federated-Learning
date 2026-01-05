
# WISDM Support (Drop-in)

This repository now supports the **WISDM v1.1** activity dataset in addition to the WEAR dataset, while keeping the exact same code structure, models, and hyperparameters.

## Quick start

1. Download `WISDM_ar_v1.1_raw.txt` from the WISDM site and note its path.
2. Edit `config_wisdm.yaml` and set:
   ```yaml
   dataset: WISDM
   data_dir: /absolute/path/to/WISDM_ar_v1.1_raw.txt
   ```
   All other settings (window size/stride, optimizer, rounds, etc.) are unchanged.
3. Run:
   ```bash
   python run_all.py --config config_wisdm.yaml
   ```

## Notes

* Per-user **clients** are created from the `user` field (1..36).
* We use **sliding windows** (size/stride from config) over the raw 20Hz accelerometer `x,y,z` signals, assigning each window a **majority** label.
* We perform a **chronological per-class split**: the earliest `test_pct` of windows for each class are assigned to **TEST**, the rest to **TRAIN**, matching the FedFitTech pre-window spirit.
* Standardization is done **per-client**, using TRAIN statistics, then applied to TEST.
* The label mapping to integers is **global** across users to keep class indices consistent.

Everything else (models, losses, training loops, evaluation) remains the same.
