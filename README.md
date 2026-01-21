# FedLDAM: Federated Label-Distribution-Aware Margin Loss for HAR

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)]()
[![Topic](https://img.shields.io/badge/Topic-Federated%20Learning-green)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

## Abstract
This repository contains the official implementation of **FedLDAM**, a novel framework designed to address the severe class imbalance problem in **Federated Learning (FL)** for **Human Activity Recognition (HAR)**.

In real-world healthcare scenarios, data is naturally imbalanced (e.g., "walking" is frequent, while "falling" is rare). Standard FL algorithms bias the global model toward majority classes. **FedLDAM** extends the Label-Distribution-Aware Margin (LDAM) loss to the federated setting, allowing clients to compute adaptive class-specific margins to rebalance minority categories without sharing raw private data.

<p align="center">
<img src="https://github.com/wasay530/Class-Imbalance-Federated-Learning/blob/29ebd5a375aa3a71b40dffb50440d83a201133c1/FedLDAM-Framework.png" title="Federated Label-Distribution-Aware Margin Framework" width="100%" alt="FedLDAM Framework">
</p>

## Key Contributions
* **Federated LDAM Formulation:** Adaptation of margin-based loss for distributed, non-IID environments.
* **Adaptive Decision Boundaries:** Dynamic margin adjustment $\Delta_y$ pushes decision boundaries toward majority classes, improving minority class generalization.
* **Privacy-Preserving:** Class statistics are aggregated or used locally to compute margins; **no raw sensor data leaves the client device.**

## Methodology: The FedLDAM Mechanism
The core innovation is the adaptation of the LDAM loss function for local client updates. Unlike standard Cross-Entropy which treats all classes equally, FedLDAM enforces a class-dependent margin.

For a client $k$ with local dataset $\mathcal{D}_k$, the local objective function is:

$$\mathcal{L}_{LDAM}^{(k)} = \frac{1}{|\mathcal{D}_k|}\sum_{(x_i, y_i) \in \mathcal{D}_k} \text{CE}\big(f_{y_i}(x_i) - \Delta_{y_i,k}, y_i\big)$$

Where:
* $f_{y_i}(x_i)$: Model’s output logit for the ground-truth class $y_i$.
* $\Delta_{y_i,k}$: Class-specific margin based on the inverse frequency of the class in $\mathcal{D}_k$.
* $\text{CE}$: Cross-Entropy loss.

**Intuition:** By subtracting the margin $\Delta$ from the true class logit, the model is forced to map inputs "deeper" into the correct class region, effectively handling sensor noise and class overlap.

## Benchmark Datasets
We evaluate the framework on three standard HAR datasets representing real-world heterogeneity:

| Dataset | Classes | Subjects | Imbalance Ratio | Description |
|:-------:|:-------:|:--------:|:---------------:|-------------|
| **MHEALTH** | 13 | 10 | 84.1:1 | High imbalance; physiological monitoring. |
| **WEAR** | 19 | 24 | 15.2:1 | Complex heterogeneity; varying sensor positions. |
| **WISDM** | 6 | 29 | 8.7:1 | Standard baseline for activity recognition. |

## Comparison Methods
This study benchmarks **FedLDAM** against state-of-the-art FL strategies:

1. **FedAvg:** Standard baseline; aggregates weights based on dataset size, ignoring imbalance.
2. **FedFitTech:** Fitness-aware aggregation optimizing for wearable constraints.
3. **FedFocal:** Integrates Focal Loss to dynamically scale loss for "hard" examples.
4. **FedRatio:** Ratio-based reweighting based on estimated global class statistics.
5. **FedLDAM (Ours):** Proposed margin-based regularization.

## Citation
`Cao, Kaidi, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. "Learning imbalanced datasets with label-distribution-aware margin loss." Advances in neural information processing systems 32 (2019).`


## License
This project is licensed under the MIT License.
