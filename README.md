# FedLDAM: Federated Label-Distribution-Aware Margin Loss for Class-Imbalanced Human Activity Recognition
## Overview
This repository contains the official implementation of **FedLDAM**, a novel framework designed to address the severe class imbalance problem in Federated Learning (FL) for Human Activity Recognition (HAR).

In real-world HAR scenarios, data is often naturally imbalanced (e.g., "walking" is far more common than "falling"). Standard FL algorithms (like FedAvg) struggle with this, often biasing the global model toward majority classes.

**Our Solution:** We extend the **Label-Distribution-Aware Margin (LDAM) loss** from centralized learning to the federated setting. FedLDAM allows clients to utilize local class statistics to compute class-specific margins that adaptively rebalance minority categories without sharing raw data.

<p align="center">
<img src = "https://github.com/wasay530/Class-Imbalance-Federated-Learning/blob/29ebd5a375aa3a71b40dffb50440d83a201133c1/FedLDAM-Framework.png" title= "Federated Label-Distribution-Aware Margin
(FedLDAM) Framework" width="800" height="400" alt>
</p>
<p align="center">
  <em>Figure 1: Federated Label-Distribution-Aware Margin (FedLDAM) Framework</em>  
</p>

## Key Contributions
* **Federated LDAM Formulation:** We adapt the LDAM loss function for distributed environments. The loss incorporates a margin $\Delta_y$ for class $y$, which is inversely proportional to the class frequency.
* **Adaptive Margin Decision Boundaries:** By enlarging the margins for minority classes, the decision boundary is pushed towards majority classes, effectively countering the bias inherent in imbalanced sensor data.
* **Privacy-Preserving Rebalancing:** Class statistics are aggregated or used locally to compute margins, ensuring user activity data remains private.

**The FedLDAM Mechanism**
The FedLDAM MechanismThe core innovation of this framework is the adaptation of the LDAM loss function for local client updates. Unlike standard Cross-Entropy which treats all classes equally, FedLDAM enforces a class-dependent margin to the decision boundary.

For a client $k$ with local dataset $\mathcal{D}_k$, the local objective function is formulated as:

$$\mathcal{L}_{LDAM}^{(k)} = \frac{1}{|\mathcal{D}_k|}\sum_{(x_i, y_i) \in \mathcal{D}_k} \text{CE}\big(f_{y_i}(x_i) - \Delta_{y_i,k}, y_i\big)$$

Where:

* $f_{y_i}(x_i)$ represents the model’s output logit for the ground-truth class $y_i$.
* $\Delta_{y_i,k}$ is the class-specific margin for class $y_i$ on client $k$, computed based on the inverse frequency of the class in $\mathcal{D}_k$.
* $\text{CE}$ denotes the Cross-Entropy loss.

**How it works:** Each client optimizes this objective locally. By subtracting the margin $\Delta$ from the true class logit, the model is forced to learn features that map the input "deeper" into the correct class region. This ensures that the margin adjustment is tailored to individual client data distributions, effectively pushing the decision boundary away from minority classes to improve generalization.

## Benchmark Datasets
We evaluate the framework on three standard HAR datasets with varying degrees of imbalance:

Dataset	| Classes	| Subjects	| Imbalance Ratio
-----------| ---------| ---------| ---------
MHEALTH	| 13	| 10 | 84.1:1
WEAR	| 19	| 24 | 15.2:1
WISDM	| 6	| 29 | 8.7:1

## Comparison Methods
This study benchmarks **FedLDAM** against four state-of-the-art baselines:
1. **FedAvg:** The standard FL baseline. It aggregates weights based on dataset size but ignores class imbalance, often leading to poor minority class performance.
2. **FedFitTech:** A fitness-aware aggregation strategy designed to optimize for wearable constraints and heterogeneity.
3. **FedFocal:** Integrates Focal Loss into the FL framework. It dynamically scales the loss to focus training on "hard" (often minority) examples.
4. **FedRatio:** Applies a ratio-based reweighting to the loss function based on estimated global class statistics.
5. **FedLDAM (Ours):** Our proposed margin-based method that actively regularizes the decision boundary locally.

## Citation
`Cao, Kaidi, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. "Learning imbalanced datasets with label-distribution-aware margin loss." Advances in neural information processing systems 32 (2019).`


## License
This project is licensed under the MIT License.
