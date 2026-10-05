# Adversarial Robustness of Swin Transformers for Face Spoofing Detection

## Overview

This project studies the adversarial robustness of a **Swin-Tiny Transformer** for **Face Anti-Spoofing (FAS)** and proposes an **Adversarial Training (AT)** strategy to improve its resistance to adversarial attacks.

Face Anti-Spoofing is a critical security component of facial biometric authentication systems. The objective is to distinguish **real facial presentations** from **spoofing attempts**, such as printed photographs, replayed videos displayed on screens, or 3D silicone masks.

Although transformer-based vision architectures achieve strong classification performance, they can remain highly vulnerable to adversarial perturbations. This project therefore evaluates the robustness of several Swin-Tiny fine-tuning strategies against two white-box attacks:

- **FGSM (Fast Gradient Sign Method)**
- **PGD (Projected Gradient Descent)**

The experiments show a strong gap between standard accuracy and adversarial robustness. The best baseline reaches **98.60% accuracy on clean data**, but its accuracy falls to only **4.09% under PGD-40**. An adversarial training procedure is then applied to this model and raises the PGD-40 accuracy to **93.89%**, while preserving **98.03% clean accuracy**.

---

## Objectives

The project has two main objectives:

1. **Measure the adversarial vulnerability** of five Swin-Tiny transfer-learning strategies against FGSM and PGD white-box attacks.
2. **Improve robustness** through adversarial training while minimizing the loss of natural performance.

A third experimental objective is to study the role of **data augmentation** in the final robustness of the model.

---

## Dataset

The experiments use the **LCC-FASD** dataset, composed of real facial images and face spoofing attempts acquired in a controlled environment.

The official dataset split is respected, with **10% of the training set used for validation**.

### Class imbalance

The dataset presents a significant imbalance between spoof and real samples:

| Split | Spoof / Real ratio |
|---|---:|
| Training | 5.78x |
| Evaluation | 23.14x |

To handle this imbalance, class weighting is used in the loss function during the baseline training phase:

- `w_spoof ≈ 0.2947`
- `w_real ≈ 1.7053`

### Image preprocessing

All images are resized to:

```text
224 × 224 pixels
```

Images are normalized using ImageNet statistics:

```text
μ = [0.485, 0.456, 0.406]
σ = [0.229, 0.224, 0.225]
```

### Data augmentation

The training pipeline includes:

- Random horizontal flip: `p = 0.5`
- Random rotation: `±5°`
- Moderate ColorJitter:
  - Brightness: `±0.2`
  - Contrast: `±0.2`
  - Saturation: `±0.1`

A separate experiment removes data augmentation in order to measure its contribution to robustness.

---

## Model

The project uses **Swin-Tiny**, a hierarchical Vision Transformer, pretrained on **ImageNet-1K** through the `timm` library.

Five transfer-learning strategies are evaluated over **10 epochs**.

| Strategy | Trainable configuration | Trainable parameters |
|---|---|---:|
| `freeze_all` | Classification head only | 0.006% |
| `fine_tune_1` | Layer 4 + normalization + head | 55.8% |
| `fine_tune_2` | Layers 3–4 + normalization + head | 95.7% |
| `fine_tune_3` | Layers 2–4 + normalization + head | 99.2% |
| `fine_tune_all` | Entire network | 100% |

### Baseline optimization

The baseline models are trained using:

- **Optimizer:** AdamW
- **Learning rate:** `2 × 10⁻⁴`
- **Weight decay:** `1 × 10⁻⁴`
- **Scheduler:** CosineAnnealingWarmRestarts
- **Warm-restart parameter:** `T₀ = 5`
- **AMP:** enabled
- **Gradient clipping:** maximum norm `1.0`
- **Loss:** weighted Cross-Entropy
- **Label smoothing:** `ε = 0.1`

---

## Adversarial Attacks

Robustness is evaluated using two **white-box adversarial attacks**.

### FGSM

The Fast Gradient Sign Method generates a single perturbation step:

$$
x_{adv} = x + \varepsilon \cdot \\mathrm{sign}\left(\\nabla_x \\mathcal{L}(f(x), y)\\right)
$$

with:

```text
ε = 0.03
```

### PGD

Projected Gradient Descent extends this idea iteratively and starts from a random point inside the `ℓ∞` perturbation ball:

$$
x^{(t+1)} = \Pi_{x,\varepsilon}\left(
x^{(t)} + \alpha \\cdot \\mathrm{sign}\left(\\nabla_{x^{(t)}} \\mathcal{L}(f(x^{(t)}), y)\right)\
\right)
$$

Attack configuration:

```text
ε = 0.03
α = 0.008
iterations = 40
```

PGD-40 is the strongest attack considered in the experiments and produces the most severe degradation of the baseline model.

---

## Adversarial Training Defense

The best baseline model, `fine_tune_3`, is selected for adversarial training.

Adversarial examples are generated **on the fly during training** using PGD with:

```text
steps = 7
α = ε / 4
```

The total loss is defined as an equal combination of the clean and adversarial losses:

$$
\mathcal{L}_{total} =
\frac{\mathcal{L}(f(x),y) +
\mathcal{L}(f(x_{adv}),y)}{2}
$$

The robust model is then fine-tuned for:

```text
5 additional epochs
learning rate = 5 × 10⁻⁵
scheduler = cosine
```

Class weighting is disabled during this phase in order to stabilize optimization on adversarial examples.

---

## Results

### 1. Transfer-learning baseline comparison

All five strategies achieve more than **93% clean accuracy**. The best baseline is `fine_tune_3`.

| Strategy | Accuracy | F1 | FGSM | PGD-40 |
|---|---:|---:|---:|---:|
| `freeze_all` | 93.4% | 94.8% | --- | --- |
| `fine_tune_1` | 98.0% | 98.02% | --- | --- |
| `fine_tune_2` | 97.8% | 97.93% | --- | --- |
| **`fine_tune_3`** | **98.6%** | **98.60%** | **83.89%** | **4.09%** |
| `fine_tune_all` | 98.1% | 98.12% | --- | --- |

For `fine_tune_3`:

- Clean accuracy: **98.60%**
- Clean F1: **98.60%**
- Clean AUC: **0.947**
- FGSM accuracy: **83.89%**
- PGD-40 accuracy: **4.09%**

Under PGD-40, the model is almost completely compromised despite its excellent clean-data performance.

---

### 2. Impact of adversarial training

The following comparison shows the effect of adversarial training on `fine_tune_3`.

| Condition | Model | Accuracy | Precision | F1 | AUC |
|---|---|---:|---:|---:|---:|
| Clean | Baseline | 98.60% | 98.76% | 98.60% | 0.947 |
| Clean | Robust | 98.03% | 98.18% | 98.09% | 0.955 |
| FGSM | Baseline | 83.89% | 92.76% | 87.85% | 0.404 |
| FGSM | Robust | **95.86%** | 91.89% | **93.83%** | 0.773 |
| PGD-40 | Baseline | 4.09% | 47.62% | 7.34% | 0.003 |
| PGD-40 | Robust | **93.89%** | 91.81% | **92.84%** | 0.001 |

### Main gains

Under FGSM:

```text
Accuracy: 83.89% → 95.86%   (+11.97 points)
F1:       87.85% → 93.83%   (+5.98 points)
```

Under PGD-40:

```text
Accuracy: 4.09% → 93.89%   (+89.80 points)
F1:       7.34% → 92.84%  (+85.50 points)
```

The clean accuracy decreases only slightly:

```text
98.60% → 98.03%
Change = -0.57 point
```

This shows that the proposed defense provides a substantial robustness improvement while preserving most of the natural classification performance.

---

## Confusion Matrix Analysis

On clean data, the baseline model correctly classifies:

- **7,236 spoof images (99.6%)**
- **239 real images (76.1%)**

After adversarial training:

- **7,170 spoof images (98.7%)**
- **261 real images (83.1%)**

The number of false-positive spoof predictions increases from `30` to `96`. This is interpreted as a smoothing of the decision boundaries induced by adversarial training, while remaining at an operationally acceptable level according to the study.

---

## Impact of Data Augmentation

A parallel experiment was performed without data augmentation.

| Metric | Baseline (FT-3) | Robust (AT) |
|---|---:|---:|
| Clean Accuracy | 99.47% | 98.73% |
| Clean F1-Score | 99.46% | 98.75% |
| FGSM Accuracy | 26.39% | 89.67% |
| PGD Accuracy | 0.00% | 86.43% |

Without augmentation, the baseline reaches a higher clean accuracy (**99.47%**) but becomes extremely vulnerable to adversarial attacks:

- FGSM accuracy: **26.39%**
- PGD accuracy: **0.00%**

Adversarial training partially restores robustness:

- FGSM: **89.67%**
- PGD: **86.43%**

However, these values remain below the results obtained with augmentation, especially under PGD (**93.89%** with augmentation).

The experiment therefore highlights a strong synergy between **data augmentation and adversarial training**.

---

## Training Curves

During the baseline training of `fine_tune_3`:

- Loss decreases from approximately **0.42 to 0.33** over 10 epochs.
- A slight increase around epoch 4 is associated with the cosine scheduler restarts.
- Validation metrics converge to values above **0.97** from epoch 2.

During adversarial training:

- PGD-7 adversarial training accuracy increases from **82.82% to 85.26%**.

The paper includes the following visualizations:

```text
fig1_strategies.png
fig2_training_curves.png
fig3_robustesse.png
fig4_confusion.png
```

---

## Key Findings

### 1. High clean accuracy does not imply robustness

The best baseline reaches **98.60% clean accuracy**, but only **4.09% accuracy under PGD-40**. Standard accuracy alone is therefore insufficient to validate a secure biometric model.

### 2. PGD is highly effective against the baseline

The iterative PGD attack produces a much stronger degradation than FGSM, reducing the baseline to near-random performance.

### 3. Adversarial training is highly effective

Adversarial training increases PGD-40 accuracy from **4.09% to 93.89%**, representing an improvement of **89.80 percentage points**.

### 4. The cost on clean performance is small

Clean accuracy decreases by only **0.57 percentage points**, showing a favorable robustness/accuracy trade-off in the reported experiments.

### 5. Data augmentation contributes to robustness

Removing augmentation dramatically weakens the final robust model, reducing PGD accuracy from **93.89% to 86.43%**.

### 6. Accuracy and AUC can diverge under attack

An important observation is that the robust model reaches **93.89% accuracy under PGD-40**, while its reported AUC remains very low (**0.001**). This suggests that correct binary decisions can coexist with highly perturbed output probabilities and points to probability calibration as a separate robustness dimension.

---

## Limitations

The study identifies several limitations:

1. Adversarial training is limited by the available **Kaggle computational resources**, with only 5 additional epochs.
2. The evaluation protocol focuses on **white-box attacks**.
3. The equal weighting of clean and adversarial losses was not optimized.

---

## Future Work

The paper identifies three main directions for future work:

- Evaluation against **black-box adversarial attacks**.
- Study of **transferability to other Transformer architectures**.
- Investigation of **adaptively weighted loss functions**.

---

## Experimental Pipeline

The complete experimental workflow can be summarized as follows:

```text
LCC-FASD Dataset
       │
       ▼
Image Preprocessing
224×224 + ImageNet Normalization
       │
       ▼
Data Augmentation
Flip + Rotation + ColorJitter
       │
       ▼
Swin-Tiny (ImageNet-1K pretrained)
       │
       ├── freeze_all
       ├── fine_tune_1
       ├── fine_tune_2
       ├── fine_tune_3  ◄── Best baseline
       └── fine_tune_all
                    │
                    ▼
          Clean Evaluation
                    │
                    ▼
             FGSM / PGD-40
                    │
                    ▼
          Robustness Analysis
                    │
                    ▼
        Adversarial Training
        PGD-7 + Clean Loss
                    │
                    ▼
          Robust Swin-Tiny
                    │
                    ▼
       Clean / FGSM / PGD-40
          Final Evaluation
```

---

## Technologies and Methods

The project is based on the following techniques and components:

- **Python / Deep Learning**
- **PyTorch-based training workflow**
- **Swin-Tiny Transformer**
- **timm** for the ImageNet-1K pretrained architecture
- **LCC-FASD** for Face Anti-Spoofing evaluation
- **FGSM** adversarial attack
- **PGD** adversarial attack
- **Adversarial Training**
- **AdamW** optimization
- **CosineAnnealingWarmRestarts** / cosine scheduling
- **AMP (Automatic Mixed Precision)**
- **Gradient clipping**
- **Data augmentation**

---

## Repository Content

The paper explicitly references the following result figures:

```text
fig1_strategies.png
fig2_training_curves.png
fig3_robustesse.png
fig4_confusion.png
```

The exact software installation instructions, dataset download procedure, training commands, and repository directory structure are **not specified in the article**. Those elements should therefore be documented separately from the experimental paper if they are available in the project source code.

---

## References

1. Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Y. Zhang, S. Lin, and B. Guo, **“Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,”** Proc. IEEE/CVF ICCV, 2021, pp. 10012–10022.
2. I. J. Goodfellow, J. Shlens, and C. Szegedy, **“Explaining and Harnessing Adversarial Examples,”** ICLR, 2015.
3. A. Madry, A. Makelov, L. Schmidt, T. Tsipras, and A. Vladu, **“Towards Deep Learning Models Resistant to Adversarial Attacks,”** ICLR, 2018.
4. Y. Liu, J. Stehouwer, A. Jourabloo, and X. Liu, **“Deep Tree Learning for Zero-shot Face Anti-Spoofing,”** Proc. IEEE/CVF CVPR, 2019, pp. 4680–4689.
5. M. Buda, A. Maki, and M. A. Mazurowski, **“A Systematic Study of the Class Imbalance Problem in Convolutional Neural Networks,”** Neural Networks, vol. 106, pp. 249–259, 2018.

---

## Authors

- **Badereddine El Alouani**
- **Yassine Bouachrine**
- **Mohamed Khacha**
- **Ayoub Idlhaj**

**École Nationale Supérieure de l'Intelligence Artificielle et Sciences des Données (ENSIASD)**  
Taroudant, Morocco

---

## Acknowledgments

The authors thank **Professor Basma Saad** for her supervision within the **“Computer Vision & Generative AI”** module.
