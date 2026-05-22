# 📰 Media Bias Detection Using Feature-Based Analysis

> A comprehensive research project for detecting political bias in news articles using handcrafted linguistic and stylistic features, machine learning classifiers, and fallacy-based feature augmentation.

---

## 🗂️ Repository Structure

```
├── 📁 standard dataset/
├── 📁 prop meth and res/
├── 📁 updated dataset with new feature and w.../
├── 📁 work on cocolofa dataset/
└── 📁 ReportsandPapers/
```

---

## 🔬 Research Overview

This project investigates **political bias detection in news articles** using a feature engineering pipeline. Rather than relying on black-box deep learning, we extract interpretable linguistic, stylistic, and rhetorical features from article text and feed them into classical machine learning classifiers — making the predictions explainable and auditable.

### Pipeline

```
Article Text → Feature Extraction → Feature Grouping → Feature Vector → ML Models → Bias Prediction
```

Each article is analyzed across **20+ handcrafted features** grouped into semantic categories, then classified as **Left**, **Right**, or **Center** leaning.

---

## 📁 Folder Details

### 🗃️ `standard dataset/`

Contains the core dataset of **~1,500 news articles** with ground-truth ideology labels (Left / Center / Right).

**What's inside:**
- Article content with ~20 extracted features per article
- Jupyter notebook: `article_bias_documented.ipynb` — full feature extraction and classification pipeline
- **Feature graphs** for all 20 features showing their distribution across articles
- **Box plots** for each feature broken down by ideology (Left, Center, Right) — used to visually assess how each feature behaves across political leanings and whether it has discriminative power

**Feature Groups used:**

| Symbol | Description |
|--------|-------------|
| `Fs` | Structural features (article length, paragraph count, etc.) |
| `Fl` | Lexical features (vocabulary richness, word complexity, etc.) |
| `Ft` | Tonal/sentiment features |
| `Fa` | Argumentative features |
| `Fp` | Punctuation / stylistic features |
| `Ff` | **Fallacy-based features** *(new — see below)* |

---

### 📊 `prop meth and res/`

Contains all **methodology diagrams** and **results visualizations** used in the paper and report.

**Methodology Diagrams:**

Two versions of the end-to-end pipeline diagram:

![Pipeline v1](prop%20meth%20and%20res/prop_meth_dia_1.png)
*Figure: Proposed Methodology — Feature-based bias detection pipeline (Version 1)*

![Pipeline v2](prop%20meth%20and%20res/prop_meth_diag_2.png)
*Figure: Proposed Methodology — Feature-based bias detection pipeline (Version 2)*

**Results:**

- **Table 1 — Group-based Feature F1-score (Article Dataset)**  
  Evaluates each individual feature group (Fs, Fl, Ft, Fa, Fp, Ff) in isolation across RF, SVM, LR, and XG-Boost.

- **Table 2 — Cumulative Feature Combination Results**  
  Evaluates cumulative feature stacking (Fs → Fs+Fl → Fs+Fl+Ft → ...) and the effect of PCA on the combined vector.

- **Table 3 — New Fallacy Feature Results**  
  Micro F1 scores comparing approaches with the updated dataset including the new fallacy-based feature (Ff).

---

### 🆕 `updated dataset with new feature and w.../`

This folder contains the extended work introducing **fallacy-based features (Ff)** — a novel contribution to the media bias detection task.

**What we did:**
- Identified rhetorical fallacies present in news articles (e.g., ad hominem, straw man, appeal to emotion, false dichotomy)
- Implemented automated detection of these fallacies as a new feature group `Ff`
- Measured their contribution to bias prediction when added to the existing feature vector
- Evaluated whether fallacy usage patterns differ significantly across Left, Center, and Right-leaning outlets

**Key Finding:** Fallacy-based features show measurable signal for bias detection, especially when combined with structural and lexical features.

---

### 🧪 `work on cocolofa dataset/`

Before deploying the new fallacy features on the main article dataset, we **validated them on COCOLOFA** — a publicly available, standardized dataset from Kaggle designed for logical fallacy classification.

**Why COCOLOFA?**
- Provides ground-truth labels for individual fallacy types
- Allows us to verify that our fallacy detection approach is accurate before applying it to bias detection
- Serves as an independent benchmark to confirm feature quality

**What's inside:**
- Notebooks implementing fallacy detection on COCOLOFA
- Validation metrics confirming detection accuracy
- Analysis linking fallacy types to political bias signals

---

### 📄 `ReportsandPapers/`

Contains all academic output from this project:

- 📝 **Research Paper** — draft manuscript written in the style of a conference/journal submission, covering motivation, related work, methodology, experiments, and conclusions
- 📋 **Project Reports** — intermediate and final reports documenting the full research process, including literature review, implementation details, and result analysis

---

## 📈 Key Results

### Group-based Feature F1-score (Article Dataset)

| Approach | RF | SVM | LR | XG-Boost | RF (Micro) | SVM (Micro) | LR (Micro) | XGB (Micro) |
|----------|-----|-----|-----|----------|------------|-------------|------------|-------------|
| Fs | 0.7764 | 0.8372 | 0.8372 | 0.7879 | 0.6467 | 0.72 | 0.72 | 0.6733 |
| Fl | 0.8145 | 0.8372 | 0.8099 | 0.7544 | 0.6933 | 0.72 | 0.6933 | 0.6267 |
| Ft | 0.7204 | 0.8372 | 0.8372 | 0.7431 | 0.6067 | 0.72 | 0.72 | 0.6267 |
| Fa | 0.8372 | 0.8372 | 0.8372 | 0.8372 | 0.72 | 0.72 | 0.72 | 0.72 |
| Fp | 0.7623 | 0.8372 | 0.8372 | 0.7636 | 0.6467 | 0.72 | 0.72 | 0.6533 |
| Ff | 0.7478 | 0.8327 | 0.8235 | 0.72 | 0.6133 | 0.7133 | 0.70 | 0.58 |

### Cumulative Feature Combination — Micro F1 Score

| Approach | RF | SVM | LR | XGB |
|----------|-----|-----|-----|-----|
| Fs | 0.64 | 0.71 | 0.71 | 0.6067 |
| Fs + Fl | 0.6767 | 0.71 | 0.6833 | 0.6767 |
| Fs + Fl + Ft | 0.70 | 0.71 | 0.6767 | 0.6467 |
| Fs + Fl + Ft + Fa | 0.69 | 0.71 | 0.67 | 0.65 |
| Fs + Fl + Ft + Fa + Fp | 0.71 | 0.71 | 0.68 | 0.6667 |
| Fs + Fl + Ft + Fa + Fp (with PCA) | 0.5467 | 0.71 | 0.71 | 0.58 |
| Fs + Fl + Ft + Fa + Fp + Ff | 0.7233 | 0.71 | 0.67 | 0.6667 |

> **Best overall performance:** SVM consistently achieves the highest F1-scores across feature combinations. Adding the new fallacy feature (Ff) improves RF performance to 0.7233 Micro F1.

---

## 🤖 Classifiers Used

| Classifier | Description |
|------------|-------------|
| **RF** | Random Forest |
| **SVM** | Support Vector Machine |
| **LR** | Logistic Regression |
| **XG-Boost** | Extreme Gradient Boosting |

---

## 🧠 Methodology

1. **Feature Extraction** — Each article is parsed to extract ~20 handcrafted features across 6 groups
2. **Feature Grouping** — Features are organized into semantic groups (structural, lexical, tonal, argumentative, punctuation, fallacy)
3. **Feature Vector Construction** — Groups are concatenated into a unified feature vector; PCA variant also tested
4. **Classification** — Four ML classifiers trained and evaluated using 5-fold cross-validation
5. **Bias Prediction** — Articles classified as Left, Center, or Right

---

## 📚 Datasets

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| Article Bias Dataset | Custom-curated | ~1,500 articles | Main benchmark |
| COCOLOFA | [Kaggle](https://www.kaggle.com) | Standard | Fallacy feature validation |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/tanuj-j18/media-Bias-BTP-work.git
cd media-Bias-BTP-work

# Install dependencies
pip install -r requirements.txt

# Run the main analysis notebook
jupyter notebook "standard dataset/article_bias_documented.ipynb"
```

---

## 📌 Citation

If you use this work, please cite:

```bibtex
@article{mediabias2026,
  title   = {Media Bias Detection Using Feature-Based Analysis},
  author  = {[Your Name(s)]},
  year    = {2026},
  note    = {[Conference/Journal or Preprint]}
}
```

---

## 📬 Contact

For questions or collaboration, feel free to open an issue or reach out via the repository.

---

<p align="center">
  <i>Built with ❤️ for transparent, explainable AI in media analysis</i>
</p>
