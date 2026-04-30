<div align="center">
  <h1>Wine Quality Prediction with SMOTEWB & LightGBM</h1>
  <p><i>An advanced machine learning pipeline tackling severe class imbalance for robust quality classification.</i></p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Model-LightGBM-ff69b4.svg" alt="Model">
    <img src="https://img.shields.io/badge/Tuning-Optuna-brightgreen.svg" alt="Tuning">
    <img src="https://img.shields.io/badge/Resampling-SMOTEWB-orange.svg" alt="SMOTE">
  </p>
</div>

<br>

> [!IMPORTANT]  
> **DISCLAIMER**: This project and the methodologies applied herein were recreated and adapted from the following academic research paper:
>
> 📄 **Title:** A Fault Diagnosis Method for Avionics Equipment Based on SMOTEWB-LGBM  
> ✍️ **Authors:** Gen Li, Wenhai Li, Tianzhu Wen, Weichao Sun, and Xi Tang  
> 📅 **Publication Year:** 2024

## Overview

This repository contains an end-to-end Machine Learning pipeline designed to predict wine quality ratings. The dataset presents a challenging scenario characterized by **severe class imbalance**, where extreme ratings (like 3, 4, and 8) are heavily underrepresented compared to average ratings (5 and 6).

To solve this, the project implements a **Successive SMOTEWB** (Synthetic Minority Over-sampling Technique with Window-Based) approach to synthetically augment minority classes, coupled with a highly optimized **LightGBM (LGBM) Classifier** tuned via **Optuna**.

---

## Repository Structure

```text
SMOTEWB-LGBM/
├── data_training.csv       # Training dataset with features and target labels ('quality')
├── data_testing.csv        # Testing dataset without target labels
├── PDAB_UTS.ipynb          # Main Jupyter Notebook containing the full pipeline
├── hasilprediksi_011.csv   # Final prediction output for the test dataset
├── Reference.pdf           # Key literature and references on SMOTEWB methodologies
└── README.md               # This documentation file
```

---

## Methodology

Our pipeline is structured into several robust data processing and modeling stages:

### 1. Exploratory Data Analysis (EDA) & Preprocessing

- **Visualizations:** Generates distribution density plots (KDE) and boxplots for all physicochemical features.
- **Outlier Handling:** Implements **Interquartile Range (IQR) bounds** clipping (`np.clip`) to mitigate the effect of extreme outliers without discarding valuable minority samples.
- **Scaling:** Utilizes `StandardScaler` to ensure all features contribute equally to the distance-based algorithms.

### 2. Successive SMOTEWB Resampling

Standard SMOTE can sometimes struggle with extreme multi-class imbalance. We implement a custom `successive_smotewb` algorithm:

- Dynamically resamples each minority class up to the majority class count (`max_count`).
- Iteratively processes each class in ascending order of their frequencies.
- Incorporates fallback mechanisms (Random Oversampling) to ensure dataset stability if SMOTEWB fails to generate valid synthetic samples.

### 3. Hyperparameter Tuning with Optuna

We maximize the **Macro-F1 Score** using `Optuna` to ensure the model doesn't just predict the majority class well, but performs strongly across _all_ quality classes.

- **Cross-Validation:** 5-Fold Stratified K-Fold.
- **Tuned Parameters:** `n_estimators`, `num_leaves`, `learning_rate`, `lambda_l1`, `lambda_l2`, `feature_fraction`, `bagging_fraction`, etc.

### 4. Custom Class Weighting & LightGBM

- To further penalize misclassifications of rare classes (3, 4, and 8), custom multiplier weights (e.g., `3x`) are applied.
- The final **LGBMClassifier** is trained using the synthetically balanced dataset and optimized hyperparameters.

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/galihkjaya/SMOTEWB-LGBM.git
   cd SMOTEWB-LGBM
   ```
2. **Install Dependencies:**
   Ensure you have the required Python libraries installed:
   ```bash
   pip install numpy pandas scikit-learn lightgbm smote-variants optuna matplotlib seaborn
   ```
3. **Execute the Notebook:**
   Open and run the `PDAB_UTS.ipynb` file in Jupyter Notebook, JupyterLab, or Google Colab.
4. **Output:**
   The notebook will output the final predictions into `hasilprediksi_011.csv`, which maps the `Id` to the predicted `quality`.

---

## Reference

This project builds upon advanced oversampling techniques to deal with highly imbalanced learning.

Please refer to the included **`Reference.pdf`** file for the primary theoretical background, literature, and empirical studies supporting the use of the **SMOTEWB** algorithm in classification tasks.

- **Key Libraries:** [LightGBM](https://lightgbm.readthedocs.io/en/latest/), [Optuna](https://optuna.readthedocs.io/en/stable/), [smote-variants](https://smote-variants.readthedocs.io/).
