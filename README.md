# Machine Learning for Breast Cancer Classification and Regression

A robust, modular machine learning pipeline developed for breast cancer prognosis, predicting both binary pathological complete response (**`pCR`**) and continuous relapse-free survival (**`RelapseFreeSurvival`**). 

Built as a foundational project for exploring end-to-end machine learning engineering, this repository emphasizes modular architecture and comprehensive evaluation workflows.

## 📌 Project Overview

Working with a clinical and imaging dataset of **400 patient records**, this project tackles two distinct prognostic tasks:
1. **Classification (`pCR`):** Predicting whether a patient achieves pathological complete response following treatment (~21% positive class imbalance).
2. **Regression (`RelapseFreeSurvival`):** Predicting time-to-relapse for continuous survival analysis.

The codebase is structured to mirror production machine learning standards, cleanly separating configuration management, preprocessing, model training, and visual evaluation.

## 🔬 Key Engineering Highlights & Lessons Learned

As a foundational machine learning project, this repository served as a hands-on exploration of real-world data challenges:
* **Strict Leak-Free Pipeline Architecture:** Implemented `sklearn.pipeline.Pipeline` combined with `ColumnTransformer` inside 5-fold cross-validation (`StratifiedKFold` for classification, `KFold` for regression) to ensure that imputation, feature scaling, and 13-component **Principal Component Analysis (PCA)** are fitted strictly on training folds, keeping test data completely unseen.
* **Mixed Data Type Handling:** Designed preprocessors to isolate binary flags (`ER`, `PgR`, `HER2`, `LNStatus`) from continuous and ordinal clinical variables (`Age`, `TumourStage`, `Proliferation`) and unscaled or whitened image components, preventing distance-based models (`SVM`, `KNN`) from being dominated by scale discrepancies.
* **Class Imbalance & Non-Linearity:** Addressed extreme class imbalance in `pCR` using class weighting and optimized evaluation via **ROC-AUC** and **PR-AUC**, alongside non-linear models (`Random Forest`, `SVM`, `XGBoost`, `Decision Tree`, `KNN`, and `Naive Bayes`).
* **Robustness & Reproducibility:** Fixed random seeds (`random_state=23`) across all splits and added a **10-seed stability evaluation** to analyze model sensitivity against initialization randomness via boxplots and stripplots.

## 🗂️ Project Structure

> **Note on Architecture:** This framework and pipeline are built specifically for this dataset. While minor adjustments can be made to the configuration or helper scripts within the `src/` folder, the code architecture and preprocessing steps are tightly coupled to the analytical workflow demonstrated in the companion Jupyter notebook.

```text
Machine-Learning-for-Breast-Cancer-Classification-and-Regression-Tasks/
├── configs/
│   └── configurations.json        # Centralized hyperparameter grids and fixed parameters.
├── data/
│   ├── external/                  # Raw dataset storage.
│   └── processed/                 # Intermediate preprocessed features and targets.
├── results/
│   ├── classification/            # CV summaries, test metrics (.xlsx), and 2x2/stability plots (.png).
│   └── regression/                # CV summaries, test metrics (.xlsx), and 2x2/stability plots (.png).
├── src/
│   ├── data_loader.py             # Data loader utilities.
│   ├── data_preprocessing.py      # Imputation, encoding, and PCA feature manipulation.
│   ├── evaluation.py              # Metric logging, seed stability testing, and 2x2 dashboard rendering.
│   ├── model.py                   # Dynamic model initialization from configurations.json.
│   └── train.py                   # Pipeline wrapping and GridSearchCV execution.
├── main.py                        # Main orchestration script.
└── requirements.txt               # Project dependencies.
```

## ⚙️ Tech Stack & Dependencies

* **Language:** Python 3.13
* **Libraries:** `scikit-learn`, `xgboost`, `pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`, `openpyxl`, `xlrd`, `ipython`

Install all required packages via pip:
```bash
pip install -r requirements.txt
```

## 📝 Note on Origin
This repository is a curated, standalone version of a collaborative university project. While other exploratory analysis steps (such as exhaustive feature selection experiments and hidden test-set evaluations) were part of the broader group work, this repository focuses specifically on the modular core pipeline, hyperparameter tuning, and multi-seed stability framework.
