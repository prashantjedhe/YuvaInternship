# Week 4 – Supervised Learning Model Implementation

## Overview

This project implements a supervised machine learning classification model using the publicly available **Breast Cancer Wisconsin (Diagnostic)** dataset. The objective is to predict whether a breast tumor is malignant or benign from numeric measurements computed from digitized fine-needle aspiration images.

## Dataset

- Samples: 569
- Predictive features: 30
- Target: `target` (0 = malignant, 1 = malignant)
- Missing values: 0

The dataset is loaded from scikit-learn's copy of the UCI Breast Cancer Wisconsin Diagnostic dataset.

## Model

The primary model is **Logistic Regression**, selected because it is a strong baseline for binary classification, provides interpretable coefficients, and works well with standardized numerical features.

A preprocessing-and-model pipeline was used:

1. Median imputation (for robustness)
2. StandardScaler feature standardization
3. Logistic Regression with increased iteration limit

The data was divided into an 80% training set and a 20% stratified test set.

## Validation

Five-fold stratified cross-validation was applied to the training data using accuracy, precision, recall, F1 score, and ROC-AUC. The final model was evaluated once on the held-out test set.

## Key Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.965 |
| Precision | 0.975 |
| Recall | 0.929 |
| F1 Score | 0.951 |
| ROC-AUC | 0.996 |

## Repository Structure

```text
Week_4_Task/
├── README.md
├── code/
│   └── supervised_learning_classification.ipynb
├── dataset/
│   └── breast_cancer.csv
├── figures/
│   ├── fig1_confusion_matrix.png
│   ├── fig2_roc_curve.png
│   ├── fig3_precision_recall.png
│   ├── fig4_feature_coefficients.png
│   ├── fig5_class_distribution.png
│   └── fig6_correlation_heatmap.png
├── results/
│   ├── test_metrics.csv
│   ├── cross_validation_metrics.csv
│   ├── confusion_matrix.csv
│   ├── classification_report.csv
│   ├── test_predictions.csv
│   └── feature_coefficients.csv
└── report/
    └── Week_4_Supervised_Learning_Model_Implementation.docx
```

## Interpretation

The model demonstrates strong predictive performance on the held-out test data. Recall is particularly important in this application because failing to identify a malignant case is more consequential than a benign case being flagged for further review.

The coefficient analysis provides an interpretable view of which standardized features contribute most strongly to the classification decision.

## Limitations and Improvements

- Logistic Regression assumes a linear decision boundary in the transformed feature space.
- Model performance may vary with the random train/test split and dataset characteristics.
- Threshold tuning could be used to emphasize recall or precision depending on the operational objective.
- Future work can compare Logistic Regression with Random Forest, Support Vector Machine, Gradient Boosting, and calibrated ensemble models.
- External validation on an independent clinical dataset would be required before any real-world medical use.

## Reproducibility

Install dependencies and run the notebook from top to bottom:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Conclusion

This project demonstrates the complete supervised learning workflow: problem definition, data preparation, feature scaling, model training, stratified cross-validation, test-set evaluation, visualization, interpretation, and discussion of limitations and improvements.
