# Week 6 - Integrative Capstone Project and Evaluation

## Overview
This capstone demonstrates an end-to-end Data Science pipeline using Python. It combines data acquisition, preprocessing, exploratory data analysis, supervised regression, cross-validation, unsupervised clustering, evaluation, visualization, and recommendations.

## Dataset
**Dataset:** scikit-learn Diabetes Dataset  
**Samples:** 442  
**Predictor features:** 10  
**Task:** Regression + exploratory segmentation

Official source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

> The dataset is used for educational modeling and is not intended for individual clinical diagnosis or treatment decisions.

## Workflow
1. Load a public dataset reproducibly.
2. Inspect data quality, distributions, and correlations.
3. Build leakage-safe preprocessing pipelines.
4. Compare Linear Regression, Ridge Regression, and Random Forest.
5. Evaluate with MAE, RMSE, R2, and five-fold cross-validation.
6. Apply K-Means clustering to discover descriptive profiles.
7. Select k using inertia and silhouette score.
8. Visualize clusters with PCA and a profile heatmap.
9. Translate results into recommendations and limitations.

## Key Results
- Best test-set model: **Ridge Regression**
- Test R2: **0.454**
- Test RMSE: **53.78**
- Test MAE: **42.81**
- Best 5-fold CV R2: **0.479 +/- 0.083**
- Selected K-Means clusters: **2**
- Best silhouette score: **0.237**

## Folder Structure
```text
Week_6_Task/
├── README.md
├── code/
│   ├── capstone_pipeline.ipynb
│   └── requirements.txt
├── dataset/
│   └── diabetes.csv
├── figures/
│   ├── fig1_target_distribution.png
│   ├── fig2_correlation_heatmap.png
│   ├── fig3_bmi_vs_target.png
│   ├── fig4_s5_vs_target.png
│   ├── fig5_actual_vs_predicted.png
│   ├── fig6_residuals.png
│   ├── fig7_feature_importance.png
│   ├── fig8_cluster_selection.png
│   ├── fig9_pca_clusters.png
│   └── fig10_cluster_heatmap.png
├── results/
│   ├── descriptive_statistics.csv
│   ├── missing_values.csv
│   ├── correlation_matrix.csv
│   ├── feature_target_correlations.csv
│   ├── model_comparison.csv
│   ├── cross_validation_summary.csv
│   ├── test_predictions.csv
│   ├── feature_importance.csv
│   ├── cluster_k_selection.csv
│   ├── cluster_assignments.csv
│   ├── cluster_sizes.csv
│   └── cluster_profiles.csv
└── report/
    └── Week_6_Integrative_Capstone_Project_and_Evaluation.docx
```

## Reproduce the Project
```bash
pip install -r code/requirements.txt
jupyter notebook code/capstone_pipeline.ipynb
```

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, SciPy, Jupyter Notebook, python-docx.
