# Week 5 – Deep Learning Application in Data Science

## Overview

This project is part of the Yuva Internship Week 5 task and demonstrates a complete deep learning workflow for a multiclass image-classification problem. The publicly available **scikit-learn Digits dataset** contains 1,797 grayscale images of handwritten digits from 0 to 9. Each image is represented by 64 pixel-intensity features corresponding to an 8×8 image.

A feed-forward neural network (MLP) was implemented with **PyTorch**. The architecture contains an input layer of 64 features, two hidden dense layers of 128 and 64 neurons, ReLU activations, Batch Normalization, Dropout regularization, and a 10-class output layer. The design balances expressive capacity with the small size of the dataset.

The data was split into training, validation, and held-out test sets using stratification. Features were standardized using parameters learned only from the training data. The model was trained with the Adam optimizer and cross-entropy loss. Validation performance was monitored to select the best model state, while dropout, batch normalization, and weight decay helped reduce overfitting.

The final model achieved **98.9% test accuracy**, **98.9% weighted F1**, and a **0.9998 weighted one-vs-rest ROC-AUC**. A separate five-fold cross-validation experiment produced a mean accuracy of approximately **97.8%**, supporting the model’s stability across different data splits.

## Repository Structure

```text
Week_5_Task/
├── README.md
├── code/
│   └── deep_learning_digits.ipynb
├── dataset/
│   └── digits.csv
├── figures/
│   ├── fig1_training_loss.png
│   ├── fig2_training_accuracy.png
│   ├── fig3_confusion_matrix.png
│   ├── fig4_architecture.png
│   ├── fig5_sample_predictions.png
│   └── fig6_class_distribution.png
├── results/
│   ├── training_history.csv
│   ├── test_metrics.csv
│   ├── cross_validation_folds.csv
│   ├── cross_validation_summary.csv
│   ├── confusion_matrix.csv
│   ├── classification_report.csv
│   ├── test_predictions.csv
│   └── model_parameters.csv
└── report/
    └── Week_5_Deep_Learning_Application_in_Data_Science.docx
```

## Key Technologies

Python, PyTorch, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SciPy, Jupyter Notebook.

## Reproducibility

Install dependencies with `pip install torch pandas numpy scikit-learn matplotlib seaborn scipy jupyter`, open the notebook in `code/`, and run all cells from top to bottom. Random seeds are fixed for reproducibility.

## Key Takeaway

The project demonstrates how a modest fully connected neural network can solve a multiclass image-classification task effectively while using regularization, validation, cross-validation, and visual diagnostics to make the training process more reliable and interpretable.
