# Credit Card Fraud Prediction and Predictive Analysis

This project applies supervised and unsupervised machine learning techniques to credit card transaction data. It covers fraud classification, transaction amount prediction, clustering, model evaluation, and data visualization.

## Project Overview

The original dataset contains **284,807 transactions**, **30 input features**, and a target column named `Class`.

- `Class = 0`: Genuine transaction
- `Class = 1`: Fraudulent transaction
- Genuine transactions: 284,315
- Fraudulent transactions: 492
- Fraud rate: Approximately **0.17%**

The dataset is cleaned by removing duplicate rows and missing values. For model training, all 492 fraudulent transactions and a random sample of 5,000 genuine transactions are selected, producing a working sample of **5,492 transactions**.

## Analyses Included

### 1. Fraud Classification

Five classification models are trained using an 80/20 stratified train-test split:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

The features `Time`, `Amount`, and `V1` through `V28` are standardized using `StandardScaler`.

### 2. Transaction Amount Regression

The project predicts transaction amounts using:

- Simple Linear Regression
- Polynomial Regression (Degree 2)
- Multiple Linear Regression

The models are evaluated using MAE, MSE, RMSE, and R² score.

### 3. Unsupervised Clustering

The project also performs:

- K-Means clustering with the Elbow Method
- K-Means initialization comparison
- Agglomerative Hierarchical Clustering
- Dendrogram analysis using single, complete, average, and centroid linkage

## Key Results

### Classification Results

The models are sorted by F1 score in `project_outputs/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Log Loss |
|---|---:|---:|---:|---:|---:|---:|
| SVM | 0.9817 | 0.9870 | 0.8000 | **0.8837** | 0.9847 | 0.0767 |
| Logistic Regression | 0.9808 | 0.9512 | 0.8211 | 0.8814 | 0.9826 | **0.0671** |
| Decision Tree | 0.9808 | 0.9625 | 0.8105 | 0.8800 | 0.9494 | 0.1718 |
| KNN | 0.9799 | 0.9620 | 0.8000 | 0.8736 | 0.9332 | 0.4474 |
| Naive Bayes | 0.9644 | 0.7800 | 0.8211 | 0.8000 | 0.9713 | 0.8224 |

**Best model by F1 score:** SVM  
**Best model by Log Loss:** Logistic Regression

### Regression Results

| Model | MAE | MSE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Simple Linear Regression | 114.0089 | 66145.5142 | 257.1877 | 0.2456 |
| Polynomial Regression (Degree 2) | 106.8985 | 58019.3905 | 240.8721 | 0.3383 |
| Multiple Linear Regression | **37.6475** | **8943.6042** | **94.5706** | **0.8980** |

**Best regression model:** Multiple Linear Regression

It provides the lowest prediction errors and the highest R² score among the tested regression models.

## Repository Structure

```text
Credit-Card-Fraud-Detection-Using-Machine-Learning/
├── creditcard.csv
├── final.py
├── int234 predictive.py
├── int234Report.docx
├── int234Report.pdf
├── README.md
└── project_outputs/
    ├── 01_class_distribution.png
    ├── 02_model_accuracy.png
    ├── 03_model_f1_score.png
    ├── 04_model_log_loss.png
    ├── 05_roc_curves.png
    ├── 06_confusion_matrix.png
    ├── 07_correlation_with_fraud.png
    ├── 08_amount_distribution.png
    ├── 09_simple_linear_regression.png
    ├── 10_polynomial_regression.png
    ├── 11_multiple_regression.png
    ├── 12_kmeans_elbow.png
    ├── 13_kmeans_clusters.png
    ├── 14_dendrograms.png
    ├── model_comparison.csv
    ├── regression_comparison.csv
    └── test_predictions.csv
```

## Requirements

Python 3.9 or later is recommended.

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

## How to Run

Run the main Python script from the project directory:

```bash
python final.py
```

The script:

1. Loads and cleans `creditcard.csv`.
2. Samples genuine transactions and combines them with fraudulent transactions.
3. Trains and evaluates five classification models.
4. Generates classification charts and saves model predictions.
5. Trains and evaluates three regression models.
6. Performs K-Means and hierarchical clustering.
7. Saves charts and CSV reports in the `project_outputs/` directory.

**Important:** Use a relative file path instead of an absolute Windows path so the project can run on different computers.

```python
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data = pd.read_csv(BASE_DIR / "creditcard.csv")
```

## Generated Visual Results

### Classification and Exploratory Analysis

| Class Distribution | Model Accuracy |
|---|---|
| ![Class Distribution](project_outputs/01_class_distribution.png) | ![Model Accuracy](project_outputs/02_model_accuracy.png) |

| Model F1 Score | Model Log Loss |
|---|---|
| ![Model F1 Score](project_outputs/03_model_f1_score.png) | ![Model Log Loss](project_outputs/04_model_log_loss.png) |

| ROC Curves | Confusion Matrix |
|---|---|
| ![ROC Curves](project_outputs/05_roc_curves.png) | ![Confusion Matrix](project_outputs/06_confusion_matrix.png) |

| Feature Correlation with Fraud | Transaction Amount Distribution |
|---|---|
| ![Feature Correlation](project_outputs/07_correlation_with_fraud.png) | ![Amount Distribution](project_outputs/08_amount_distribution.png) |

### Regression

| Simple Linear Regression | Polynomial Regression |
|---|---|
| ![Simple Linear Regression](project_outputs/09_simple_linear_regression.png) | ![Polynomial Regression](project_outputs/10_polynomial_regression.png) |

**Multiple Linear Regression**

![Multiple Linear Regression](project_outputs/11_multiple_regression.png)

### Clustering

| K-Means Elbow Method | K-Means Clusters |
|---|---|
| ![K-Means Elbow Method](project_outputs/12_kmeans_elbow.png) | ![K-Means Clusters](project_outputs/13_kmeans_clusters.png) |

**Hierarchical Clustering**

![Hierarchical Clustering Dendrograms](project_outputs/14_dendrograms.png)

## Tabular Reports

| File | Description |
|---|---|
| `model_comparison.csv` | Classification metrics for all five models |
| `regression_comparison.csv` | Regression metrics for all three models |
| `test_predictions.csv` | Test features, actual status, predicted status, and fraud probability from the best model |

## Reproducibility

The main script uses `random_state=42` for sampling, train-test splitting, model initialization, and clustering wherever applicable.

This ensures consistent results across multiple runs when the same dataset and software environment are used.

## Important Note

The dataset is highly imbalanced, with fraudulent transactions representing approximately **0.17%** of the original data.

Therefore, **accuracy alone is not sufficient** to evaluate fraud detection performance. Precision, Recall, F1 Score, ROC AUC, Log Loss, and the Confusion Matrix should be considered together.

## Conclusion

This project demonstrates the application of machine learning techniques to credit card fraud detection and predictive analysis.

Among the tested classification models, **SVM achieved the highest F1 score**, while **Logistic Regression achieved the lowest Log Loss**.

For transaction amount prediction, **Multiple Linear Regression performed best**, achieving an R² score of **0.8980**.

The project also demonstrates K-Means and hierarchical clustering for exploring transaction patterns.

Overall, this project provides practical experience in data preprocessing, exploratory data analysis, supervised learning, unsupervised learning, model evaluation, and visualization using Python.
