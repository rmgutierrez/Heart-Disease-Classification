# Heart Disease Classification

This project is focused on predicting the presence of heart disease using machine learning techniques. It leverages multiple classification algorithms and evaluates their performance based on accuracy and confusion matrices.

## Project Overview

Heart disease is one of the leading causes of death globally. Early detection can significantly improve patient outcomes. In this project, we use a heart disease dataset to build and compare the performance of three classification models:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest

The project includes data preprocessing, hyperparameter tuning using GridSearchCV, model evaluation, and visualization of confusion matrices.

## Features

- Data loading and preprocessing (standardization and train-test split)
- Training and hyperparameter tuning for three classifiers
- Model evaluation using:
  - Accuracy
  - Classification report
  - Confusion matrix
- Visualization of confusion matrices using Seaborn and Matplotlib

## Repository Structure

```
Heart-Disease-Classification/
│
├── heart.csv                     # Dataset
├── main.py                       # Main Python script with model logic
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── report.pdf                    # PDF report (summary, methodology, results)
└── confusion-matrix/             # Confusion matrix plots
    ├── logistic-regression.png
    ├── SVM.png
    └── random-forest.png
```

## Installation

1. Clone this repository:

   ```bash
   git clone <repo-url>
   cd Heart-Disease-Classification
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the project:

```bash
python main.py
```

Make sure that `heart.csv` is in the same directory as `main.py`.

## Sample Results

| Model                     | Accuracy | Precision (0/1) | Recall (0/1) | F1-score (0/1) |
|--------------------------|----------|------------------|--------------|----------------|
| Logistic Regression      | 0.81     | 0.86 / 0.76      | 0.75 / 0.87  | 0.80 / 0.81    |
| Support Vector Machine   | 0.96     | 0.95 / 0.99      | 0.99 / 0.94  | 0.97 / 0.96    |
| Random Forest Classifier | 0.99     | 0.98 / 1.00      | 1.00 / 0.98  | 0.99 / 0.99    |

### Model Insights

- **Logistic Regression**: Performed as a baseline model with reasonable accuracy, with balanced performance between classes but struggled with class imbalance compared to other models.
- **Support Vector Machine**: Significantly improved accuracy to 96%, with the RBF kernel and `C = 10` yielding strong performance.
- **Random Forest**: Outperformed all other models, achieving an accuracy of 99%. This was likely due to its ability to handle complex feature interactions and ensemble learning.

## Report

Refer to `report.pdf` for a detailed explanation of the project's goals, methods, and results.

## Requirements

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn

Install these via the provided `requirements.txt`.

