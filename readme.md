# User & Bot Classifier

## Overview

This project is a Python library designed to automatically classify social media accounts as either real users or bots. It uses machine learning techniques and follows object-oriented programming (OOP) principles for better organization, maintainability, and extensibility.

**Goal:** Build a tool that can analyze profile characteristics and activity patterns to predict whether an account is a bot (`1`) or a human user (`0`).

## Features

- Load data from a CSV file.
- Automatic data preprocessing:
  - Handle missing values (numeric and categorical).
  - Remove duplicate rows.
- Feature engineering (e.g., create a `profile_completeness` feature).
- Feature vectorization:
  - One-Hot Encoding for categorical variables.
  - Standard scaling for numeric features.
- Train multiple classification models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC AUC
- Visualize results:
  - ROC curves for model comparison.
  - Feature importance for tree-based models.
  - Classification results plot for test data.
- Predict new data using the best model.
- Structured output with interpretable results.

## Dataset

The project expects a file named `bots_vs_users.csv` in the root directory.

- **Format:** CSV
- **Header:** First row must contain column names.
- **Target column:** `target` — where `0` indicates a human user and `1` indicates a bot.
- **Missing values:** `'Unknown'` and empty cells are treated as missing automatically.

## Requirements

- Python 3.7+
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## How to Use

1. Ensure the file `bots_vs_users.csv` is in the same directory as `main.py`.
2. Run the script from the root directory:
   ```bash
   python main.py
