"""
evaluate.py
Shared evaluation tools for every model in the project.

Both the linear models and the ensemble models use these functions,
so every model is trained and tested on the SAME split and scored
with the SAME metrics. That keeps the comparison fair.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Settings used by every model (same as the lecturer's examples)
TEST_SIZE = 0.2        # 20% of districts for testing, 80% for training
RANDOM_STATE = 42      # fixes the shuffle so results can be repeated
CV_FOLDS = 5           # number of splits for cross-validation

# Folders next to this file (works no matter where you run it from)
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"


def split_data(dataset):
    """
    Split the dataset into training (80%) and testing (20%) parts.
    Returns X_train, X_test, y_train, y_test.
    """
    X = dataset.data
    y = dataset.target
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def evaluate_predictions(y_true, y_pred):
    """
    Score predictions against the real values.
    Returns a dictionary with MAE, MSE, RMSE and R².
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def cross_validate_model(model, X, y):
    """
    Train and test the model on 5 different splits of the data.
    Returns the average and spread (standard deviation) of MAE and R².
    """
    scores = cross_validate(
        model, X, y,
        cv=CV_FOLDS,
        scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"},
    )
    # scikit-learn reports MAE as a negative number, so flip the sign
    mae = -scores["test_MAE"]
    r2 = scores["test_R2"]
    return {
        "CV_MAE_mean": mae.mean(),
        "CV_MAE_std": mae.std(),
        "CV_R2_mean": r2.mean(),
        "CV_R2_std": r2.std(),
    }


def save_results(rows, filename):
    """
    Save a list of result dictionaries as a CSV table in the results folder.
    Returns the table so it can also be printed.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / filename, index=False)
    return table
