"""
linear_models.py
Baseline and Linear Regression for the California Housing dataset.

Run on its own with:   python linear_models.py

What it does:
1. Splits the data 80/20 (shared settings from evaluate.py)
2. Trains a training-mean baseline and a Linear Regression model
3. Scores both with MAE, MSE, RMSE and R², plus 5-fold cross-validation
4. Saves the results table to results/linear_results.csv
5. Saves an actual-vs-predicted plot to figures/linear-actual-vs-predicted.png
6. Saves the Linear Regression coefficients to results/linear_coefficients.csv
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from load import load_dataset
from evaluate import (
    split_data,
    evaluate_predictions,
    cross_validate_model,
    save_results,
    RESULTS_DIR,
    FIGURES_DIR,
)


def build_models():
    """
    Return the models to compare, as (name, model) pairs.
    """
    return [
        # Always predicts the average house price of the training data
        ("Baseline (training mean)", DummyRegressor(strategy="mean")),
        # Learns one weight per feature and adds them up
        ("Linear Regression", LinearRegression()),
    ]


def run_linear_models(dataset):
    """
    Train, test and cross-validate each model.
    Returns the results table and the trained Linear Regression model.
    """
    X_train, X_test, y_train, y_test = split_data(dataset)
    print("Training samples:", len(X_train), "| Testing samples:", len(X_test))

    rows = []
    trained = {}
    for name, model in build_models():
        model.fit(X_train, y_train)                    # learn from training data
        y_pred = model.predict(X_test)                 # guess prices for test data
        scores = evaluate_predictions(y_test, y_pred)  # compare with real prices
        cv_scores = cross_validate_model(model, dataset.data, dataset.target)
        rows.append({"Model": name, **scores, **cv_scores})
        trained[name] = (model, y_pred)

    table = save_results(rows, "linear_results.csv")
    lr_model, lr_pred = trained["Linear Regression"]
    plot_actual_vs_predicted(y_test, lr_pred)
    save_coefficients(lr_model, dataset.feature_names)
    return table, lr_model


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Scatter plot of real vs predicted prices on the test set.
    Points on the dashed diagonal are perfect predictions.
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, s=5, alpha=0.3)
    plt.plot([0, 5], [0, 5], "r--", label="Perfect prediction")
    plt.xlabel("Actual median house value ($100,000)")
    plt.ylabel("Predicted median house value ($100,000)")
    plt.title("Linear Regression: Actual vs. Predicted (Test Set)")
    plt.legend()
    plt.savefig(FIGURES_DIR / "linear-actual-vs-predicted.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_coefficients(model, feature_names):
    """
    Save the weight Linear Regression learned for each feature.
    """
    coefs = pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
    coefs = coefs.sort_values("Coefficient", ascending=False)
    RESULTS_DIR.mkdir(exist_ok=True)
    coefs.to_csv(RESULTS_DIR / "linear_coefficients.csv", index=False)
    return coefs


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    dataset = load_dataset()
    table, model = run_linear_models(dataset)
    print("\n===== Results =====")
    print(table.round(3).to_string(index=False))
    print("\nSaved: results/linear_results.csv, results/linear_coefficients.csv,")
    print("       figures/linear-actual-vs-predicted.png")
