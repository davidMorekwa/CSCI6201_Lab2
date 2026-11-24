from pathlib import Path
from model_random_forest import random_forest_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load import load_dataset

CURR_DIR = Path(__file__).resolve().parent

def evaluate_random_forest():

    model, search, x_train, x_test, y_train, y_test = random_forest_model()
    y_pred = model.predict(x_test)
    errors = y_test - y_pred
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results_df = pd.DataFrame({
        "y_actual": y_test,
        "y_pred": y_pred,
        "error": errors,
        "abs_error": np.abs(errors)
    })
    # print("[+] RF: Mean Squared Error: ", mse)
    report = (f"California Housing dataset regression model using RandomForest\n"
                f"Training samples: {len(x_train)}\nTest samples: {len(x_test)}\n"
                f"Fine Tuned Parameter Values: {search.best_params_}\n"
                f"Mean Squared error: {mse:.2f}\nr2-score: {r2:.2f}\nMean Absolute Error: {mae:.2f}\n"
                )
    (CURR_DIR / "results").mkdir(exist_ok=True)
    (CURR_DIR / "results" / "random_forest_preliminary_results.txt").write_text(report, encoding="utf-8")
    results_df.to_csv((CURR_DIR/"results"/"random_forest_predictions.cvs"), index=False, float_format="%.2f")

def plot_evaluation():
    df = pd.read_csv((CURR_DIR/"results"/"random_forest_predictions.cvs"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 1. Actual vs. Predicted
    axes[0].scatter(df["y_actual"], df["y_pred"], alpha=0.25, s=12)
    lims = [0, max(df["y_actual"].max(), df["y_pred"].max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y_pred = y_actual)")
    axes[0].set_xlabel("Actual Value ($100k)")
    axes[0].set_ylabel("Predicted Value ($100k)")
    axes[0].set_title("Actual vs. Predicted")
    axes[0].legend()

    # 2. Residuals vs. Predicted
    axes[1].scatter(df["y_pred"], df["error"], alpha=0.25, s=12)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted Value ($100k)")
    axes[1].set_ylabel("Error (y_actual - y_pred)")
    axes[1].set_title("Residuals vs. Predicted")

    plt.tight_layout()
    plt.savefig((CURR_DIR/"figures"/"random_forest_performance.png"), dpi=300)

plot_evaluation()
