"""
explore.py
Basic exploratory analysis of the California Housing dataset.

Run on its own with:   python explore.py
"""
import pandas as pd

from load import load_dataset

# Show every column when printing tables (instead of hiding some with "...")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ---------- Small helper functions (these are what test.py checks) ----------

def summarize_column(values):
    """
    Return the min, max and mean of a list (or pandas column) of numbers.
    Example: summarize_column([1, 2, 3]) -> {"min": 1.0, "max": 3.0, "mean": 2.0}
    """
    values = list(values)
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def count_values_above(values, threshold):
    """
    Count how many numbers are strictly greater than the threshold.
    Example: count_values_above([1, 5, 5, 7], 5) -> 1
    """
    return sum(1 for v in values if v > threshold)


def count_missing(dataset_df):
    """
    Return the total number of missing (empty) values in the whole table.
    """
    return int(dataset_df.isnull().sum().sum())


def correlation_with_target(dataset_df, target_name="MedHouseVal"):
    """
    Show how strongly each feature moves together with the target.
    Close to +1 = strong positive link, close to -1 = strong negative link,
    close to 0 = little or no straight-line link.
    """
    correlations = dataset_df.corr()[target_name].drop(target_name)
    return correlations.sort_values(ascending=False)


# ---------- Main exploration ----------

def explore_dataset(dataset):
    """
    Print a basic overview of the dataset.
    """
    df = dataset.frame              # full table: 8 features + target column
    X = dataset.data                # features (the inputs)
    y = dataset.target              # target (what we want to predict)

    print("===== Dataset overview =====")
    print("Shape of full table (rows, columns):", df.shape)
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("Feature names:", list(X.columns))
    print("Target variable:", y.name, "(median house value, in $100,000s)")
    print("Task type: regression (we predict a number, not a category)")

    print("\n===== Descriptive statistics =====")
    print(df.describe().round(2))   # count, mean, std, min, quartiles, max

    print("\n===== Missing values =====")
    print(df.isnull().sum())
    print("Total missing values:", count_missing(df))

    print("\n===== Target summary =====")
    stats = summarize_column(y)
    print("Min: {:.2f}  Max: {:.2f}  Mean: {:.2f}".format(
        stats["min"], stats["max"], stats["mean"]))

    # House values in this dataset were capped at about 5.0 ($500,000),
    # so many districts sit exactly at the top. This is worth knowing for modeling.
    capped = count_values_above(y, 5.0)
    print("Districts at the $500,000 cap:", capped)

    print("\n===== Correlation of each feature with house value =====")
    print(correlation_with_target(df).round(2))


if __name__ == "__main__":
    dataset = load_dataset()
    explore_dataset(dataset)
