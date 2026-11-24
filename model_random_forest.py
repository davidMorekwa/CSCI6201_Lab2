from load import load_dataset;
from model import split_dataset;
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "results"

def random_forest_model():
    """
    Random Forest ensemble model to train on the dataset
    """
    dataset = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(dataset)
    # TODO: Try other paramters and compare performance
    rf_model = RandomForestRegressor(
        max_depth = None,
        min_samples_split=5,
        random_state = 42,
        n_jobs = 1
    )

    rf_model.fit(x_train, y_train)

    y_pred = rf_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # print("[+] RF: Mean Squared Error: ", mse)
    report = (f"California Housing dataset regression model using RandomForest\n"
              f"Training samples: {len(x_train)}\nTest samples: {len(x_test)}\n"
              f"Mean Squared error: {mse:.2f}\nr2-score: {r2:.2f}\nMean Absolute Error: {mae:.2f}\n"
              )
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "random_forest_preliminary_results.txt").write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    random_forest_model()