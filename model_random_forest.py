from load import load_dataset;
from model import split_dataset;
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from pathlib import Path


def random_forest_model():
    """
    Random Forest ensemble model to train on the dataset
    """
    dataset = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(dataset)
    param_dist = {
        "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8],
        "max_depth": [None, 15, 20, 25, 30],
        "min_samples_split": [2,5,10],
        "min_samples_leaf": [1,2,4,8]
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        param_distributions = param_dist,
        n_iter = 25,
        cv = 5,
        scoring = "neg_mean_squared_error",
        random_state = 42,
        n_jobs = -1
    )
    search.fit(x_train, y_train)
    best_rf_model = RandomForestRegressor(
        **search.best_params_, n_estimators=300, random_state=42, n_jobs=-1
    )
    best_rf_model.fit(x_train, y_train)

    return best_rf_model, search, x_train, x_test, y_train, y_test

# if __name__ == "__main__":
#     random_forest_model()