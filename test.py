"""
test.py
Simple checks that our helper functions and dataset loading work correctly.

Run with:   python test.py
"""
import unittest

import pandas as pd

from load import load_dataset
from explore import (
    summarize_column,
    count_values_above,
    count_missing,
    correlation_with_target,
)
from evaluate import split_data, evaluate_predictions
from linear_models import build_models


class TestExploreHelpers(unittest.TestCase):
    """Tests that use small, hand-made numbers so we know the right answer."""

    def test_summary(self):
        stats = summarize_column([1, 2, 3])
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertEqual(stats["mean"], 2.0)

    def test_count_above(self):
        # Only 7 is strictly greater than 5
        self.assertEqual(count_values_above([1, 5, 5, 7], 5), 1)

    def test_count_missing(self):
        # A tiny table with exactly one empty cell
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        self.assertEqual(count_missing(df), 1)

    def test_correlation_direction(self):
        # "up" rises with the target, "down" falls as the target rises
        df = pd.DataFrame({
            "up": [1, 2, 3, 4],
            "down": [4, 3, 2, 1],
            "MedHouseVal": [10, 20, 30, 40],
        })
        corr = correlation_with_target(df)
        self.assertAlmostEqual(corr["up"], 1.0)
        self.assertAlmostEqual(corr["down"], -1.0)


class TestCaliforniaDataset(unittest.TestCase):
    """Tests on the real dataset."""

    @classmethod
    def setUpClass(cls):
        # Load once and reuse in every test below (faster)
        cls.dataset = load_dataset()

    def test_dataset_size(self):
        # 20,640 districts and 8 features
        self.assertEqual(self.dataset.data.shape, (20640, 8))

    def test_target_name(self):
        self.assertEqual(self.dataset.target.name, "MedHouseVal")

    def test_no_missing_values(self):
        self.assertEqual(count_missing(self.dataset.frame), 0)


class TestEvaluation(unittest.TestCase):
    """Tests for the shared evaluation tools and the baseline."""

    def test_metrics_known_values(self):
        # Same numbers as the lecturer's MAE example:
        # errors are 10, 20, 10 -> MAE = 13.33, MSE = (100 + 400 + 100) / 3 = 200
        scores = evaluate_predictions([100, 150, 200], [110, 130, 190])
        self.assertAlmostEqual(scores["MAE"], 13.333, places=2)
        self.assertAlmostEqual(scores["MSE"], 200.0)
        self.assertAlmostEqual(scores["R2"], 0.88)

    def test_perfect_predictions(self):
        scores = evaluate_predictions([1, 2, 3], [1, 2, 3])
        self.assertEqual(scores["MAE"], 0)
        self.assertEqual(scores["R2"], 1.0)

    def test_split_is_80_20(self):
        dataset = load_dataset()
        X_train, X_test, y_train, y_test = split_data(dataset)
        self.assertEqual(len(X_train), 16512)
        self.assertEqual(len(X_test), 4128)

    def test_baseline_predicts_training_mean(self):
        baseline = dict(build_models())["Baseline (training mean)"]
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = [10, 20, 30, 40]
        baseline.fit(X, y)
        # Whatever the input, it always guesses the average: 25
        self.assertEqual(list(baseline.predict(X)), [25, 25, 25, 25])


if __name__ == "__main__":
    unittest.main(verbosity=2)
