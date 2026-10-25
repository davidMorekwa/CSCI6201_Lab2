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


if __name__ == "__main__":
    unittest.main(verbosity=2)
