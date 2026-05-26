import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import calculate_all_metrics, calculate_basic_metrics, calculate_regression_metrics


class TestForecastingMetrics(unittest.TestCase):
    def test_all_metrics_keys(self):
        result = calculate_all_metrics(np.array([10, 20, 30]), np.array([12, 18, 33]))
        self.assertIn('MAE', result)
        self.assertIn('RMSE', result)
        self.assertIn('MAPE', result)
        self.assertIn('NMAE', result)
        self.assertIn('R2', result)

    def test_basic_metrics_keys(self):
        result = calculate_basic_metrics(np.array([10, 20, 30]), np.array([12, 18, 33]))
        self.assertIn('MAE', result)
        self.assertIn('RMSE', result)
        self.assertIn('MAPE', result)
        self.assertIn('NMAE', result)

    def test_regression_metrics_keys(self):
        result = calculate_regression_metrics(np.array([10, 20, 30]), np.array([12, 18, 33]))
        self.assertIn('MAE', result)
        self.assertIn('RMSE', result)
        self.assertIn('R2', result)


if __name__ == '__main__':
    unittest.main()
