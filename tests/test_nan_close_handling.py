import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import premarket_main
from src import breadth_monitor, data_loader, momentum, sector_monitor


class FakeTicker:
    def __init__(self, frame):
        self.frame = frame

    def history(self, **kwargs):
        return self.frame.copy()


class NanCloseHandlingTests(unittest.TestCase):
    def test_current_price_uses_latest_non_nan_close(self):
        frame = pd.DataFrame({"Close": [100.0, 101.5, np.nan]})
        with patch.object(data_loader.yf, "Ticker", return_value=FakeTicker(frame)):
            self.assertEqual(data_loader.fetch_current_prices(["TEST"]), {"TEST": 101.5})

    def test_spy_regime_uses_latest_non_nan_close(self):
        closes = [100.0] * 200 + [110.0, np.nan]
        frame = pd.DataFrame({"Close": closes})
        with patch.object(premarket_main.yf, "Ticker", return_value=FakeTicker(frame)):
            result = premarket_main.get_spy_regime()
        self.assertEqual(result["spy_price"], 110.0)
        self.assertTrue(result["is_bull"])
        self.assertFalse(np.isnan(result["ma200"]))

    def test_momentum_ignores_trailing_nan_close(self):
        closes = list(range(100, 122)) + [np.nan]
        frame = pd.DataFrame({"Close": closes})
        with patch.object(momentum.yf, "Ticker", return_value=FakeTicker(frame)):
            result = momentum.calculate_momentum("TEST", period=21)
        self.assertEqual(result, 21.0)

    def test_sector_return_ignores_trailing_nan_close(self):
        frame = pd.DataFrame({"Close": [100.0, 102.0, 104.0, np.nan]})
        with patch.object(sector_monitor.yf, "Ticker", return_value=FakeTicker(frame)):
            result = sector_monitor._get_return("TEST", 2)
        self.assertEqual(result, round(104.0 / 100.0 - 1, 4))

    def test_breadth_uses_latest_valid_close_per_symbol(self):
        rows = 51
        columns = [f"S{i}" for i in range(120)]
        values = np.tile(np.arange(1, rows + 1, dtype=float)[:, None], (1, len(columns)))
        values[-1, :] = np.nan
        frame = pd.DataFrame(values, columns=columns)
        with patch.object(breadth_monitor, "_load_close_df", return_value=frame):
            result = breadth_monitor.get_stock_breadth()
        self.assertEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
