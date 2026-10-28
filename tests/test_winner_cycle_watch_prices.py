import unittest

import premarket_main


class WinnerCycleWatchPriceTests(unittest.TestCase):
    def test_price_universe_includes_watch_symbols_and_deduplicates(self):
        builder = getattr(premarket_main, "_build_price_symbol_universe", None)
        if not callable(builder):
            self.fail("watch-price universe helper is missing")

        symbols = builder(
            ["AAPL", "MU"],
            ["MU", "AMAT"],
            {"SNDK": {}, "KLAC": {}},
        )

        self.assertEqual(symbols, ["AAPL", "MU", "AMAT", "SNDK", "KLAC"])

    def test_missing_watch_quote_renders_as_unavailable_not_zero(self):
        formatter = getattr(premarket_main, "_format_winner_cycle_watch_quote", None)
        if not callable(formatter):
            self.fail("watch quote formatter is missing")

        self.assertEqual(formatter(None, 1641.11), ("N/A", "N/A"))
        self.assertEqual(formatter(0, 1641.11), ("N/A", "N/A"))

    def test_valid_watch_quote_renders_price_and_rebound(self):
        formatter = getattr(premarket_main, "_format_winner_cycle_watch_quote", None)
        if not callable(formatter):
            self.fail("watch quote formatter is missing")

        self.assertEqual(formatter(1581.82, 1641.11), ("$1581.82", "-3.6%"))


if __name__ == "__main__":
    unittest.main()
