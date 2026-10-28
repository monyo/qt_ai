import unittest

from src.value_pool import (
    screen_independent_fundamental_candidates,
    screen_value_candidates,
    prioritize_fundamental_screen_candidates,
)


class ValuePoolScreeningTests(unittest.TestCase):
    def test_independent_fundamental_screen_uses_only_fundamentals_and_valuation(self):
        result = screen_independent_fundamental_candidates(
            symbols=["PASS", "HIGH_PE", "NO_GROWTH", "LOW_UPSIDE"],
            fundamentals={
                "PASS": {"forward_pe": 20, "peg": 1.2, "revenue_growth": 0.08, "target_mean": 120},
                "HIGH_PE": {"forward_pe": 26, "peg": 1.0, "revenue_growth": 0.1, "target_mean": 150},
                "NO_GROWTH": {"forward_pe": 15, "peg": 0.8, "revenue_growth": -0.01, "target_mean": 130},
                "LOW_UPSIDE": {"forward_pe": 15, "peg": 0.8, "revenue_growth": 0.1, "target_mean": 110},
            },
            current_prices={"PASS": 100, "HIGH_PE": 100, "NO_GROWTH": 100, "LOW_UPSIDE": 100},
        )

        self.assertEqual([item["symbol"] for item in result], ["PASS"])
        self.assertNotIn("momentum", result[0])
        self.assertNotIn("alpha_3y", result[0])

    def test_fundamental_shortlist_prioritizes_quality_before_target_upside(self):
        result = prioritize_fundamental_screen_candidates([
            {"symbol": "HIGH_UPSIDE", "upside_pct": 80},
            {
                "symbol": "QUALITY", "upside_pct": 20, "free_cashflow": 1,
                "operating_cashflow": 1, "roic": 0.1,
                "gross_margin": 0.4, "operating_margin": 0.2,
            },
        ], limit=1)

        self.assertEqual([item["symbol"] for item in result], ["QUALITY"])
        self.assertEqual(result[0]["screen_quality_score"], 5)
        self.assertEqual(result[0]["screen_quality_coverage"], 5)

    def test_keeps_value_candidate_when_momentum_has_confirmed(self):
        result = screen_value_candidates(
            symbols=["AVGO"],
            momentum_map={"AVGO": 18.5},
            alpha_3y_map={"AVGO": 12.0},
            fundamentals={
                "AVGO": {
                    "forward_pe": 19.0,
                    "peg": 0.35,
                    "target_mean": 533.0,
                    "revenue_growth": 0.24,
                    "sector": "Technology",
                }
            },
            current_prices={"AVGO": 365.0},
            held_symbols=set(),
            slots=1,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "AVGO")
        self.assertEqual(result[0]["value_type"], "value_with_confirmation")

    def test_does_not_duplicate_main_pool_candidate(self):
        result = screen_value_candidates(
            symbols=["AVGO"],
            momentum_map={"AVGO": 18.5},
            alpha_3y_map={"AVGO": 12.0},
            fundamentals={
                "AVGO": {
                    "forward_pe": 19.0,
                    "peg": 0.35,
                    "target_mean": 533.0,
                    "revenue_growth": 0.24,
                    "sector": "Technology",
                }
            },
            current_prices={"AVGO": 365.0},
            held_symbols=set(),
            slots=1,
            excluded_symbols={"AVGO"},
        )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
