import unittest
from datetime import date, timedelta

from src.risk import check_all_exit_conditions, update_winner_cycle_highs

TODAY = str(date.today())
YESTERDAY = str(date.today() - timedelta(days=1))
LONG_AGO = str(date.today() - timedelta(days=120))


def _pos(stop_type="tight_2", **tranche):
    t = {"n": 2, "shares": 1, "entry_price": 100.0, "entry_date": LONG_AGO,
         "high": 100.0, "stop_type": stop_type}
    t.update(tranche)
    return {"shares": 1, "avg_price": 100.0, "tranches": [t]}


class StopConfirmationTests(unittest.TestCase):
    def test_confirmed_exit_persists_when_not_executed(self):
        positions = {"X": _pos(stop_pending_since=YESTERDAY)}
        exits, _ = check_all_exit_conditions(positions, {"X": 85.0}, {})
        self.assertIn("X", exits)
        t = positions["X"]["tranches"][0]
        self.assertEqual(t["stop_confirmed_since"], TODAY)
        self.assertNotIn("stop_pending_since", t)

        # 隔日未執行、仍在停損線下 → 仍是 EXIT，不會被重置成第一日待確認
        t["stop_confirmed_since"] = YESTERDAY
        exits, pending = check_all_exit_conditions(positions, {"X": 84.0}, {})
        self.assertIn("X", exits)
        self.assertNotIn("X", pending)

    def test_confirmed_marker_cleared_when_price_recovers(self):
        positions = {"X": _pos(stop_confirmed_since=YESTERDAY)}
        exits, pending = check_all_exit_conditions(positions, {"X": 99.0}, {})
        self.assertNotIn("X", exits)
        self.assertNotIn("stop_confirmed_since", positions["X"]["tranches"][0])

    def test_ma200_exit_not_delayed_by_pending_fixed_stop(self):
        positions = {"X": _pos(stop_type="standard")}
        exits, pending = check_all_exit_conditions(positions, {"X": 80.0}, {"X": 120.0})
        self.assertEqual(exits["X"][0]["reason"], "ma200_stop")
        self.assertNotIn("X", pending)


class WinnerCyclePoolTests(unittest.TestCase):
    def test_pool_kept_while_pullback_exit_active(self):
        portfolio = {"positions": {"X": {"winner_cycle_high": 100.0, "tranches": []}}}
        update_winner_cycle_highs(portfolio, {"X": 85.0}, alpha_1y_map={"X": 90.0})
        self.assertEqual(portfolio["positions"]["X"]["winner_cycle_high"], 100.0)

    def test_pool_removed_when_alpha_drops_without_pullback(self):
        portfolio = {"positions": {"X": {"winner_cycle_high": 100.0, "tranches": []}}}
        update_winner_cycle_highs(portfolio, {"X": 95.0}, alpha_1y_map={"X": 90.0})
        self.assertNotIn("winner_cycle_high", portfolio["positions"]["X"])


if __name__ == "__main__":
    unittest.main()
