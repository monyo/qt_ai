"""
_intraday_stop_backtest.py

比較四種停損觸發方式：
  A  收盤 < 停損，單日出場（最激進）
  B  收盤 < 停損，兩日確認出場（現行系統）
  C  盤中低點 < 停損，視為第一天；隔日收盤再確認才出場
  D  盤中低點 < 停損，當日即出場（最敏感）

使用：
    conda run -n qt_env python research/_intraday_stop_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
print("載入 OHLCV 快取...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
df_low   = ohlcv["Low"]

symbols = list(df_close.columns)
dates   = df_close.index
print(f"  {len(symbols)} 支  /  {len(dates)} 日  ({dates[0].date()} ~ {dates[-1].date()})")

FIXED_STOP = 0.15
TRAIL_STOP = 0.25
HOLD_MAX   = 63
MOM_SHORT  = 21
MOM_LONG   = 252
TOP_N      = 5
REBAL_FREQ = 21
START_T    = MOM_LONG + 5

STRATEGIES = {
    "A": "A 收盤單日出場（無確認）",
    "B": "B 收盤兩日確認（現行系統）",
    "C": "C 盤中觸及→收盤確認（混合）",
    "D": "D 盤中低點單日出場（最敏感）",
}

def simulate(strategy):
    results = []
    stop_count = 0
    close = {s: df_close[s].values.astype(float) for s in symbols}
    low   = {s: df_low[s].values.astype(float)   for s in symbols}

    rebal_dates = []
    t = START_T
    while t < len(dates) - HOLD_MAX:
        rebal_dates.append(t)
        t += REBAL_FREQ

    for t0 in rebal_dates:
        # 動能排名
        mom_scores = []
        for s in symbols:
            c = close[s]
            if np.isnan(c[t0]) or np.isnan(c[t0 - MOM_SHORT]) or np.isnan(c[t0 - MOM_LONG]):
                continue
            m_short = c[t0] / c[t0 - MOM_SHORT] - 1
            m_long  = c[t0] / c[t0 - MOM_LONG]  - 1
            mom_scores.append((0.5 * m_short + 0.5 * m_long, s))
        mom_scores.sort(reverse=True)
        picks = [s for _, s in mom_scores[:TOP_N]]

        for sym in picks:
            c = close[sym]
            lo = low[sym]
            entry = c[t0]
            if np.isnan(entry) or entry <= 0:
                continue

            high_px   = entry
            fixed_stop = entry * (1 - FIXED_STOP)
            pending   = False   # 第一天已觸發，等待確認
            exit_ret  = None
            stopped   = False

            for dt in range(1, HOLD_MAX + 1):
                ti = t0 + dt
                if ti >= len(dates):
                    break
                px_c  = c[ti]
                px_lo = lo[ti]
                if np.isnan(px_c):
                    break
                high_px = max(high_px, px_c)
                trail_stop = high_px * (1 - TRAIL_STOP)
                eff_stop   = max(fixed_stop, trail_stop)

                if strategy == "A":
                    # 收盤單日出場
                    if px_c < eff_stop:
                        exit_ret = px_c / entry - 1
                        stopped = True
                        break

                elif strategy == "B":
                    # 收盤兩日確認
                    if pending:
                        exit_ret = px_c / entry - 1
                        stopped = True
                        break
                    if px_c < eff_stop:
                        pending = True

                elif strategy == "C":
                    # 盤中低點觸及 → 視為第一天；隔日收盤確認
                    if pending:
                        exit_ret = px_c / entry - 1
                        stopped = True
                        break
                    if px_lo < eff_stop:
                        pending = True

                elif strategy == "D":
                    # 盤中低點單日出場
                    if px_lo < eff_stop:
                        exit_ret = min(px_c, eff_stop) / entry - 1
                        stopped = True
                        break

            if exit_ret is None:
                exit_ret = c[min(t0 + HOLD_MAX, len(dates) - 1)] / entry - 1

            results.append(exit_ret)
            if stopped:
                stop_count += 1

    return results, stop_count


print("\n回測中（4 種策略）...\n")

summary = {}
for key, label in STRATEGIES.items():
    rets, stops = simulate(key)
    arr = np.array(rets)
    summary[key] = {
        "label":   label,
        "mean":    arr.mean() * 100,
        "median":  np.median(arr) * 100,
        "winrate": (arr > 0).mean() * 100,
        "stops":   stops,
        "n":       len(arr),
    }
    print(f"  {label} 完成（{len(arr)} 筆）")

print()
print("=" * 72)
print(f"  {'策略':<32}  {'平均':>7}  {'中位':>7}  {'勝率':>7}  {'停損次數':>8}")
print("-" * 72)
for key, r in summary.items():
    print(f"  {r['label']:<32}  {r['mean']:>6.2f}%  {r['median']:>6.2f}%  {r['winrate']:>6.1f}%  {r['stops']:>8d}")
print("=" * 72)
print()
print("說明：")
print("  A vs B：收盤確認，單日 vs 兩日（已知 B 勝）")
print("  B vs C：兩日確認，收盤觸發 vs 盤中觸發（今天的問題）")
print("  C vs D：盤中觸發，兩日確認 vs 單日出場")
