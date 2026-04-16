"""
_single_day_drop_backtest.py

回測：動能持倉中遇到「單日跌幅 ≥ -X%」，提早平倉 vs 繼續持有哪個好？

策略 A：當天收盤出場（forward return = 0%）
策略 B：繼續持有，用標準停損（固定 -15%，追蹤 -25%），最多再撐 63 天

使用：
    conda run -n qt_env python _single_day_drop_backtest.py
"""

import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── 載入 OHLCV 快取 ────────────────────────────────────────────────────────────
OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
if not os.path.exists(OHLCV_PATH):
    print(f"❌ 找不到 {OHLCV_PATH}")
    sys.exit(1)

print("載入 OHLCV 快取...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
symbols  = list(df_close.columns)
dates    = df_close.index
n_dates  = len(dates)
print(f"  {len(symbols)} 支股票  /  {n_dates} 個交易日  ({dates[0].date()} ~ {dates[-1].date()})")

# ── 參數 ──────────────────────────────────────────────────────────────────────
DROP_THRESHOLDS = [0.08, 0.10, 0.12, 0.15]  # 測試不同門檻
MOM_WINDOW      = 63    # 動能窗口（確認進場時正動能）
FIXED_STOP      = 0.15  # 繼續持有的固定停損
TRAIL_STOP      = 0.25  # 繼續持有的追蹤停損
HOLD_MAX        = 63    # 繼續持有最多天數
MIN_T           = MOM_WINDOW + 5

# ── 掃描事件 ──────────────────────────────────────────────────────────────────
print("\n掃描單日暴跌事件...")

# 先預算各門檻的事件
all_events = {thr: [] for thr in DROP_THRESHOLDS}

for sym in symbols:
    c = df_close[sym].values.astype(float)

    for t in range(MIN_T, n_dates - HOLD_MAX - 2):
        ct   = c[t]
        ct_1 = c[t - 1]

        if np.isnan(ct) or np.isnan(ct_1) or ct_1 <= 0:
            continue

        day_ret = ct / ct_1 - 1

        # 確認進場時動能為正（過去 63 天）
        p_mom = c[t - MOM_WINDOW]
        if np.isnan(p_mom) or p_mom <= 0:
            continue
        if ct / p_mom - 1 <= 0:   # 動能負，跳過
            continue

        # 各門檻獨立計算（一個事件可同時滿足多個門檻）
        for thr in DROP_THRESHOLDS:
            if day_ret <= -thr:
                fixed_stop_px = ct * (1 - FIXED_STOP)
                high_wm       = ct
                b_exit_price  = None

                for dt in range(1, HOLD_MAX + 1):
                    t2 = t + dt
                    if t2 >= n_dates:
                        break
                    c2 = c[t2]
                    if np.isnan(c2) or c2 <= 0:
                        continue
                    high_wm = max(high_wm, c2)
                    trail_stop_px = high_wm * (1 - TRAIL_STOP)

                    if c2 <= fixed_stop_px or c2 <= trail_stop_px:
                        b_exit_price = c2
                        break

                if b_exit_price is None:
                    t_exit = min(t + HOLD_MAX, n_dates - 1)
                    b_exit_price = c[t_exit]
                    if np.isnan(b_exit_price) or b_exit_price <= 0:
                        continue

                fwd_ret_b = b_exit_price / ct - 1
                all_events[thr].append({
                    "fwd_ret_b": fwd_ret_b,
                    "day_ret":   day_ret,
                })

# ── 輸出結果 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  單日暴跌後「提早出場 vs 繼續持有」比較")
print(f"{'='*68}")
print(f"  A 提早出場：當天收盤賣出，forward return = 0%")
print(f"  B 繼續持有：固定停損 -{FIXED_STOP*100:.0f}%，追蹤停損 -{TRAIL_STOP*100:.0f}%，最多 {HOLD_MAX} 天\n")

print(f"  {'門檻':<8} {'N':>6}  {'B平均':>8}  {'B中位':>8}  {'B勝率':>7}  {'B>0%':>7}  {'B P10':>8}  {'B P90':>8}")
print(f"  {'-'*68}")

for thr in DROP_THRESHOLDS:
    events = all_events[thr]
    if not events:
        print(f"  -{thr*100:.0f}%    0 個事件")
        continue
    rets = np.array([e["fwd_ret_b"] for e in events])
    n       = len(rets)
    mean_r  = np.nanmean(rets) * 100
    med_r   = np.nanmedian(rets) * 100
    win     = np.nanmean(rets > 0) * 100
    gt0     = np.nanmean(rets > 0) * 100
    p10     = np.nanpercentile(rets, 10) * 100
    p90     = np.nanpercentile(rets, 90) * 100
    verdict = "✅ 繼續持有" if mean_r > 0 else "❌ 提早出場"
    print(f"  -{thr*100:.0f}%  {n:>6}  {mean_r:>+7.2f}%  {med_r:>+7.2f}%  {win:>6.1f}%  {gt0:>6.1f}%  {p10:>+7.1f}%  {p90:>+7.1f}%  {verdict}")

# ── 詳細分析最常用門檻（-10%）─────────────────────────────────────────────────
thr = 0.10
events = all_events[thr]
if events:
    rets = np.array([e["fwd_ret_b"] for e in events])
    print(f"\n{'='*68}")
    print(f"  詳細分析：單日跌幅 ≥ -{thr*100:.0f}% 後繼續持有的 forward return")
    print(f"{'='*68}")
    print(f"  樣本數：{len(rets)}\n")

    # 分位數分佈
    percs = [5, 10, 25, 50, 75, 90, 95]
    print(f"  分位數：")
    for pp in percs:
        v = np.nanpercentile(rets, pp) * 100
        bar = "█" * int(abs(v) / 2) if abs(v) < 60 else "█" * 30
        sign = "+" if v >= 0 else ""
        print(f"    P{pp:<3} {sign}{v:>+6.1f}%  {'↗' if v >= 0 else '↘'} {bar}")

    # 按「跌幅大小」分組
    print(f"\n  按當日跌幅分組（繼續持有的平均 forward return）：")
    buckets = [(-0.15, -0.10), (-0.20, -0.15), (-0.30, -0.20), (-1.0, -0.30)]
    labels  = ["-10~-15%", "-15~-20%", "-20~-30%", "-30%以上"]
    for (lo, hi), label in zip(buckets, labels):
        mask = [(lo < e["day_ret"] <= hi) for e in events]
        sub  = rets[mask]
        if len(sub) == 0:
            continue
        print(f"    {label:<12}  N={len(sub):>4}  平均 {np.nanmean(sub)*100:>+6.2f}%  "
              f"勝率 {np.nanmean(sub>0)*100:.0f}%")

print(f"\n  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股")
print(f"  ⚠️  不含交易成本、滑點")
print(f"  ⚠️  B 策略的 forward return 從暴跌收盤價起算（非原始進場成本）")
