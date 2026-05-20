"""
_recovery_momentum_backtest.py

回測「恢復期動能」選股策略 vs 標準動能策略 vs SPY

恢復期條件（每個再平衡點掃描）：
  1. 過去 252 日最大跌幅 > -20%
  2. 從 252 日低點反彈 > +20%
  3. 21 日動能 > +5%
  4. 系統動能分數（50% 短期 + 50% 長期）< +20%（尚未被標準系統發現）
  5. 1Y alpha > 0

比較三個策略（各選 TOP 5，持有 63 天，每 21 天再平衡）：
  A. 標準動能（現行系統）
  B. 恢復期動能（今天討論的）
  C. SPY B&H
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
symbols  = list(df_close.columns)
dates    = df_close.index
print(f"  {len(symbols)} 支  /  {len(dates)} 日  ({dates[0].date()} ~ {dates[-1].date()})")

# ── 參數 ───────────────────────────────────────────────────────────────────
MOM_SHORT   = 21
MOM_LONG    = 252
HOLD_DAYS   = 63
REBAL_FREQ  = 21
TOP_N       = 5
FIXED_STOP  = 0.15
TRAIL_STOP  = 0.25
START_T     = MOM_LONG + 10

# 恢復期篩選門檻
MIN_DRAWDOWN  = -0.20   # 過去最大跌幅需超過 -20%
MIN_BOUNCE    = 0.20    # 從低點反彈需超過 +20%
MIN_SHORT_MOM = 0.05    # 21 日動能需 > +5%
MAX_SYS_MOM   = 0.20    # 系統動能分數 < +20%（標準系統尚未發現）

# ── 建立每日收盤矩陣 ────────────────────────────────────────────────────────
close = {s: df_close[s].values.astype(float) for s in symbols}
spy_c = close.get("SPY", None)

def get_mom_score(c, t):
    if t < MOM_LONG or np.isnan(c[t]) or np.isnan(c[t-MOM_SHORT]) or np.isnan(c[t-MOM_LONG]):
        return None
    m_s = c[t] / c[t-MOM_SHORT] - 1
    m_l = c[t] / c[t-MOM_LONG]  - 1
    return 0.5*m_s + 0.5*m_l

def sim_strategy(strategy):
    all_rets = []
    rebal_dates = []
    t = START_T
    while t < len(dates) - HOLD_DAYS:
        rebal_dates.append(t)
        t += REBAL_FREQ

    for t0 in rebal_dates:
        candidates = []
        spy_ret_1y = (spy_c[t0] / spy_c[t0-MOM_LONG] - 1) if spy_c is not None else 0

        for s in symbols:
            c = close[s]
            if len(c) <= t0 or np.isnan(c[t0]):
                continue

            mom = get_mom_score(c, t0)
            if mom is None:
                continue

            if strategy == "A":
                # 標準動能：系統分數最高
                candidates.append((mom, s))

            elif strategy == "B":
                # 恢復期動能
                px     = c[t0]
                m_s    = px / c[t0-MOM_SHORT] - 1
                m_l    = px / c[t0-MOM_LONG]  - 1
                window = c[t0-MOM_LONG:t0+1]
                high252 = np.nanmax(window)
                low252  = np.nanmin(window)

                drawdown = (low252 - high252) / high252
                bounce   = (px - low252) / low252
                alpha_1y = m_l - spy_ret_1y

                if drawdown > MIN_DRAWDOWN:      continue
                if bounce   < MIN_BOUNCE:        continue
                if m_s      < MIN_SHORT_MOM:     continue
                if mom      > MAX_SYS_MOM:       continue
                if alpha_1y < 0:                 continue

                candidates.append((m_s, s))   # 恢復期用短期動能排序

        candidates.sort(reverse=True)
        picks = [s for _, s in candidates[:TOP_N]]
        if not picks:
            continue

        for sym in picks:
            c      = close[sym]
            entry  = c[t0]
            high_px = entry
            fixed_s = entry * (1 - FIXED_STOP)
            exit_r  = None

            for dt in range(1, HOLD_DAYS+1):
                ti = t0 + dt
                if ti >= len(dates): break
                px_c = c[ti]
                if np.isnan(px_c): break
                high_px  = max(high_px, px_c)
                trail_s  = high_px * (1 - TRAIL_STOP)
                eff_stop = max(fixed_s, trail_s)
                if px_c < eff_stop:
                    exit_r = px_c / entry - 1
                    break

            if exit_r is None:
                exit_r = c[min(t0+HOLD_DAYS, len(dates)-1)] / entry - 1
            all_rets.append(exit_r)

    return np.array(all_rets)

print("\n回測中...")
rets_A = sim_strategy("A")
print(f"  標準動能完成（{len(rets_A)} 筆）")
rets_B = sim_strategy("B")
print(f"  恢復期動能完成（{len(rets_B)} 筆）")

# SPY B&H（以 HOLD_DAYS 為單位）
spy_c = close.get("SPY", None)
spy_periods = []
t = START_T
while t + HOLD_DAYS < len(dates) and spy_c is not None:
    r = spy_c[t+HOLD_DAYS] / spy_c[t] - 1
    spy_periods.append(r)
    t += REBAL_FREQ
spy_mean = np.mean(spy_periods) * 100 if spy_periods else 0

def stats(arr, label):
    if len(arr) == 0:
        print(f"  {label}: 無資料")
        return
    mean   = arr.mean() * 100
    median = np.median(arr) * 100
    win    = (arr > 0).mean() * 100
    # 簡單 alpha vs SPY
    spy_alpha = mean - spy_mean
    print(f"  {label:<22}  平均: {mean:>+6.2f}%  中位: {median:>+6.2f}%  "
          f"勝率: {win:>5.1f}%  vs SPY: {spy_alpha:>+6.2f}%  n={len(arr)}")

print()
print("=" * 80)
print(f"  持有期 {HOLD_DAYS} 天  /  再平衡 {REBAL_FREQ} 天  /  TOP {TOP_N}  /  停損 -{FIXED_STOP*100:.0f}%/-{TRAIL_STOP*100:.0f}%")
print("-" * 80)
stats(rets_A, "A 標準動能（現行系統）")
stats(rets_B, "B 恢復期動能（新掃描）")
print(f"  {'C SPY B&H':<22}  平均: {spy_mean:>+6.2f}%  （基準）")
print("=" * 80)

# 分布比較
print()
print("報酬分布（A vs B）：")
for label, arr in [("A 標準動能", rets_A), ("B 恢復期", rets_B)]:
    p25 = np.percentile(arr, 25) * 100
    p75 = np.percentile(arr, 75) * 100
    p10 = np.percentile(arr, 10) * 100
    p90 = np.percentile(arr, 90) * 100
    print(f"  {label}:  P10={p10:+.1f}%  P25={p25:+.1f}%  P75={p75:+.1f}%  P90={p90:+.1f}%")
