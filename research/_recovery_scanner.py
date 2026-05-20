"""
_recovery_scanner.py

掃描「恢復期動能」標的：
  過去有大跌 → 現在強力反彈 → 但 252 日動能還被壓著
  → 等壞成績滾出窗口後，系統動能分數會大幅跳升

篩選條件：
  1. 距 252 日高點跌幅曾超過 -20%（確認有過大跌）
  2. 從 252 日低點已反彈 > +20%（確認恢復中）
  3. 目前 21 日動能 > +5%（短期仍在上升）
  4. 252 日動能 < +20%（長期還被壓著，尚未被系統發現）
  5. 1Y alpha > 0（對 SPY 有超額報酬）

輸出：最有可能在未來 1-3 個月衝進系統前 20 名的標的
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from src.data_loader import get_sp500_tickers

# ── 載入快取（波浪掃描快取有當日價量）──────────────────────────────────────────
import glob
cache_files = sorted(glob.glob("data/_wave_daily_*.pkl"))
if not cache_files:
    print("找不到波浪快取，請先跑一次盤前分析")
    sys.exit(1)

latest = cache_files[-1]
print(f"載入快取：{latest}")
wave_data = pd.read_pickle(latest)
df_close = wave_data["close"]   # shape: (days, symbols)

# 確認欄位
symbols = list(df_close.columns)
dates   = df_close.index
print(f"  {len(symbols)} 支  /  {len(dates)} 日  最新: {dates[-1].date()}")

# ── SPY 基準 ────────────────────────────────────────────────────────────────
spy_c = df_close["SPY"].values if "SPY" in df_close.columns else None

MOM_SHORT  = 21
MOM_LONG   = 252
MIN_DAYS   = MOM_LONG + 10

results = []

for sym in symbols:
    if sym not in df_close.columns:
        continue
    c = df_close[sym].dropna().values
    if len(c) < MIN_DAYS:
        continue

    # 最新值
    px      = c[-1]
    px_21   = c[-MOM_SHORT]
    px_252  = c[-MOM_LONG]
    high252 = c[-MOM_LONG:].max()
    low252  = c[-MOM_LONG:].min()

    if px <= 0 or px_21 <= 0 or px_252 <= 0:
        continue

    m_short = px / px_21  - 1      # 21 日動能
    m_long  = px / px_252 - 1      # 252 日動能
    mom     = 0.5 * m_short + 0.5 * m_long  # 系統動能分數

    from_high252  = (px - high252) / high252         # 距 252 日高點
    from_low252   = (px - low252)  / low252           # 距 252 日低點（反彈幅度）
    drawdown_max  = (low252 - high252) / high252      # 最大跌幅

    # 1Y alpha（對 SPY）
    alpha_1y = None
    if spy_c is not None and len(spy_c) >= MOM_LONG:
        spy_ret = spy_c[-1] / spy_c[-MOM_LONG] - 1
        alpha_1y = m_long - spy_ret

    # ── 篩選條件 ──────────────────────────────────────────────────────────────
    if drawdown_max > -0.20:           # 跌幅未達 -20%，不算大跌
        continue
    if from_low252 < 0.20:            # 從低點反彈不到 +20%
        continue
    if m_short < 0.05:                # 短期動能不夠強
        continue
    if mom > 0.20:                    # 長期動能已超過 +20%，系統早就看到了
        continue
    if alpha_1y is not None and alpha_1y < 0:   # 對 SPY 沒有超額報酬
        continue

    # 估算多久後壞成績滾出視窗
    # 找最低點在過去多少天前
    low_idx = np.argmin(c[-MOM_LONG:])  # 0 ~ 251
    days_since_low = MOM_LONG - low_idx   # 低點距今幾天
    days_to_rolloff = max(0, MOM_LONG - days_since_low)  # 還需多少天滾出

    results.append({
        "symbol":        sym,
        "price":         px,
        "mom_21d":       m_short * 100,
        "mom_252d":      m_long  * 100,
        "mom_score":     mom     * 100,
        "drawdown_max":  drawdown_max * 100,
        "from_low":      from_low252  * 100,
        "from_high252":  from_high252 * 100,
        "alpha_1y":      (alpha_1y or 0) * 100,
        "days_to_rolloff": days_to_rolloff,
    })

df = pd.DataFrame(results)
if df.empty:
    print("沒有符合條件的標的")
    sys.exit(0)

# 排序：短期動能最強 + 距高點最近（快要被系統發現）
df["score"] = df["mom_21d"] * 0.5 + df["from_high252"] * 0.3 + df["alpha_1y"] * 0.2
df = df.sort_values("score", ascending=False).head(20)

print()
print("=" * 90)
print(f"  {'標的':6}  {'現價':>7}  {'21日動能':>8}  {'252日動能':>9}  {'最大跌幅':>8}  {'反彈幅度':>8}  {'距252高':>8}  {'滾出天數':>8}")
print("-" * 90)
for _, r in df.iterrows():
    print(f"  {r['symbol']:6}  ${r['price']:>6.0f}  "
          f"{r['mom_21d']:>+7.1f}%  {r['mom_252d']:>+8.1f}%  "
          f"{r['drawdown_max']:>+7.1f}%  {r['from_low']:>+7.1f}%  "
          f"{r['from_high252']:>+7.1f}%  {r['days_to_rolloff']:>6.0f}天")
print("=" * 90)
print()
print("說明：")
print("  21日動能高  = 短期正在上漲")
print("  252日動能低 = 長期分數還被過去拖累，系統尚未看見")
print("  距252高近   = 快要突破歷史高點，動能分數即將爆發")
print("  滾出天數少  = 壞成績快要出窗口，近期會衝進系統排名")
