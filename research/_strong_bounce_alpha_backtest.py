"""
_strong_bounce_alpha_backtest.py

強化版強彈回測：計算相對 SPY 的超額報酬（alpha），
並分析 COVID 期間對結果的影響。

問題：距40日高點 -15%~-25% + 單日 ≥+8% 強彈，
     往後的 21天/63天 alpha（vs SPY）是否顯著？

使用：
    conda run -n qt_env python _strong_bounce_alpha_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import yfinance as yf
from datetime import timedelta

# ── 載入價格快取 ──────────────────────────────────────────────────────────────
CACHE = "data/_protection_bt_prices.pkl"
print("載入價格快取...")
df_prices = pickle.load(open(CACHE, "rb"))  # DataFrame: (1695 dates, 501 symbols)
symbols = list(df_prices.columns)
dates   = df_prices.index
n_dates = len(dates)
print(f"  {len(symbols)} 支股票  /  {n_dates} 個交易日  ({dates[0].date()} ~ {dates[-1].date()})")

# ── 下載 SPY ──────────────────────────────────────────────────────────────────
print("下載 SPY 歷史資料...")
spy_raw = yf.download("SPY", start=dates[0].strftime("%Y-%m-%d"),
                      end=(dates[-1] + timedelta(days=5)).strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
spy_series = spy_raw["Close"].reindex(dates, method="ffill")
spy = spy_series.values.astype(float)
print(f"  SPY 資料筆數: {(~np.isnan(spy)).sum()}")

# ── 參數 ──────────────────────────────────────────────────────────────────────
HIGH_WINDOW    = 40
FROM_HIGH_MIN  = -0.25
FROM_HIGH_MAX  = -0.15
BOUNCE_THRESH  = 0.08
FORWARD_DAYS   = [5, 10, 21, 42, 63]

# COVID 爆發 + 反彈期：2020-02-20 ~ 2020-12-31
COVID_START = pd.Timestamp("2020-02-20")
COVID_END   = pd.Timestamp("2020-12-31")

# ── 掃描觸發事件 ──────────────────────────────────────────────────────────────
events    = []   # 強彈
controls  = []   # 對照組

print("\n掃描觸發事件...")
for sym in symbols:
    closes = df_prices[sym].values.astype(float)
    n = len(closes)

    for t in range(HIGH_WINDOW + 1, n - max(FORWARD_DAYS) - 1):
        p0 = closes[t]
        p1 = closes[t - 1]
        if np.isnan(p0) or np.isnan(p1) or p0 <= 0 or p1 <= 0:
            continue

        day_ret  = p0 / p1 - 1
        high_40d = np.nanmax(closes[t - HIGH_WINDOW: t])
        if high_40d <= 0:
            continue

        from_high = p0 / high_40d - 1
        if not (FROM_HIGH_MIN <= from_high <= FROM_HIGH_MAX):
            continue

        # SPY 同期資料齊全？
        spy_slice = spy[t: t + max(FORWARD_DAYS) + 1]
        if np.any(np.isnan(spy_slice)) or spy[t] <= 0:
            continue

        ev = {
            "sym": sym, "t": t,
            "date": dates[t],
            "p0": p0, "high_40d": high_40d,
            "from_high": from_high,
            "day_ret": day_ret,
            "closes_fwd": closes[t: t + max(FORWARD_DAYS) + 1],
            "spy_fwd":    spy[t: t + max(FORWARD_DAYS) + 1],
            "is_covid":   COVID_START <= dates[t] <= COVID_END,
        }

        if day_ret >= BOUNCE_THRESH:
            events.append(ev)
        elif abs(day_ret) < 0.02 and t % 5 == 0:
            controls.append(ev)

print(f"  強彈事件: {len(events)} 筆")
print(f"    其中 COVID 期間: {sum(e['is_covid'] for e in events)} 筆")
print(f"  對照組:   {len(controls)} 筆")

# ── 分析函數 ──────────────────────────────────────────────────────────────────
def analyze_alpha(group, label, fwd_days=FORWARD_DAYS):
    if not group:
        return
    print(f"\n{'='*65}")
    print(f"  {label}  (N={len(group)})")
    print(f"{'='*65}")

    print(f"\n  往後 alpha（股票報酬 - SPY 報酬）：")
    print(f"  {'天數':>6}  {'平均alpha':>10}  {'中位alpha':>10}  {'alpha>0':>9}  {'alpha>+3%':>10}")
    print(f"  {'-'*55}")
    for fwd in fwd_days:
        alphas = []
        for ev in group:
            cf = ev["closes_fwd"]
            sf = ev["spy_fwd"]
            if len(cf) > fwd and len(sf) > fwd and cf[0] > 0 and sf[0] > 0:
                stock_ret = cf[fwd] / cf[0] - 1
                spy_ret   = sf[fwd] / sf[0] - 1
                alphas.append(stock_ret - spy_ret)
        if not alphas:
            continue
        alphas = np.array(alphas)
        print(f"  {fwd:>6}天  {np.mean(alphas)*100:>+9.1f}%  "
              f"{np.median(alphas)*100:>+9.1f}%  "
              f"{np.mean(alphas>0)*100:>8.1f}%  "
              f"{np.mean(alphas>0.03)*100:>9.1f}%")

    # 回前高機率（不受 SPY 影響，但顯示供參考）
    reach_days = []
    for ev in group:
        cf  = ev["closes_fwd"]
        tgt = ev["high_40d"]
        for d in range(1, len(cf)):
            if cf[d] >= tgt:
                reach_days.append(d)
                break
    reached = len(reach_days)
    total   = len(group)
    print(f"\n  回到前高機率（{max(fwd_days)}日內）: {reached/total*100:.1f}%")
    if reach_days:
        rd = np.array(reach_days)
        print(f"  回前高中位時間: {np.median(rd):.0f} 交易日")


# ── 全期分析 ──────────────────────────────────────────────────────────────────
analyze_alpha(events,   "強彈組（全期 2019-2026）")
analyze_alpha(controls, "對照組（全期 2019-2026）")

# ── 排除 COVID 後分析 ─────────────────────────────────────────────────────────
events_no_covid   = [e for e in events   if not e["is_covid"]]
controls_no_covid = [e for e in controls if not e["is_covid"]]

analyze_alpha(events_no_covid,   "強彈組（排除 COVID 2020-02~12）")
analyze_alpha(controls_no_covid, "對照組（排除 COVID 2020-02~12）")

# ── COVID 期間單獨看 ──────────────────────────────────────────────────────────
events_covid   = [e for e in events   if e["is_covid"]]
analyze_alpha(events_covid, "強彈組（僅 COVID 期間）")

# ── 年度拆解：每年強彈 alpha 分佈 ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("  強彈組 21天 alpha（按年份）")
print(f"{'='*65}")
print(f"  {'年份':>6}  {'N':>5}  {'平均alpha':>10}  {'中位alpha':>10}  {'alpha>0':>9}")
print(f"  {'-'*50}")
for year in range(2019, 2027):
    sub = [e for e in events if e["date"].year == year]
    if not sub:
        continue
    alphas = []
    for ev in sub:
        cf = ev["closes_fwd"]
        sf = ev["spy_fwd"]
        fwd = 21
        if len(cf) > fwd and cf[0] > 0 and sf[0] > 0:
            alphas.append(cf[fwd]/cf[0] - sf[fwd]/sf[0])
    if not alphas:
        continue
    alphas = np.array(alphas)
    print(f"  {year:>6}  {len(sub):>5}  {np.mean(alphas)*100:>+9.1f}%  "
          f"{np.median(alphas)*100:>+9.1f}%  "
          f"{np.mean(alphas>0)*100:>8.1f}%")

# ── 強彈幅度分層（排除 COVID） ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  強彈幅度分層 × 21天 alpha（排除 COVID）")
print(f"{'='*65}")
buckets = [(0.08,0.12,"+8~12%"),(0.12,0.20,"+12~20%"),(0.20,1.0,"+20%+")]
for lo, hi, lb in buckets:
    sub = [e for e in events_no_covid if lo <= e["day_ret"] < hi]
    if not sub:
        continue
    alphas = [e["closes_fwd"][21]/e["closes_fwd"][0] - e["spy_fwd"][21]/e["spy_fwd"][0]
              for e in sub if len(e["closes_fwd"]) > 21]
    if not alphas:
        continue
    alphas = np.array(alphas)
    print(f"  {lb}: N={len(sub):>4}  21天平均alpha={np.mean(alphas)*100:>+6.1f}%  "
          f"中位={np.median(alphas)*100:>+6.1f}%  "
          f"alpha>0: {np.mean(alphas>0)*100:.1f}%")

# ── 總結 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  結論摘要")
print(f"{'='*65}")

# 計算全期 21天 alpha 核心數字
def alpha21(group):
    a = [e["closes_fwd"][21]/e["closes_fwd"][0] - e["spy_fwd"][21]/e["spy_fwd"][0]
         for e in group if len(e["closes_fwd"]) > 21 and e["closes_fwd"][0] > 0 and e["spy_fwd"][0] > 0]
    return np.array(a)

a_all   = alpha21(events)
a_nocov = alpha21(events_no_covid)
c_all   = alpha21(controls)
c_nocov = alpha21(controls_no_covid)

print(f"  強彈 21天 alpha   全期: {np.mean(a_all)*100:>+.1f}%  排除COVID: {np.mean(a_nocov)*100:>+.1f}%")
print(f"  對照 21天 alpha   全期: {np.mean(c_all)*100:>+.1f}%  排除COVID: {np.mean(c_nocov)*100:>+.1f}%")
print(f"  強彈 vs 對照 差距  全期: {(np.mean(a_all)-np.mean(c_all))*100:>+.1f}%  "
      f"排除COVID: {(np.mean(a_nocov)-np.mean(c_nocov))*100:>+.1f}%")
print()
print(f"  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股")
print(f"  ⚠️  不含交易成本與停損摩擦")
