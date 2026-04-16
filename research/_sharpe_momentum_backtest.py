"""
Sharpe 動能回測

核心假設：相同動能下，波動率低的股票風險調整後報酬更好。
把 mom_mixed 除以 60 日年化波動率，得到「每單位風險的動能」。

測試四種策略：
  A（基準）：top 5 by mom_mixed（現行做法）
  B（Sharpe）：top 5 by sharpe_mom = mom_mixed / vol_60d
  C（過濾）：top 5 by mom_mixed，但排除年化波動 > 60% 的股票
  D（兩段）：先取 mom_mixed 前 10，再取 sharpe_mom 最高的 5

資料：S&P500 2021-2025，每月選股，對比 SPY 1M/3M alpha
快取：data/_protection_bt_prices.pkl（與其他回測共用）
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_PATH  = "data/_protection_bt_prices.pkl"
MOM_SHORT   = 21
MOM_LONG    = 252
VOL_WIN     = 60    # 波動率計算窗口（交易日）
VOL_MIN_OBS = 30    # 最少需要幾筆才算有效
VOL_CUTOFF  = 0.60  # 策略 C：排除年化波動 > 60%
TOP_N       = 5
POOL_D      = 10

# ── 載入價格資料 ──────────────────────────────────────────────────────
if not os.path.exists(CACHE_PATH):
    print(f"找不到快取 {CACHE_PATH}，請先執行 _protection_period_backtest.py")
    raise SystemExit(1)

print(f"讀取快取：{CACHE_PATH}")
prices = pd.read_pickle(CACHE_PATH)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()
print(f"資料：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日\n")

# SPY 基準
spy = yf.Ticker("SPY").history(start="2020-12-01", end="2026-03-01",
                               auto_adjust=True)["Close"]
spy.index = pd.to_datetime(spy.index).tz_localize(None)


# ── 工具函式 ──────────────────────────────────────────────────────────
def tidx(date) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(date)))

def price_at(sym, ti):
    if ti < 0 or ti >= len(trading_days):
        return None
    v = prices.iloc[ti].get(sym)
    return float(v) if v is not None and pd.notna(v) else None

def mom_mixed_at(sym, ti):
    if ti < MOM_LONG:
        return None
    p0   = price_at(sym, ti)
    p21  = price_at(sym, ti - MOM_SHORT)
    p252 = price_at(sym, ti - MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
        return None
    return 0.5 * (p0/p21 - 1) * 100 + 0.5 * (p0/p252 - 1) * 100

def vol_at(sym, ti):
    """60 日年化波動率（0.30 = 30%）"""
    if ti < VOL_WIN:
        return None
    seg = prices.iloc[ti - VOL_WIN: ti][sym].dropna()
    if len(seg) < VOL_MIN_OBS:
        return None
    log_ret = np.log(seg / seg.shift(1)).dropna()
    if len(log_ret) < VOL_MIN_OBS:
        return None
    return float(log_ret.std() * np.sqrt(252))

def fwd_return_sym(sym, ti, offset):
    p0 = price_at(sym, ti)
    p1 = price_at(sym, ti + offset)
    if p0 and p1 and p0 > 0:
        return (p1 / p0 - 1) * 100
    return None

def spy_fwd(ref_date, offset):
    s = spy.dropna()
    ti = s.index.searchsorted(pd.Timestamp(ref_date))
    if ti + offset >= len(s):
        return None
    return (float(s.iloc[ti + offset]) / float(s.iloc[ti]) - 1) * 100


# ── 主回測循環 ────────────────────────────────────────────────────────
rebalance_dates = pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")
results = {k: [] for k in ["A", "B", "C", "D"]}

print("回測中...")
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG + VOL_WIN:
        continue

    pool = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, ti)
        if m is None or m <= 0:
            continue
        v = vol_at(sym, ti)
        if v is None or v <= 0:
            continue
        pool.append({
            "sym":        sym,
            "mom":        m,
            "vol":        v,
            "sharpe_mom": m / (v * 100),   # 動能(%) / 波動率(%)
        })

    if len(pool) < POOL_D:
        continue

    df_pool = pd.DataFrame(pool)
    picks = {}

    # A：純動能（基準）
    picks["A"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()

    # B：Sharpe 動能
    picks["B"] = df_pool.nlargest(TOP_N, "sharpe_mom")["sym"].tolist()

    # C：純動能但排除高波動
    df_c = df_pool[df_pool["vol"] <= VOL_CUTOFF]
    if len(df_c) >= TOP_N:
        picks["C"] = df_c.nlargest(TOP_N, "mom")["sym"].tolist()
    else:
        picks["C"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()  # fallback

    # D：先取前 10 mom，再取 sharpe_mom 最高的 5
    top10 = df_pool.nlargest(POOL_D, "mom")
    picks["D"] = top10.nlargest(TOP_N, "sharpe_mom")["sym"].tolist()

    spy_1m = spy_fwd(ref.date(), 21)
    spy_3m = spy_fwd(ref.date(), 63)

    for key, syms in picks.items():
        fwd_1m = [fwd_return_sym(s, ti, 21) for s in syms]
        fwd_3m = [fwd_return_sym(s, ti, 63) for s in syms]
        fwd_1m = [x for x in fwd_1m if x is not None]
        fwd_3m = [x for x in fwd_3m if x is not None]
        if not fwd_1m:
            continue
        avg_1m = np.mean(fwd_1m)
        avg_3m = np.mean(fwd_3m) if fwd_3m else None
        sub = df_pool[df_pool["sym"].isin(syms)]
        results[key].append({
            "date":      str(ref.date()),
            "picks":     syms,
            "avg_1m":    avg_1m,
            "avg_3m":    avg_3m,
            "alpha_1m":  avg_1m - spy_1m if spy_1m else None,
            "alpha_3m":  avg_3m - spy_3m if (avg_3m and spy_3m) else None,
            "avg_vol":   sub["vol"].mean(),
            "avg_sharpe_mom": sub["sharpe_mom"].mean(),
        })

print()

# ── 彙總輸出 ──────────────────────────────────────────────────────────
print("=" * 72)
print("  Sharpe 動能回測  2021-2025  S&P500 月度")
print("=" * 72)

desc = {
    "A": "A. 純動能（基準）        top5 mom_mixed",
    "B": "B. Sharpe 動能           top5 mom/vol",
    "C": "C. 過濾高波動            top5 mom，排除 vol>60%",
    "D": "D. 兩段篩選              top10 mom → top5 sharpe",
}

summary = {}
for key in ["A", "B", "C", "D"]:
    rows = results[key]
    if not rows:
        continue
    df = pd.DataFrame(rows)
    n        = len(df)
    avg_1m   = df["avg_1m"].mean()
    avg_3m   = df["avg_3m"].dropna().mean()
    alpha_1m = df["alpha_1m"].dropna().mean()
    alpha_3m = df["alpha_3m"].dropna().mean()
    win_1m   = (df["avg_1m"] > 0).sum() / n * 100
    win_a    = (df["alpha_1m"].dropna() > 0).sum() / n * 100
    avg_vol  = df["avg_vol"].mean()
    avg_sm   = df["avg_sharpe_mom"].mean()
    summary[key] = {"alpha_1m": alpha_1m, "alpha_3m": alpha_3m, "avg_vol": avg_vol}
    print(f"\n  {desc[key]}")
    print(f"    平均 1M 報酬: {avg_1m:+.2f}%   3M: {avg_3m:+.2f}%   (n={n})")
    print(f"    vs SPY alpha: 1M {alpha_1m:+.2f}%   3M {alpha_3m:+.2f}%")
    print(f"    月勝率(>0): {win_1m:.0f}%   月跑贏SPY: {win_a:.0f}%")
    print(f"    選入時平均波動率: {avg_vol*100:.0f}%/年   sharpe_mom: {avg_sm:.2f}")

# vs 基準差距
print()
print("=" * 72)
print("  相對基準 A 的改善量")
print(f"  {'策略':<42} {'波動率差':>9} {'1M alpha 差':>12} {'3M alpha 差':>12}")
print("  " + "-" * 76)
for key in ["B", "C", "D"]:
    if key in summary and "A" in summary:
        d1   = summary[key]["alpha_1m"] - summary["A"]["alpha_1m"]
        d3   = summary[key]["alpha_3m"] - summary["A"]["alpha_3m"]
        dvol = (summary[key]["avg_vol"] - summary["A"]["avg_vol"]) * 100
        print(f"  {desc[key]:<42} {dvol:>+8.1f}%  {d1:>+11.2f}%  {d3:>+11.2f}%")

# 逐月明細
print()
print("=" * 72)
print(f"  逐月 1M Alpha（vs SPY）")
print(f"  {'日期':<12} {'A 基準':>8} {'B Sharpe':>9} {'C 過濾':>8} {'D 兩段':>8}  最佳")
print("  " + "-" * 62)

by_date = {}
for key in ["A", "B", "C", "D"]:
    for row in results[key]:
        d = row["date"]
        if d not in by_date:
            by_date[d] = {}
        by_date[d][key] = row["alpha_1m"]

def fmt(v):
    return f"{v:+.1f}%" if v is not None else "  N/A "

win_count = {"A": 0, "B": 0, "C": 0, "D": 0}
for d in sorted(by_date.keys()):
    vals = {k: by_date[d].get(k) for k in ["A", "B", "C", "D"]}
    valid = {k: v for k, v in vals.items() if v is not None}
    best = max(valid, key=valid.get) if valid else "-"
    if best in win_count:
        win_count[best] += 1
    print(f"  {d:<12} {fmt(vals['A']):>8} {fmt(vals['B']):>9} "
          f"{fmt(vals['C']):>8} {fmt(vals['D']):>8}  {best}")

print()
print(f"  各策略最佳月數：" +
      "  ".join(f"{k}={v}" for k, v in win_count.items()))

# ── 波動率分組分析 ─────────────────────────────────────────────────────
print()
print("=" * 72)
print("  波動率分組分析（策略 A 的選股，按 vol 分成高/低兩半）")
print("  驗證：在動能相近時，低波動的那批是否報酬更好、風險更低？")
print()

all_A_events = []
for row in results["A"]:
    ti = tidx(row["date"])
    for sym in row["picks"]:
        m  = mom_mixed_at(sym, ti)
        v  = vol_at(sym, ti)
        r1 = fwd_return_sym(sym, ti, 21)
        r3 = fwd_return_sym(sym, ti, 63)
        s1 = spy_fwd(row["date"], 21)
        s3 = spy_fwd(row["date"], 63)
        if None in (m, v, r1, s1):
            continue
        all_A_events.append({
            "date": row["date"], "sym": sym,
            "mom": m, "vol": v,
            "r1": r1, "r3": r3,
            "alpha_1m": r1 - s1,
            "alpha_3m": (r3 - s3) if (r3 and s3) else None,
        })

df_ev = pd.DataFrame(all_A_events)
med_vol = df_ev["vol"].median()
df_hi = df_ev[df_ev["vol"] >= med_vol]
df_lo = df_ev[df_ev["vol"] <  med_vol]

print(f"  中位數波動率：{med_vol*100:.0f}%/年")
print(f"  {'分組':<20} {'1M報酬':>9} {'3M報酬':>9} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 64)
for label, sub in [("vol ≥ 中位（高波動）", df_hi), ("vol < 中位（低波動）", df_lo)]:
    r1 = sub["r1"].mean()
    r3 = sub["r3"].dropna().mean()
    a1 = sub["alpha_1m"].dropna().mean()
    a3 = sub["alpha_3m"].dropna().mean()
    print(f"  {label:<20} {r1:>+8.2f}%  {r3:>+8.2f}%  {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")

# ── 波動率 vs 動能四象限 ───────────────────────────────────────────────
print()
print("=" * 72)
print("  四象限分析：高動能 × 低波動 是否是最佳組合？")
print()

med_mom = df_ev["mom"].median()
quad = {
    "高動能 低波動": df_ev[(df_ev["mom"] >= med_mom) & (df_ev["vol"] < med_vol)],
    "高動能 高波動": df_ev[(df_ev["mom"] >= med_mom) & (df_ev["vol"] >= med_vol)],
    "低動能 低波動": df_ev[(df_ev["mom"] < med_mom)  & (df_ev["vol"] < med_vol)],
    "低動能 高波動": df_ev[(df_ev["mom"] < med_mom)  & (df_ev["vol"] >= med_vol)],
}

print(f"  {'象限':<16} {'1M報酬':>9} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 52)
for label, sub in quad.items():
    if len(sub) == 0:
        continue
    r1 = sub["r1"].mean()
    a1 = sub["alpha_1m"].dropna().mean()
    a3 = sub["alpha_3m"].dropna().mean()
    print(f"  {label:<16} {r1:>+8.2f}%  {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")
