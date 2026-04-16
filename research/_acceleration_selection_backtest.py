"""
加速度輔助選股回測

現行系統：top 5 by mom_mixed (50%×21d + 50%×252d)
實驗問題：在選股時加入「加速度」這個維度，能否找到報酬更好的標的？

測試四種策略：
  A（基準）：top 5 by mom_mixed（現行做法）
  B（過濾）：top 5 by mom_mixed，但排除 acc_pre < 0（只選動能仍在加速的）
  C（複合）：top 5 by mom_mixed + 0.4 × acc_pre（複合分數）
  D（兩段）：先取 mom_mixed 前 10，再從中取 acc_pre 最高的 5 支

資料：S&P500 2021-2025，每月選股，對比 SPY 1M/3M alpha
快取：data/_protection_bt_prices.pkl（與保護期回測共用）
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
ACC_PRE_WIN = 10    # 選股前 10 個交易日的加速度均值
TOP_N       = 5
POOL_D      = 10    # 策略 D：第一段取前 10

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
    r21  = (p0 / p21  - 1) * 100
    r252 = (p0 / p252 - 1) * 100
    return 0.5 * r21 + 0.5 * r252

def acc_pre_at(sym, ti, window=ACC_PRE_WIN):
    """ti 當日往前 window 天的加速度均值（選股前的動能變化方向）"""
    accels = []
    for d in range(window):
        idx = ti - d               # 往前數
        if idx < MOM_SHORT * 2:
            continue
        p0  = price_at(sym, idx)
        p21 = price_at(sym, idx - MOM_SHORT)
        p42 = price_at(sym, idx - MOM_SHORT * 2)
        if not (p0 and p21 and p42 and p21 > 0 and p42 > 0):
            continue
        r_now  = (p0  / p21 - 1) * 100
        r_prev = (p21 / p42 - 1) * 100
        accels.append(r_now - r_prev)
    return float(np.mean(accels)) if len(accels) >= window // 2 else None

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
    if ti < MOM_LONG + ACC_PRE_WIN:
        continue

    # 全市場指標
    pool = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, ti)
        if m is None or m <= 0:
            continue
        a = acc_pre_at(sym, ti)
        if a is None:
            continue
        pool.append({"sym": sym, "mom": m, "acc": a,
                     "composite": m + 0.4 * a})

    if len(pool) < POOL_D:
        continue

    df_pool = pd.DataFrame(pool)

    # ── 四種選股策略 ──────────────────────────────────────────────────
    picks = {}

    # A：純動能（基準）
    picks["A"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()

    # B：純動能但排除加速度 < 0
    df_b = df_pool[df_pool["acc"] >= 0]
    if len(df_b) >= TOP_N:
        picks["B"] = df_b.nlargest(TOP_N, "mom")["sym"].tolist()
    else:
        picks["B"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()  # fallback

    # C：複合分數（mom + 0.4 × acc）
    picks["C"] = df_pool.nlargest(TOP_N, "composite")["sym"].tolist()

    # D：先取前 10 mom，再取 acc 最高的 5
    top10 = df_pool.nlargest(POOL_D, "mom")
    picks["D"] = top10.nlargest(TOP_N, "acc")["sym"].tolist()

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
        results[key].append({
            "date":     str(ref.date()),
            "picks":    syms,
            "avg_1m":   avg_1m,
            "avg_3m":   avg_3m,
            "alpha_1m": avg_1m - spy_1m if spy_1m else None,
            "alpha_3m": avg_3m - spy_3m if (avg_3m and spy_3m) else None,
            "avg_acc":  df_pool[df_pool["sym"].isin(syms)]["acc"].mean(),
        })

print()

# ── 彙總輸出 ──────────────────────────────────────────────────────────
print("=" * 70)
print("  加速度輔助選股回測  2021-2025  S&P500 月度")
print("=" * 70)

desc = {
    "A": "A. 純動能（基準）          top5 mom_mixed",
    "B": "B. 過濾減速股              top5 mom，排除 acc<0",
    "C": "C. 複合分數                top5 (mom + 0.4×acc)",
    "D": "D. 兩段篩選                top10 mom → top5 acc",
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
    avg_acc  = df["avg_acc"].mean()
    summary[key] = {"alpha_1m": alpha_1m, "alpha_3m": alpha_3m}
    print(f"\n  {desc[key]}")
    print(f"    平均 1M 報酬: {avg_1m:+.2f}%   3M: {avg_3m:+.2f}%   (n={n})")
    print(f"    vs SPY alpha: 1M {alpha_1m:+.2f}%   3M {alpha_3m:+.2f}%")
    print(f"    月勝率(>0): {win_1m:.0f}%   月跑贏SPY: {win_a:.0f}%")
    print(f"    選入時平均 acc_pre: {avg_acc:+.1f}%pts")

# vs 基準差距
print()
print("=" * 70)
print("  相對基準 A 的改善量")
print(f"  {'策略':<40} {'1M alpha 差':>12} {'3M alpha 差':>12}")
print("  " + "-" * 66)
for key in ["B", "C", "D"]:
    if key in summary and "A" in summary:
        d1 = summary[key]["alpha_1m"] - summary["A"]["alpha_1m"]
        d3 = summary[key]["alpha_3m"] - summary["A"]["alpha_3m"]
        print(f"  {desc[key]:<40} {d1:>+11.2f}%  {d3:>+11.2f}%")

# ── 逐月明細 ──────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  逐月 1M Alpha（vs SPY）")
print(f"  {'日期':<12} {'A 基準':>8} {'B 過濾':>8} {'C 複合':>8} {'D 兩段':>8}  最佳")
print("  " + "-" * 60)

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
    print(f"  {d:<12} {fmt(vals['A']):>8} {fmt(vals['B']):>8} "
          f"{fmt(vals['C']):>8} {fmt(vals['D']):>8}  {best}")

print()
print(f"  各策略最佳月數：" +
      "  ".join(f"{k}={v}" for k, v in win_count.items()))

# ── acc 分組分析：acc 高 vs 低的報酬差異 ─────────────────────────────
print()
print("=" * 70)
print("  acc_pre 分組分析（策略 A 的選股，按 acc 分成高/低兩半）")
print("  驗證：在動能相近時，加速度高的那批是否報酬更好？")
print()

all_A_events = []
for row in results["A"]:
    ti = tidx(row["date"])
    for sym in row["picks"]:
        m = mom_mixed_at(sym, ti)
        a = acc_pre_at(sym, ti)
        r1 = fwd_return_sym(sym, ti, 21)
        r3 = fwd_return_sym(sym, ti, 63)
        spy_1m = spy_fwd(row["date"], 21)
        spy_3m = spy_fwd(row["date"], 63)
        if None in (m, a, r1, spy_1m):
            continue
        all_A_events.append({
            "date": row["date"], "sym": sym,
            "mom": m, "acc": a,
            "r1": r1, "r3": r3,
            "alpha_1m": r1 - spy_1m,
            "alpha_3m": (r3 - spy_3m) if (r3 and spy_3m) else None,
        })

df_ev = pd.DataFrame(all_A_events)
median_acc = df_ev["acc"].median()
df_hi = df_ev[df_ev["acc"] >= median_acc]
df_lo = df_ev[df_ev["acc"] <  median_acc]

print(f"  中位數加速度：{median_acc:+.1f}%pts")
print(f"  {'分組':<20} {'1M報酬':>9} {'3M報酬':>9} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 62)
for label, sub in [("acc ≥ 中位（高）", df_hi), ("acc < 中位（低）", df_lo)]:
    r1 = sub["r1"].mean()
    r3 = sub["r3"].dropna().mean()
    a1 = sub["alpha_1m"].dropna().mean()
    a3 = sub["alpha_3m"].dropna().mean()
    print(f"  {label:<20} {r1:>+8.2f}%  {r3:>+8.2f}%  {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")
