"""
52 週高點接近度回測

學術文獻（George & Hwang 2004）發現：接近 52 週高點的股票，
投資人因「錨定效應」猶豫不買，突破後反而解放賣壓、加速上漲。
與直覺相反——「快要創新高」比「已大漲一段」的股票未來報酬更好。

測試四種策略：
  A（基準）：top5 mom_mixed（現行做法）
  B（高點接近）：top5 by (mom_mixed × near52_score)
                near52_score = price / max(past 252d)，值域 (0, 1]
                              接近 1.0 = 快突破高點
  C（兩段篩選）：top10 mom → top5 by near52_score
  D（高點過濾）：top5 mom，但只選 near52 > 0.90 的（距高點 < 10%）

資料：S&P500 2021-2025，月度再平衡
快取：data/_protection_bt_prices.pkl（與其他回測共用）
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

CACHE_PATH = "data/_protection_bt_prices.pkl"
MOM_SHORT  = 21
MOM_LONG   = 252
TOP_N      = 5
POOL_SIZE  = 10       # 策略 C/D 的第一段候選數
NEAR52_THR = 0.90     # 策略 D：距高點 < 10%

# ── 載入資料 ──────────────────────────────────────────────────────────
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
    return 0.5 * (p0/p21 - 1)*100 + 0.5 * (p0/p252 - 1)*100

def near52_at(sym, ti):
    """當前價格 / 過去 252 日最高價，值域 (0, 1]。越接近 1 = 越近高點。"""
    if ti < MOM_LONG:
        return None
    p0 = price_at(sym, ti)
    seg = prices.iloc[ti - MOM_LONG: ti][sym].dropna()
    if p0 is None or len(seg) < 100:
        return None
    high_52w = float(seg.max())
    if high_52w <= 0:
        return None
    return p0 / high_52w

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
    if ti < MOM_LONG:
        continue

    pool = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, ti)
        if m is None or m <= 0:
            continue
        n52 = near52_at(sym, ti)
        if n52 is None:
            continue
        pool.append({
            "sym":       sym,
            "mom":       m,
            "near52":    n52,
            "composite": m * n52,   # 動能 × 高點接近度
        })

    if len(pool) < POOL_SIZE:
        continue

    df_pool = pd.DataFrame(pool)
    top10   = df_pool.nlargest(POOL_SIZE, "mom")
    picks   = {}

    # A：純動能（基準）
    picks["A"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()

    # B：複合分數 mom × near52
    picks["B"] = df_pool.nlargest(TOP_N, "composite")["sym"].tolist()

    # C：top10 mom → top5 near52（距高點最近的）
    picks["C"] = top10.nlargest(TOP_N, "near52")["sym"].tolist()

    # D：top5 mom，但只選 near52 > 0.90
    d_near = df_pool[df_pool["near52"] >= NEAR52_THR]
    if len(d_near) >= TOP_N:
        picks["D"] = d_near.nlargest(TOP_N, "mom")["sym"].tolist()
    else:
        picks["D"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()  # fallback

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
            "date":       str(ref.date()),
            "picks":      syms,
            "avg_1m":     avg_1m,
            "avg_3m":     avg_3m,
            "alpha_1m":   avg_1m - spy_1m if spy_1m else None,
            "alpha_3m":   avg_3m - spy_3m if (avg_3m and spy_3m) else None,
            "avg_near52": sub["near52"].mean(),
        })

print()

# ── 彙總輸出 ──────────────────────────────────────────────────────────
print("=" * 72)
print("  52 週高點接近度回測  2021-2025  S&P500 月度")
print("=" * 72)

desc = {
    "A": "A. 純動能（基準）        top5 mom_mixed",
    "B": "B. 複合分數              top5 (mom × near52)",
    "C": "C. 兩段篩選              top10 mom → top5 near52",
    "D": f"D. 高點過濾              top5 mom，near52 ≥ {NEAR52_THR}",
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
    avg_n52  = df["avg_near52"].mean()
    summary[key] = {"alpha_1m": alpha_1m, "alpha_3m": alpha_3m, "avg_near52": avg_n52}
    print(f"\n  {desc[key]}")
    print(f"    平均 1M 報酬: {avg_1m:+.2f}%   3M: {avg_3m:+.2f}%   (n={n})")
    print(f"    vs SPY alpha: 1M {alpha_1m:+.2f}%   3M {alpha_3m:+.2f}%")
    print(f"    月勝率(>0): {win_1m:.0f}%   月跑贏SPY: {win_a:.0f}%")
    print(f"    選入時平均 near52: {avg_n52:.3f}  (距高點 {(1-avg_n52)*100:.1f}%)")

print()
print("=" * 72)
print("  相對基準 A 的改善量")
print(f"  {'策略':<44} {'near52 差':>10} {'1M alpha 差':>12} {'3M alpha 差':>12}")
print("  " + "-" * 80)
for key in ["B", "C", "D"]:
    if key in summary and "A" in summary:
        d1   = summary[key]["alpha_1m"] - summary["A"]["alpha_1m"]
        d3   = summary[key]["alpha_3m"] - summary["A"]["alpha_3m"]
        dn52 = summary[key]["avg_near52"] - summary["A"]["avg_near52"]
        print(f"  {desc[key]:<44} {dn52:>+9.3f}  {d1:>+11.2f}%  {d3:>+11.2f}%")

# ── near52 分組分析 ───────────────────────────────────────────────────
print()
print("=" * 72)
print("  near52 分組分析（策略 A 的選股，按接近度分成五組）")
print("  驗證：越接近 52 週高點，未來報酬是否越高？")
print()

all_events = []
for row in results["A"]:
    ti = tidx(row["date"])
    for sym in row["picks"]:
        n52 = near52_at(sym, ti)
        r1  = fwd_return_sym(sym, ti, 21)
        r3  = fwd_return_sym(sym, ti, 63)
        s1  = spy_fwd(row["date"], 21)
        s3  = spy_fwd(row["date"], 63)
        if None in (n52, r1, s1):
            continue
        all_events.append({
            "sym": sym, "date": row["date"],
            "near52": n52,
            "r1": r1, "r3": r3,
            "alpha_1m": r1 - s1,
            "alpha_3m": (r3 - s3) if (r3 and s3) else None,
        })

df_ev = pd.DataFrame(all_events)

bins   = [0, 0.70, 0.80, 0.90, 0.95, 1.01]
labels = ["<70% (距高>30%)", "70-80%", "80-90%", "90-95%", "≥95% (快突破)"]
df_ev["bucket"] = pd.cut(df_ev["near52"], bins=bins, labels=labels)

print(f"  {'near52 區間':<20} {'1M報酬':>9} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 56)
for label in labels:
    sub = df_ev[df_ev["bucket"] == label]
    if len(sub) == 0:
        continue
    r1 = sub["r1"].mean()
    a1 = sub["alpha_1m"].dropna().mean()
    a3 = sub["alpha_3m"].dropna().mean()
    print(f"  {label:<20} {r1:>+8.2f}%  {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")

# ── 新高突破分析 ──────────────────────────────────────────────────────
print()
print("=" * 72)
print("  新高突破分析：near52 ≥ 1.0（當月創 52 週新高）vs 未創新高")
print()

df_new_high  = df_ev[df_ev["near52"] >= 1.0]
df_near_high = df_ev[(df_ev["near52"] >= 0.95) & (df_ev["near52"] < 1.0)]
df_rest      = df_ev[df_ev["near52"] < 0.95]

print(f"  {'分組':<28} {'1M報酬':>9} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 62)
for label, sub in [
    ("創 52W 新高 (≥1.0)",    df_new_high),
    ("接近新高 (0.95-1.0)",   df_near_high),
    ("未接近新高 (<0.95)",    df_rest),
]:
    if len(sub) == 0:
        continue
    r1 = sub["r1"].mean()
    a1 = sub["alpha_1m"].dropna().mean()
    a3 = sub["alpha_3m"].dropna().mean()
    print(f"  {label:<28} {r1:>+8.2f}%  {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")

# ── 逐月明細 ─────────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"  逐月 1M Alpha（vs SPY）")
print(f"  {'日期':<12} {'A 基準':>8} {'B 複合':>8} {'C 兩段':>8} {'D 過濾':>8}  最佳")
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
    print(f"  {d:<12} {fmt(vals['A']):>8} {fmt(vals['B']):>8} "
          f"{fmt(vals['C']):>8} {fmt(vals['D']):>8}  {best}")

print()
print(f"  各策略最佳月數：" +
      "  ".join(f"{k}={v}" for k, v in win_count.items()))
