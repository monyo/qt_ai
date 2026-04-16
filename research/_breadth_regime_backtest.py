"""
市場廣度 Regime 過濾回測

現行系統只用 SPY vs MA200 判斷牛熊，但更靈敏的警示是：
  廣度 = S&P500 中高於 50MA 的股票比例
  廣度 < 50% → 市場進入分配階段，ADD 應更保守

測試四種策略：
  A（基準）：top5 mom_mixed，不管廣度（現行）
  B（廣度過濾）：廣度 < 50% 時降為 top3，廣度 ≥ 50% 選 top5
  C（嚴格過濾）：廣度 < 40% 時不選（維持現金），否則 top5
  D（廣度加權）：廣度越高，選越多（廣度×5 取整，最少 1 最多 5）

資料：S&P500 2021-2025，月度再平衡
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

CACHE_PATH   = "data/_protection_bt_prices.pkl"
MOM_SHORT    = 21
MOM_LONG     = 252
MA_WIN       = 50     # 廣度用 50MA
TOP_N        = 5
BREADTH_MOD  = 0.50   # 策略 B：廣度 < 此值降為 top3
BREADTH_STOP = 0.40   # 策略 C：廣度 < 此值不選股

# ── 載入資料 ──────────────────────────────────────────────────────────
if not os.path.exists(CACHE_PATH):
    print(f"找不到快取 {CACHE_PATH}")
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


# ── 預計算每日廣度 ────────────────────────────────────────────────────
print("計算每日市場廣度（% 股票 > 50MA）...")

# 計算每檔的 50MA
ma50 = prices.rolling(MA_WIN, min_periods=30).mean()
above_ma50 = (prices > ma50).astype(int)
# 廣度 = 每日有效股票中，高於 50MA 的比例
valid_count = (prices.notna() & ma50.notna()).sum(axis=1)
above_count = above_ma50.sum(axis=1)
breadth_series = (above_count / valid_count).where(valid_count > 100)
breadth_series.index = trading_days

print(f"廣度資料：{breadth_series.dropna().shape[0]} 個交易日\n")


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

def get_breadth(ti):
    b = breadth_series.iloc[ti] if ti < len(breadth_series) else None
    return float(b) if b is not None and pd.notna(b) else None

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


# ── 主回測循環 ─────────────────────────────────────────────────────────
rebalance_dates = pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")
results = {k: [] for k in ["A", "B", "C", "D"]}
breadth_log = []

print("回測中...")
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG:
        continue

    breadth = get_breadth(ti)
    if breadth is None:
        continue

    breadth_log.append({"date": str(ref.date()), "breadth": breadth})

    # 全市場動能排名
    pool = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, ti)
        if m is None or m <= 0:
            continue
        pool.append({"sym": sym, "mom": m})

    if len(pool) < TOP_N:
        continue

    df_pool = pd.DataFrame(pool).nlargest(20, "mom")  # 取前20備用

    # 各策略的 top_n
    n_B = 3 if breadth < BREADTH_MOD else TOP_N
    n_C = 0 if breadth < BREADTH_STOP else TOP_N   # 0 = 不選股
    n_D = max(1, min(TOP_N, round(breadth * TOP_N)))  # 廣度×5，1~5之間

    picks = {
        "A": df_pool.head(TOP_N)["sym"].tolist(),
        "B": df_pool.head(n_B)["sym"].tolist(),
        "C": df_pool.head(n_C)["sym"].tolist() if n_C > 0 else [],
        "D": df_pool.head(n_D)["sym"].tolist(),
    }

    spy_1m = spy_fwd(ref.date(), 21)
    spy_3m = spy_fwd(ref.date(), 63)

    for key, syms in picks.items():
        if not syms:
            # 策略 C 不選股：記錄 alpha=0（保持現金，相當於 SPY 的負 alpha）
            results[key].append({
                "date": str(ref.date()), "picks": [],
                "avg_1m": 0.0, "avg_3m": 0.0,
                "alpha_1m": (0.0 - spy_1m) if spy_1m else None,
                "alpha_3m": (0.0 - spy_3m) if spy_3m else None,
                "breadth": breadth, "n_picks": 0,
            })
            continue

        fwd_1m = [x for x in [fwd_return_sym(s, ti, 21) for s in syms] if x is not None]
        fwd_3m = [x for x in [fwd_return_sym(s, ti, 63) for s in syms] if x is not None]
        if not fwd_1m:
            continue
        avg_1m = np.mean(fwd_1m)
        avg_3m = np.mean(fwd_3m) if fwd_3m else None
        results[key].append({
            "date": str(ref.date()), "picks": syms,
            "avg_1m":  avg_1m, "avg_3m":  avg_3m,
            "alpha_1m": avg_1m - spy_1m if spy_1m else None,
            "alpha_3m": avg_3m - spy_3m if (avg_3m and spy_3m) else None,
            "breadth": breadth, "n_picks": len(syms),
        })

print()

# ── 彙總輸出 ──────────────────────────────────────────────────────────
print("=" * 72)
print("  市場廣度 Regime 回測  2021-2025  S&P500 月度")
print("=" * 72)

# 廣度統計
df_bl = pd.DataFrame(breadth_log)
print(f"\n  廣度統計（{len(df_bl)} 個月）")
print(f"    平均: {df_bl['breadth'].mean()*100:.0f}%  "
      f"最低: {df_bl['breadth'].min()*100:.0f}%  "
      f"最高: {df_bl['breadth'].max()*100:.0f}%")
print(f"    廣度 < 40% 月數: {(df_bl['breadth'] < 0.40).sum()}  "
      f"廣度 < 50% 月數: {(df_bl['breadth'] < 0.50).sum()}  "
      f"廣度 ≥ 50% 月數: {(df_bl['breadth'] >= 0.50).sum()}")

desc = {
    "A": "A. 純動能（基準）          top5，不管廣度",
    "B": "B. 廣度過濾（寬鬆）        廣度<50%→top3，否則top5",
    "C": "C. 廣度過濾（嚴格）        廣度<40%→不選，否則top5",
    "D": "D. 廣度加權                選股數 = round(廣度×5)",
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
    avg_n    = df["n_picks"].mean()
    summary[key] = {"alpha_1m": alpha_1m, "alpha_3m": alpha_3m}
    print(f"\n  {desc[key]}")
    print(f"    平均 1M 報酬: {avg_1m:+.2f}%   3M: {avg_3m:+.2f}%   (n={n})")
    print(f"    vs SPY alpha: 1M {alpha_1m:+.2f}%   3M {alpha_3m:+.2f}%")
    print(f"    月勝率(>0): {win_1m:.0f}%   月跑贏SPY: {win_a:.0f}%")
    print(f"    平均選股數: {avg_n:.1f}")

print()
print("=" * 72)
print("  相對基準 A 的改善量")
print(f"  {'策略':<46} {'1M alpha 差':>12} {'3M alpha 差':>12}")
print("  " + "-" * 72)
for key in ["B", "C", "D"]:
    if key in summary and "A" in summary:
        d1 = summary[key]["alpha_1m"] - summary["A"]["alpha_1m"]
        d3 = summary[key]["alpha_3m"] - summary["A"]["alpha_3m"]
        print(f"  {desc[key]:<46} {d1:>+11.2f}%  {d3:>+11.2f}%")

# ── 廣度分組：高廣度 vs 低廣度的報酬差異 ─────────────────────────────
print()
print("=" * 72)
print("  廣度分組分析（策略 A，按當月廣度高低分組）")
print("  核心問題：廣度高的月份，動能選股報酬是否更好？")
print()

df_A = pd.DataFrame(results["A"])
bins   = [0, 0.35, 0.45, 0.55, 0.65, 1.01]
labels = ["<35%（嚴重不健康）", "35-45%（偏弱）", "45-55%（中性）", "55-65%（偏強）", ">65%（強健）"]
df_A["bucket"] = pd.cut(df_A["breadth"], bins=bins, labels=labels)

print(f"  {'廣度區間':<22} {'1M報酬':>9} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 58)
for label in labels:
    sub = df_A[df_A["bucket"] == label]
    if len(sub) == 0:
        continue
    r1 = sub["avg_1m"].mean()
    a1 = sub["alpha_1m"].dropna().mean()
    a3 = sub["alpha_3m"].dropna().mean()
    print(f"  {label:<22} {r1:>+8.2f}%  {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")

# ── 逐月廣度 + alpha 一覽 ────────────────────────────────────────────
print()
print("=" * 72)
print(f"  逐月廣度 + 1M Alpha")
print(f"  {'日期':<12} {'廣度':>7} {'A 基準':>8} {'B 過濾':>8} {'C 嚴格':>8} {'D 加權':>8}  A最佳?")
print("  " + "-" * 68)

by_date = {}
for key in ["A", "B", "C", "D"]:
    for row in results[key]:
        d = row["date"]
        if d not in by_date:
            by_date[d] = {"breadth": row.get("breadth")}
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
    b = by_date[d].get("breadth")
    b_str = f"{b*100:.0f}%" if b is not None else " N/A"
    a_best = "✓" if best == "A" else " "
    print(f"  {d:<12} {b_str:>7} {fmt(vals['A']):>8} {fmt(vals['B']):>8} "
          f"{fmt(vals['C']):>8} {fmt(vals['D']):>8}  {a_best}")

print()
print(f"  各策略最佳月數：" +
      "  ".join(f"{k}={v}" for k, v in win_count.items()))
