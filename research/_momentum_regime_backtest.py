"""
動能體制回測：頂端動能 vs 剛冒出來的動能

比較三種選股策略：
  A) 無過濾：混合動能排名前5（現行做法）
  B) 適中：混合動能前5，但 252d < 100%（過濾極端長期動能）
  C) 早期：混合動能前5，但 252d < 50%（只找剛冒頭的）
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.data_loader import get_sp500_tickers

# ── 參數 ──────────────────────────────────────────────────────────
CACHE_PATH = "data/_regime_prices.pkl"
REBALANCE_DATES = [
    "2024-01-02", "2024-02-01", "2024-03-01", "2024-04-01",
    "2024-05-01", "2024-06-03", "2024-07-01", "2024-08-01",
    "2024-09-03", "2024-10-01", "2024-11-01", "2024-12-02",
]
TOP_N = 5
MOM_SHORT = 21
MOM_LONG = 252
EXTREME_THRESHOLD = 100
MODERATE_THRESHOLD = 50
BATCH_SIZE = 50

# ── 下載資料（分批，有快取） ─────────────────────────────────────
if os.path.exists(CACHE_PATH):
    print(f"讀取快取：{CACHE_PATH}")
    prices = pd.read_pickle(CACHE_PATH)
else:
    tickers = get_sp500_tickers()
    print(f"分批下載 {len(tickers)} 檔（每批 {BATCH_SIZE}）...")
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_dfs = []
    for i, batch in enumerate(batches):
        print(f"  批次 {i+1}/{len(batches)}...", end=" ", flush=True)
        try:
            raw = yf.download(batch, start="2022-06-01", end="2025-04-01",
                              auto_adjust=True, progress=False)
            if raw.empty:
                print("空")
                continue
            close = raw["Close"]
            if isinstance(close.columns, pd.MultiIndex):
                close.columns = close.columns.droplevel(0)
            all_dfs.append(close)
            print(f"OK ({close.shape[1]} 檔)")
        except Exception as e:
            print(f"ERROR: {e}")

    prices = pd.concat(all_dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(axis=1, how="all")
    os.makedirs("data", exist_ok=True)
    prices.to_pickle(CACHE_PATH)
    print(f"已快取至 {CACHE_PATH}")

print(f"資料：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日\n")

spy_prices = yf.Ticker("SPY").history(start="2022-06-01", end="2025-04-01",
                                       auto_adjust=True)["Close"]
spy_prices.index = spy_prices.index.tz_localize(None)


# ── 工具函式 ──────────────────────────────────────────────────────
def get_price_at(sym, ref_date, offset=0):
    if sym not in prices.columns:
        return None
    ser = prices[sym].dropna()
    idx = ser.index.searchsorted(pd.Timestamp(ref_date))
    idx += offset
    if idx < 0 or idx >= len(ser):
        return None
    return float(ser.iloc[idx])


def calc_momentum_at(sym, ref_date):
    if sym not in prices.columns:
        return None
    ser = prices[sym].dropna()
    idx = ser.index.searchsorted(pd.Timestamp(ref_date))
    if idx < MOM_LONG:
        return None
    p_now = float(ser.iloc[idx])
    p_21  = float(ser.iloc[idx - MOM_SHORT])
    p_252 = float(ser.iloc[idx - MOM_LONG])
    if p_21 <= 0 or p_252 <= 0:
        return None
    mom_short = (p_now / p_21 - 1) * 100
    mom_long  = (p_now / p_252 - 1) * 100
    mom_mixed = 0.5 * mom_short + 0.5 * mom_long
    return {"mom_short": mom_short, "mom_long": mom_long, "mom_mixed": mom_mixed}


def forward_return(sym, ref_date, days):
    if sym not in prices.columns:
        return None
    ser = prices[sym].dropna()
    idx = ser.index.searchsorted(pd.Timestamp(ref_date))
    end_idx = idx + days
    if end_idx >= len(ser):
        return None
    p0 = float(ser.iloc[idx])
    p1 = float(ser.iloc[end_idx])
    return (p1 / p0 - 1) * 100 if p0 > 0 else None


def spy_fwd(ref_date, days):
    ser = spy_prices.dropna()
    idx = ser.index.searchsorted(pd.Timestamp(ref_date))
    end_idx = idx + days
    if end_idx >= len(ser):
        return None
    return (float(ser.iloc[end_idx]) / float(ser.iloc[idx]) - 1) * 100


# ── 主回測循環 ─────────────────────────────────────────────────────
results = {"A": [], "B": [], "C": []}

print("回測中...")
for ref in REBALANCE_DATES:
    moms = []
    for sym in prices.columns:
        m = calc_momentum_at(sym, ref)
        if m is None or m["mom_mixed"] <= 0:
            continue
        moms.append({"symbol": sym, **m})

    if not moms:
        continue

    df_m = pd.DataFrame(moms).sort_values("mom_mixed", ascending=False)

    pool_A = df_m
    pool_B = df_m[df_m["mom_long"] < EXTREME_THRESHOLD]
    pool_C = df_m[df_m["mom_long"] < MODERATE_THRESHOLD]

    spy_1m = spy_fwd(ref, 21)
    spy_3m = spy_fwd(ref, 63)

    for key, pool in [("A", pool_A), ("B", pool_B), ("C", pool_C)]:
        picks = pool.head(TOP_N)
        fwd_1m = [forward_return(s, ref, 21) for s in picks["symbol"]]
        fwd_3m = [forward_return(s, ref, 63) for s in picks["symbol"]]
        fwd_1m = [x for x in fwd_1m if x is not None]
        fwd_3m = [x for x in fwd_3m if x is not None]
        if fwd_1m:
            avg_1m = np.mean(fwd_1m)
            avg_3m = np.mean(fwd_3m) if fwd_3m else None
            results[key].append({
                "date": ref,
                "picks": picks["symbol"].tolist(),
                "avg_mom_long": picks["mom_long"].mean(),
                "avg_1m": avg_1m,
                "avg_3m": avg_3m,
                "alpha_1m": avg_1m - spy_1m if spy_1m else None,
                "alpha_3m": avg_3m - spy_3m if (avg_3m and spy_3m) else None,
                "pool_size": len(pool),
            })

    spy_str = f"SPY 1M={spy_1m:+.1f}%" if spy_1m else ""
    print(f"  {ref}  候選池 A={len(pool_A):3d} B={len(pool_B):3d} C={len(pool_C):3d}  {spy_str}")

# ── 彙總結果 ───────────────────────────────────────────────────────
print()
print("=" * 65)
print("  2024 年回測結果：動能體制比較（每期選前 5 名）")
print("=" * 65)

desc = {
    "A": f"A. 無過濾（現行做法）  252d=任意",
    "B": f"B. 適中過濾           252d < {EXTREME_THRESHOLD}%",
    "C": f"C. 早期動能           252d < {MODERATE_THRESHOLD}%",
}

summary = {}
for key in ["A", "B", "C"]:
    rows = results[key]
    if not rows:
        print(f"  {desc[key]}：資料不足")
        continue
    df = pd.DataFrame(rows)
    n = len(df)
    avg_1m    = df["avg_1m"].mean()
    avg_3m    = df["avg_3m"].dropna().mean()
    alpha_1m  = df["alpha_1m"].dropna().mean()
    alpha_3m  = df["alpha_3m"].dropna().mean()
    avg_long  = df["avg_mom_long"].mean()
    win_1m    = (df["avg_1m"] > 0).sum() / n * 100
    win_alpha = (df["alpha_1m"].dropna() > 0).sum() / n * 100
    summary[key] = {"avg_1m": avg_1m, "avg_3m": avg_3m,
                    "alpha_1m": alpha_1m, "alpha_3m": alpha_3m}
    print(f"\n  {desc[key]}")
    print(f"    平均 1M 報酬: {avg_1m:+.2f}%   3M: {avg_3m:+.2f}%")
    print(f"    vs SPY alpha: 1M {alpha_1m:+.2f}%   3M {alpha_3m:+.2f}%")
    print(f"    月勝率(>0): {win_1m:.0f}%   月跑贏SPY: {win_alpha:.0f}%")
    print(f"    選入時平均 252d 動能: {avg_long:+.1f}%")

# ── 逐月明細 ──────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"  逐月 1M Alpha（vs SPY）")
print(f"  {'日期':<12} {'A 現行':>9} {'B 適中':>9} {'C 早期':>9}  最佳")
print("-" * 65)

by_date = {}
for key in ["A", "B", "C"]:
    for row in results[key]:
        d = row["date"]
        if d not in by_date:
            by_date[d] = {}
        by_date[d][key] = row["alpha_1m"]

for d in sorted(by_date.keys()):
    vals = {k: by_date[d].get(k) for k in ["A", "B", "C"]}
    valid = {k: v for k, v in vals.items() if v is not None}
    best = max(valid, key=valid.get) if valid else "-"
    def fmt(v): return f"{v:+.1f}%" if v is not None else " N/A "
    print(f"  {d:<12} {fmt(vals['A']):>9} {fmt(vals['B']):>9} {fmt(vals['C']):>9}  {best}")

# 選股比較：最後一期
print()
print("=" * 65)
print("  最後一期（2024-12）選股對照")
print("=" * 65)
for key in ["A", "B", "C"]:
    rows = [r for r in results[key] if r["date"] == "2024-12-02"]
    if rows:
        r = rows[0]
        print(f"  {desc[key]}")
        print(f"    選股: {', '.join(r['picks'])}")
        print(f"    252d均: {r['avg_mom_long']:+.0f}%   1M: {r['avg_1m']:+.1f}%   alpha: {r['alpha_1m']:+.1f}%")
