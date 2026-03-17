"""
相對板塊強弱選股回測

假設：動能高的股票裡，真正的板塊領頭羊（超越自己的板塊 ETF）
比只是吃板塊 beta 的股票，未來報酬更好。

指標定義：
  rel_sector_mom = 0.5*(stock_21d - sector_21d) + 0.5*(stock_252d - sector_252d)
  → 相對板塊的超額動能

測試四種策略：
  A（基準）：top5 mom_mixed（現行）
  B（相對強弱）：top5 by rel_sector_mom（純看 vs 板塊的超額）
  C（複合）：top5 by (mom_mixed + rel_sector_mom)
  D（兩段）：top10 mom → top5 by rel_sector_mom

快取：
  - data/_protection_bt_prices.pkl（股票價格）
  - data/_sector_etf_prices.pkl（板塊 ETF 價格，首次下載）
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from src.data_loader import get_sp500_sector_map

PRICE_CACHE     = "data/_protection_bt_prices.pkl"
SECTOR_ETF_CACHE = "data/_sector_etf_prices.pkl"
MOM_SHORT = 21
MOM_LONG  = 252
TOP_N     = 5
POOL_D    = 10

SECTOR_ETF_MAP = {
    "科技":       "XLK",
    "醫療":       "XLV",
    "金融":       "XLF",
    "非必需消費":  "XLY",
    "必需消費":   "XLP",
    "能源":       "XLE",
    "工業":       "XLI",
    "原物料":     "XLB",
    "通訊":       "XLC",
    "房地產":     "XLRE",
    "公用事業":   "XLU",
}

# ── 載入股票價格 ───────────────────────────────────────────────────────
if not os.path.exists(PRICE_CACHE):
    print(f"找不到快取 {PRICE_CACHE}")
    raise SystemExit(1)

print(f"讀取股票價格快取：{PRICE_CACHE}")
prices = pd.read_pickle(PRICE_CACHE)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()
print(f"股票：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日")

# ── 載入板塊 ETF 價格 ──────────────────────────────────────────────────
etf_tickers = list(set(SECTOR_ETF_MAP.values()))
if os.path.exists(SECTOR_ETF_CACHE):
    print(f"讀取 ETF 快取：{SECTOR_ETF_CACHE}")
    etf_prices = pd.read_pickle(SECTOR_ETF_CACHE)
else:
    print(f"下載 {len(etf_tickers)} 個板塊 ETF...")
    raw = yf.download(etf_tickers, start="2019-01-01", end="2026-03-01",
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(0)
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(None)
    pd.to_pickle(raw, SECTOR_ETF_CACHE)
    etf_prices = raw
    print(f"已快取至 {SECTOR_ETF_CACHE}")

# 對齊交易日
etf_prices = etf_prices.reindex(trading_days, method="ffill")
print(f"ETF：{etf_prices.shape[1]} 檔\n")

# ── 板塊對應 ────────────────────────────────────────────────────────────
print("載入板塊對應...")
sector_map = get_sp500_sector_map()        # {sym: "Information Technology", ...}
sym_to_etf = {sym: SECTOR_ETF_MAP.get(sec)
              for sym, sec in sector_map.items()
              if SECTOR_ETF_MAP.get(sec)}
print(f"有板塊對應的標的：{len(sym_to_etf)} 檔\n")

# ── SPY 基準 ─────────────────────────────────────────────────────────────
spy = yf.Ticker("SPY").history(start="2020-12-01", end="2026-03-01",
                               auto_adjust=True)["Close"]
spy.index = pd.to_datetime(spy.index).tz_localize(None)


# ── 工具函式 ─────────────────────────────────────────────────────────────
def tidx(date) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(date)))

def price_at(df, sym, ti):
    if ti < 0 or ti >= len(trading_days):
        return None
    v = df.iloc[ti].get(sym)
    return float(v) if v is not None and pd.notna(v) else None

def return_pct(df, sym, ti, lag):
    p0 = price_at(df, sym, ti)
    pl = price_at(df, sym, ti - lag)
    if p0 and pl and pl > 0:
        return (p0 / pl - 1) * 100
    return None

def mom_mixed_at(sym, ti):
    r21  = return_pct(prices, sym, ti, MOM_SHORT)
    r252 = return_pct(prices, sym, ti, MOM_LONG)
    if r21 is None or r252 is None:
        return None
    return 0.5 * r21 + 0.5 * r252

def rel_sector_mom_at(sym, ti):
    """股票 vs 板塊 ETF 的相對動能"""
    etf = sym_to_etf.get(sym)
    if etf is None:
        return None
    s_r21  = return_pct(prices,    sym, ti, MOM_SHORT)
    s_r252 = return_pct(prices,    sym, ti, MOM_LONG)
    e_r21  = return_pct(etf_prices, etf, ti, MOM_SHORT)
    e_r252 = return_pct(etf_prices, etf, ti, MOM_LONG)
    if any(x is None for x in [s_r21, s_r252, e_r21, e_r252]):
        return None
    return 0.5 * (s_r21 - e_r21) + 0.5 * (s_r252 - e_r252)

def fwd_return_sym(sym, ti, offset):
    p0 = price_at(prices, sym, ti)
    p1 = price_at(prices, sym, ti + offset)
    if p0 and p1 and p0 > 0:
        return (p1 / p0 - 1) * 100
    return None

def spy_fwd(ref_date, offset):
    s = spy.dropna()
    ti = s.index.searchsorted(pd.Timestamp(ref_date))
    if ti + offset >= len(s):
        return None
    return (float(s.iloc[ti + offset]) / float(s.iloc[ti]) - 1) * 100


# ── 主回測循環 ────────────────────────────────────────────────────────────
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
        r = rel_sector_mom_at(sym, ti)
        if r is None:
            continue
        pool.append({"sym": sym, "mom": m, "rel": r, "composite": m + r})

    if len(pool) < POOL_D:
        continue

    df_pool = pd.DataFrame(pool)
    top10   = df_pool.nlargest(POOL_D, "mom")
    picks   = {}

    picks["A"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()
    picks["B"] = df_pool.nlargest(TOP_N, "rel")["sym"].tolist()
    picks["C"] = df_pool.nlargest(TOP_N, "composite")["sym"].tolist()
    picks["D"] = top10.nlargest(TOP_N, "rel")["sym"].tolist()

    spy_1m = spy_fwd(ref.date(), 21)
    spy_3m = spy_fwd(ref.date(), 63)

    for key, syms in picks.items():
        fwd_1m = [x for x in [fwd_return_sym(s, ti, 21) for s in syms] if x is not None]
        fwd_3m = [x for x in [fwd_return_sym(s, ti, 63) for s in syms] if x is not None]
        if not fwd_1m:
            continue
        avg_1m = np.mean(fwd_1m)
        avg_3m = np.mean(fwd_3m) if fwd_3m else None
        sub    = df_pool[df_pool["sym"].isin(syms)]
        results[key].append({
            "date":    str(ref.date()),
            "picks":   syms,
            "avg_1m":  avg_1m,
            "avg_3m":  avg_3m,
            "alpha_1m": avg_1m - spy_1m if spy_1m else None,
            "alpha_3m": avg_3m - spy_3m if (avg_3m and spy_3m) else None,
            "avg_rel": sub["rel"].mean(),
        })

print()

# ── 彙總輸出 ──────────────────────────────────────────────────────────────
print("=" * 72)
print("  相對板塊強弱回測  2021-2025  S&P500 月度")
print("=" * 72)

desc = {
    "A": "A. 純動能（基準）          top5 mom_mixed",
    "B": "B. 相對板塊強弱            top5 rel_sector_mom",
    "C": "C. 複合分數                top5 (mom + rel)",
    "D": "D. 兩段篩選                top10 mom → top5 rel",
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
    avg_rel  = df["avg_rel"].mean()
    summary[key] = {"alpha_1m": alpha_1m, "alpha_3m": alpha_3m}
    print(f"\n  {desc[key]}")
    print(f"    平均 1M 報酬: {avg_1m:+.2f}%   3M: {avg_3m:+.2f}%   (n={n})")
    print(f"    vs SPY alpha: 1M {alpha_1m:+.2f}%   3M {alpha_3m:+.2f}%")
    print(f"    月勝率(>0): {win_1m:.0f}%   月跑贏SPY: {win_a:.0f}%")
    print(f"    選入時平均 rel_sector_mom: {avg_rel:+.1f}%")

print()
print("=" * 72)
print("  相對基準 A 的改善量")
print(f"  {'策略':<44} {'1M alpha 差':>12} {'3M alpha 差':>12}")
print("  " + "-" * 70)
for key in ["B", "C", "D"]:
    if key in summary and "A" in summary:
        d1 = summary[key]["alpha_1m"] - summary["A"]["alpha_1m"]
        d3 = summary[key]["alpha_3m"] - summary["A"]["alpha_3m"]
        print(f"  {desc[key]:<44} {d1:>+11.2f}%  {d3:>+11.2f}%")

# ── rel_sector_mom 分組分析 ────────────────────────────────────────────────
print()
print("=" * 72)
print("  rel_sector_mom 分組（策略 A 選股，按相對板塊強弱分高/低）")
print("  驗證：在動能相近時，板塊領頭羊是否報酬更好？")
print()

all_events = []
for row in results["A"]:
    ti = tidx(row["date"])
    for sym in row["picks"]:
        r   = rel_sector_mom_at(sym, ti)
        r1  = fwd_return_sym(sym, ti, 21)
        r3  = fwd_return_sym(sym, ti, 63)
        s1  = spy_fwd(row["date"], 21)
        s3  = spy_fwd(row["date"], 63)
        if None in (r, r1, s1):
            continue
        all_events.append({
            "sym": sym, "date": row["date"], "rel": r,
            "r1": r1, "r3": r3,
            "alpha_1m": r1 - s1,
            "alpha_3m": (r3 - s3) if (r3 and s3) else None,
        })

df_ev = pd.DataFrame(all_events)
med_rel = df_ev["rel"].median()
df_hi = df_ev[df_ev["rel"] >= med_rel]
df_lo = df_ev[df_ev["rel"] <  med_rel]

print(f"  中位數 rel_sector_mom：{med_rel:+.1f}%")
print(f"  {'分組':<26} {'1M報酬':>9} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 60)
for label, sub in [("rel ≥ 中位（板塊領頭羊）", df_hi), ("rel < 中位（板塊跟隨者）", df_lo)]:
    r1 = sub["r1"].mean()
    a1 = sub["alpha_1m"].dropna().mean()
    a3 = sub["alpha_3m"].dropna().mean()
    print(f"  {label:<26} {r1:>+8.2f}%  {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")

# rel > 0 vs < 0（真正超越 vs 只是跟隨板塊）
print()
df_outperform = df_ev[df_ev["rel"] > 0]
df_underperform = df_ev[df_ev["rel"] <= 0]
print(f"  {'rel > 0（真正超越板塊）':<26} {df_outperform['r1'].mean():>+8.2f}%  "
      f"{df_outperform['alpha_1m'].dropna().mean():>+9.2f}%  "
      f"{df_outperform['alpha_3m'].dropna().mean():>+9.2f}%  {len(df_outperform)}")
print(f"  {'rel ≤ 0（跑輸板塊）':<26} {df_underperform['r1'].mean():>+8.2f}%  "
      f"{df_underperform['alpha_1m'].dropna().mean():>+9.2f}%  "
      f"{df_underperform['alpha_3m'].dropna().mean():>+9.2f}%  {len(df_underperform)}")

# 逐月明細
print()
print("=" * 72)
print(f"  逐月 1M Alpha（vs SPY）")
print(f"  {'日期':<12} {'A 基準':>8} {'B 相對':>8} {'C 複合':>8} {'D 兩段':>8}  最佳")
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
