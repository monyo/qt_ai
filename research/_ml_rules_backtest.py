"""
ML 規則萃取回測

把 ML 實驗得到的最強訊號蒸餾成可解釋的規則，驗證回測效果：

策略：
  A（基準）      ：top5 mom_mixed
  B（板塊加權）  ：score = mom_mixed + k × sector_rel_21d，取 top5
  C（VIX 過濾）  ：VIX<20→5支, VIX<30→3支, VIX≥30→1支
  D（B + C 組合）：板塊加權 + VIX 縮減

資料：S&P500 2021-2025，月度再平衡
快取：data/_protection_bt_prices.pkl + 已有輔助快取
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_PATH   = "data/_protection_bt_prices.pkl"
SECTOR_CACHE = "data/_ml_sector_map.pkl"
ETF_CACHE    = "data/_ml_sector_etf_prices.pkl"
VIX_CACHE    = "data/_ml_vix.pkl"

MOM_SHORT    = 21
MOM_LONG     = 252
TOP_N        = 5
FWD_OFFSET   = 21
SECTOR_WEIGHT = 0.3   # 板塊加權比例：score = 0.7*mom_norm + 0.3*sector_norm

SECTOR_ETF_MAP = {
    "Technology":             "XLK", "Information Technology": "XLK",
    "Financial Services":     "XLF", "Financials":             "XLF",
    "Healthcare":             "XLV", "Health Care":            "XLV",
    "Energy":                 "XLE", "Utilities":              "XLU",
    "Industrials":            "XLI", "Basic Materials":        "XLB",
    "Materials":              "XLB", "Real Estate":            "XLRE",
    "Consumer Cyclical":      "XLY", "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP", "Consumer Staples":       "XLP",
    "Communication Services": "XLC",
}
SECTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()))

# ── 載入快取 ───────────────────────────────────────────────────────────
for p in [CACHE_PATH, SECTOR_CACHE, ETF_CACHE]:
    if not os.path.exists(p):
        print(f"找不到：{p}，請先執行前置腳本"); raise SystemExit(1)

print(f"讀取快取...")
prices = pd.read_pickle(CACHE_PATH)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()
print(f"資料：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日\n")

sector_map = pd.read_pickle(SECTOR_CACHE)
if isinstance(sector_map, pd.Series):
    sector_map = sector_map.to_dict()

etf_prices = pd.read_pickle(ETF_CACHE)
etf_prices.index = (pd.to_datetime(etf_prices.index).tz_localize(None)
                    if etf_prices.index.tz is None
                    else etf_prices.index.tz_convert(None))
etf_prices = etf_prices.reindex(trading_days)

# VIX
if os.path.exists(VIX_CACHE):
    vix = pd.read_pickle(VIX_CACHE)
else:
    vix = yf.Ticker("^VIX").history(start="2020-01-01", end="2026-03-01",
                                     auto_adjust=True)["Close"]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.to_pickle(VIX_CACHE)
vix = vix.reindex(trading_days, method="ffill")

spy = yf.Ticker("SPY").history(start="2020-12-01", end="2026-03-01",
                               auto_adjust=True)["Close"]
spy.index = pd.to_datetime(spy.index).tz_localize(None)


# ── 預計算 ─────────────────────────────────────────────────────────────
print("預計算板塊相對強弱...")
spy_etf = etf_prices["SPY"]
etf_rel_21 = {}
for etf in SECTOR_ETFS:
    if etf not in etf_prices.columns:
        continue
    rel = etf_prices[etf] / spy_etf
    etf_rel_21[etf] = (rel / rel.shift(21) - 1) * 100


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

def sector_rel_at(sym, ti):
    sector = sector_map.get(sym)
    etf    = SECTOR_ETF_MAP.get(sector) if sector else None
    if etf is None or etf not in etf_rel_21:
        return None
    series = etf_rel_21[etf]
    if ti >= len(series):
        return None
    v = series.iloc[ti]
    return float(v) if pd.notna(v) else None

def vix_at(ti):
    if ti >= len(vix):
        return None
    v = vix.iloc[ti]
    return float(v) if pd.notna(v) else None

def fwd_return_sym(sym, ti, offset):
    p0 = price_at(sym, ti)
    p1 = price_at(sym, ti + offset)
    if p0 and p1 and p0 > 0:
        return (p1 / p0 - 1) * 100
    return None

def spy_fwd(ref_date, offset):
    s  = spy.dropna()
    ti = s.index.searchsorted(pd.Timestamp(ref_date))
    if ti + offset >= len(s):
        return None
    return (float(s.iloc[ti + offset]) / float(s.iloc[ti]) - 1) * 100

def vix_n_picks(vix_val):
    """VIX 高低決定選幾支"""
    if vix_val is None:
        return TOP_N
    if vix_val >= 30:
        return 1
    if vix_val >= 20:
        return 3
    return 5


# ── 主回測 ─────────────────────────────────────────────────────────────
rebalance_dates = pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")
results = {k: [] for k in ["A", "B", "C", "D"]}
vix_log = []

print("回測中...")
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG:
        continue

    spy_ret = spy_fwd(ref.date(), FWD_OFFSET)
    if spy_ret is None:
        continue

    vix_val = vix_at(ti)
    vix_log.append(vix_val or np.nan)
    n_vix = vix_n_picks(vix_val)

    # 全市場動能 + 板塊相對強弱
    pool = []
    for sym in prices.columns:
        mom = mom_mixed_at(sym, ti)
        if mom is None:
            continue
        sec = sector_rel_at(sym, ti)
        pool.append({"sym": sym, "mom": mom, "sec": sec or 0.0})

    if len(pool) < TOP_N:
        continue

    df_pool = pd.DataFrame(pool)

    # 標準化（排除 nan 後 zscore）
    mom_std = df_pool["mom"].std()
    sec_std = df_pool["sec"].std()
    df_pool["mom_z"] = (df_pool["mom"] - df_pool["mom"].mean()) / (mom_std if mom_std > 0 else 1)
    df_pool["sec_z"] = (df_pool["sec"] - df_pool["sec"].mean()) / (sec_std if sec_std > 0 else 1)
    df_pool["score_B"] = (1 - SECTOR_WEIGHT) * df_pool["mom_z"] + SECTOR_WEIGHT * df_pool["sec_z"]

    def avg_alpha(syms):
        vals = []
        for sym in syms:
            fwd = fwd_return_sym(sym, ti, FWD_OFFSET)
            if fwd is not None:
                vals.append(fwd - spy_ret)
        return float(np.median(vals)) if vals else None

    # A：純動能 top5
    top_A = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()
    alpha_A = avg_alpha(top_A)

    # B：板塊加權 top5
    top_B = df_pool.nlargest(TOP_N, "score_B")["sym"].tolist()
    alpha_B = avg_alpha(top_B)

    # C：VIX 縮減（純動能 + 數量控制）
    top_C = df_pool.nlargest(n_vix, "mom")["sym"].tolist()
    alpha_C = avg_alpha(top_C)

    # D：板塊加權 + VIX 縮減
    top_D = df_pool.nlargest(n_vix, "score_B")["sym"].tolist()
    alpha_D = avg_alpha(top_D)

    if all(x is not None for x in [alpha_A, alpha_B, alpha_C, alpha_D]):
        results["A"].append(alpha_A)
        results["B"].append(alpha_B)
        results["C"].append(alpha_C)
        results["D"].append(alpha_D)


# ── 輸出 ───────────────────────────────────────────────────────────────
print(f"\n=== ML 規則萃取回測  S&P500 2021-2025 ===\n")
print(f"回測月份：{len(results['A'])} 個月")
print(f"VIX 均值：{np.nanmean(vix_log):.1f}（最小 {np.nanmin(vix_log):.1f} / 最大 {np.nanmax(vix_log):.1f}）\n")

labels = {
    "A": "A  基準（top5 mom_mixed）",
    "B": "B  板塊加權（0.7mom+0.3sector）",
    "C": "C  VIX 縮減（<20→5支,<30→3支,≥30→1支）",
    "D": "D  B+C 組合",
}
print(f"  {'策略':<38s}  {'1M alpha 中位':>12s}  {'平均':>8s}  {'勝A月數':>8s}")
print("  " + "-" * 72)

median_A = float(np.median(results["A"]))
for k in ["A", "B", "C", "D"]:
    vals = results[k]
    med  = float(np.median(vals))
    avg  = float(np.mean(vals))
    win  = sum(1 for v, a in zip(vals, results["A"]) if v > a) if k != "A" else "-"
    flag = ""
    if k != "A":
        flag = "✅" if med > median_A + 0.2 else ("➖" if abs(med - median_A) < 0.1 else "")
    print(f"  {labels[k]:<38s}  {med:>+11.2f}%  {avg:>+7.2f}%  {str(win):>8s}  {flag}")

# VIX 環境分析：高/低 VIX 時各策略比較
print("\n=== VIX 環境分析 ===\n")
vix_vals = np.array(vix_log[:len(results["A"])])
low_mask  = vix_vals < 20
high_mask = vix_vals >= 20

for env_name, mask in [("低 VIX（<20）", low_mask), ("高 VIX（≥20）", high_mask)]:
    n = mask.sum()
    print(f"  {env_name}  {n} 個月")
    for k in ["A", "B", "C", "D"]:
        arr  = np.array(results[k])
        vals = arr[mask]
        if len(vals) > 0:
            print(f"    {labels[k]:<38s}  中位 {np.median(vals):+.2f}%")
    print()
