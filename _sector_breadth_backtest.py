"""
板塊動能廣度回測

概念：市場的強勢動能集中在少數板塊時，ADD 不應分散買多個板塊的標的，
     應集中在有動能的板塊。

板塊動能廣度 = 當前有正動能的板塊數（vs SPY）/ 11

測試三種策略：
  A（基準）：top5 mom_mixed，不考慮板塊（現行）
  B（板塊過濾）：從 top20 mom 中，只選強勢板塊（ETF > SPY）的標的
                 若強勢板塊太少（< 3 個），fallback 到 top5
  C（板塊加權）：選股數 = max(2, round(強勢板塊數 / 11 × 5))
                 強勢板塊多 → 多選幾支，強勢板塊少 → 集中押少數

分析面向：
  - 不同「板塊廣度」下的選股效果
  - 板塊過濾是否剔除了劣質候選

快取：
  - data/_protection_bt_prices.pkl（股票價格）
  - data/_sector_etf_prices.pkl（板塊 ETF，由 _relative_sector_strength_backtest.py 建立）
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
POOL_SIZE = 20    # 策略 B/C 的第一段候選池

SECTOR_ETF_MAP = {
    "科技":     "XLK",
    "醫療":     "XLV",
    "金融":     "XLF",
    "非必需消費": "XLY",
    "必需消費":  "XLP",
    "能源":     "XLE",
    "工業":     "XLI",
    "原物料":   "XLB",
    "通訊":     "XLC",
    "房地產":   "XLRE",
    "公用事業":  "XLU",
}

# ── 載入資料 ──────────────────────────────────────────────────────────
if not os.path.exists(PRICE_CACHE):
    print(f"找不到快取 {PRICE_CACHE}")
    raise SystemExit(1)
if not os.path.exists(SECTOR_ETF_CACHE):
    print(f"找不到 {SECTOR_ETF_CACHE}，請先執行 _relative_sector_strength_backtest.py")
    raise SystemExit(1)

print(f"讀取股票價格快取：{PRICE_CACHE}")
prices = pd.read_pickle(PRICE_CACHE)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()
print(f"股票：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日")

print(f"讀取 ETF 快取：{SECTOR_ETF_CACHE}")
etf_prices = pd.read_pickle(SECTOR_ETF_CACHE)
etf_prices.index = pd.to_datetime(etf_prices.index)
if etf_prices.index.tz is not None:
    etf_prices.index = etf_prices.index.tz_convert(None)
etf_prices = etf_prices.reindex(trading_days, method="ffill")

print(f"載入板塊對應...")
sector_map = get_sp500_sector_map()   # {sym: "科技", ...}
sym_to_sector = sector_map            # 直接用
print(f"有板塊對應：{len(sym_to_sector)} 檔\n")

spy = yf.Ticker("SPY").history(start="2020-12-01", end="2026-03-01",
                               auto_adjust=True)["Close"]
spy.index = pd.to_datetime(spy.index).tz_localize(None)
spy = spy.reindex(trading_days, method="ffill")


# ── 工具函式 ──────────────────────────────────────────────────────────
def tidx(date) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(date)))

def ret(df, sym, ti, lag):
    if ti - lag < 0 or ti >= len(trading_days):
        return None
    p0 = df.iloc[ti].get(sym)
    pl = df.iloc[ti - lag].get(sym)
    if p0 is None or pl is None:
        return None
    p0, pl = float(p0), float(pl)
    if pd.isna(p0) or pd.isna(pl) or pl <= 0:
        return None
    return (p0 / pl - 1) * 100

def mom_mixed_at(sym, ti):
    if ti < MOM_LONG:
        return None
    r21  = ret(prices, sym, ti, MOM_SHORT)
    r252 = ret(prices, sym, ti, MOM_LONG)
    if r21 is None or r252 is None:
        return None
    return 0.5 * r21 + 0.5 * r252

def get_strong_sectors(ti):
    """
    計算哪些板塊相對 SPY 有正動能（21d + 252d 都超過 SPY）。
    回傳 (strong_set, sector_breadth, sector_scores)
    """
    spy_r21  = ret(spy.to_frame("SPY"), "SPY", ti, MOM_SHORT)
    spy_r252 = ret(spy.to_frame("SPY"), "SPY", ti, MOM_LONG)
    if spy_r21 is None or spy_r252 is None:
        return set(), 0.0, {}

    strong = set()
    scores = {}
    for sector, etf in SECTOR_ETF_MAP.items():
        if etf not in etf_prices.columns:
            continue
        e_r21  = ret(etf_prices, etf, ti, MOM_SHORT)
        e_r252 = ret(etf_prices, etf, ti, MOM_LONG)
        if e_r21 is None or e_r252 is None:
            continue
        rel_21  = e_r21  - spy_r21
        rel_252 = e_r252 - spy_r252
        score = 0.5 * rel_21 + 0.5 * rel_252
        scores[sector] = score
        # 「強勢」定義：21d OR 252d 任一超過 SPY 均算正動能（寬鬆）
        # 用 score > 0 作為門檻
        if score > 0:
            strong.add(sector)

    breadth = len(strong) / len(SECTOR_ETF_MAP)
    return strong, breadth, scores

def fwd_return_sym(sym, ti, offset):
    p0 = prices.iloc[ti].get(sym)
    if ti + offset >= len(trading_days):
        return None
    p1 = prices.iloc[ti + offset].get(sym)
    if p0 is None or p1 is None:
        return None
    p0, p1 = float(p0), float(p1)
    if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
        return None
    return (p1 / p0 - 1) * 100

def spy_fwd(ref_date, offset):
    s = spy.dropna()
    ti = s.index.searchsorted(pd.Timestamp(ref_date))
    if ti + offset >= len(s):
        return None
    return (float(s.iloc[ti + offset]) / float(s.iloc[ti]) - 1) * 100


# ── 主回測循環 ─────────────────────────────────────────────────────────
rebalance_dates = pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")
results = {k: [] for k in ["A", "B", "C"]}
breadth_log = []

print("回測中...")
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG:
        continue

    strong_sectors, sec_breadth, sec_scores = get_strong_sectors(ti)
    breadth_log.append({
        "date": str(ref.date()),
        "breadth": sec_breadth,
        "n_strong": len(strong_sectors),
        "strong": sorted(strong_sectors),
    })

    # 全市場動能排名（前 POOL_SIZE）
    pool = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, ti)
        if m is None or m <= 0:
            continue
        sector = sym_to_sector.get(sym)
        in_strong = (sector in strong_sectors) if sector else False
        pool.append({"sym": sym, "mom": m, "sector": sector, "in_strong": in_strong})

    if len(pool) < TOP_N:
        continue

    df_pool = pd.DataFrame(pool)
    top20   = df_pool.nlargest(POOL_SIZE, "mom")
    picks   = {}

    # A：純動能基準
    picks["A"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()

    # B：從 top20 中只選強勢板塊的標的；若強勢板塊不足 3 個則 fallback
    if len(strong_sectors) >= 3:
        b_strong = top20[top20["in_strong"]].head(TOP_N)
        if len(b_strong) >= TOP_N:
            picks["B"] = b_strong["sym"].tolist()
        else:
            # 強勢板塊的標的不足 5 支，補齊
            b_rest = top20[~top20["sym"].isin(b_strong["sym"])].head(TOP_N - len(b_strong))
            picks["B"] = pd.concat([b_strong, b_rest])["sym"].tolist()
    else:
        picks["B"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()  # fallback

    # C：選股數 = max(2, round(板塊廣度 × 5))，從 top20 中的強勢板塊取
    n_c = max(2, round(sec_breadth * TOP_N))
    if len(strong_sectors) >= 2:
        c_strong = top20[top20["in_strong"]].head(n_c)
        if len(c_strong) >= n_c:
            picks["C"] = c_strong["sym"].tolist()
        else:
            c_rest = top20[~top20["sym"].isin(c_strong["sym"])].head(n_c - len(c_strong))
            picks["C"] = pd.concat([c_strong, c_rest])["sym"].tolist()
    else:
        picks["C"] = df_pool.nlargest(n_c, "mom")["sym"].tolist()

    spy_1m = spy_fwd(ref.date(), 21)
    spy_3m = spy_fwd(ref.date(), 63)

    for key, syms in picks.items():
        fwd_1m = [x for x in [fwd_return_sym(s, ti, 21) for s in syms] if x is not None]
        fwd_3m = [x for x in [fwd_return_sym(s, ti, 63) for s in syms] if x is not None]
        if not fwd_1m:
            continue
        avg_1m = np.mean(fwd_1m)
        avg_3m = np.mean(fwd_3m) if fwd_3m else None
        results[key].append({
            "date":      str(ref.date()),
            "picks":     syms,
            "avg_1m":    avg_1m,
            "avg_3m":    avg_3m,
            "alpha_1m":  avg_1m - spy_1m if spy_1m else None,
            "alpha_3m":  avg_3m - spy_3m if (avg_3m and spy_3m) else None,
            "n_strong":  len(strong_sectors),
            "sec_breadth": sec_breadth,
            "n_picks":   len(syms),
        })

print()

# ── 彙總輸出 ──────────────────────────────────────────────────────────
print("=" * 72)
print("  板塊動能廣度回測  2021-2025  S&P500 月度")
print("=" * 72)

df_bl = pd.DataFrame(breadth_log)
print(f"\n  板塊廣度統計（{len(df_bl)} 個月）")
print(f"    平均強勢板塊數: {df_bl['n_strong'].mean():.1f}/11  "
      f"最低: {df_bl['n_strong'].min()}  最高: {df_bl['n_strong'].max()}")
print(f"    廣度 < 30%（≤3 板塊）月數: {(df_bl['breadth'] < 0.30).sum()}")
print(f"    廣度 30-60%（4-6 板塊）月數: {((df_bl['breadth'] >= 0.30) & (df_bl['breadth'] < 0.60)).sum()}")
print(f"    廣度 ≥ 60%（≥7 板塊）月數: {(df_bl['breadth'] >= 0.60).sum()}")

# 顯示各月強勢板塊
print(f"\n  {'日期':<12} {'強勢板塊數':>10}  強勢板塊")
print("  " + "-" * 70)
for _, row in df_bl.iterrows():
    print(f"  {row['date']:<12} {row['n_strong']:>3}/11       {', '.join(row['strong'])}")

desc = {
    "A": "A. 純動能（基準）       top5 mom_mixed",
    "B": "B. 板塊過濾             top20 mom → 只選強勢板塊",
    "C": "C. 板塊廣度加權         選股數 = round(板塊廣度×5)",
}

print()
print("=" * 72)

summary = {}
for key in ["A", "B", "C"]:
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
print(f"  {'策略':<44} {'1M alpha 差':>12} {'3M alpha 差':>12}")
print("  " + "-" * 70)
for key in ["B", "C"]:
    if key in summary and "A" in summary:
        d1 = summary[key]["alpha_1m"] - summary["A"]["alpha_1m"]
        d3 = summary[key]["alpha_3m"] - summary["A"]["alpha_3m"]
        print(f"  {desc[key]:<44} {d1:>+11.2f}%  {d3:>+11.2f}%")

# ── 板塊廣度分組分析 ───────────────────────────────────────────────────
print()
print("=" * 72)
print("  板塊廣度分組：廣度不同時，策略 A vs B vs C 的表現")
print()

all_rows = []
for key in ["A", "B", "C"]:
    for row in results[key]:
        all_rows.append({**row, "strategy": key})
df_all = pd.DataFrame(all_rows)

bins   = [0, 0.27, 0.45, 0.64, 1.01]
labels = ["≤3 板塊（集中）", "4-5 板塊（偏窄）", "6-7 板塊（均衡）", "≥8 板塊（廣泛）"]
df_all["bucket"] = pd.cut(df_all["sec_breadth"], bins=bins, labels=labels)

print(f"  {'廣度區間':<18} {'策略':>4} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 50)
for label in labels:
    for key in ["A", "B", "C"]:
        sub = df_all[(df_all["bucket"] == label) & (df_all["strategy"] == key)]
        if len(sub) == 0:
            continue
        a1 = sub["alpha_1m"].dropna().mean()
        a3 = sub["alpha_3m"].dropna().mean()
        print(f"  {label if key=='A' else '':<18} {key:>4} {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")
    print()

# ── 逐月明細 ──────────────────────────────────────────────────────────
print("=" * 72)
print(f"  逐月：板塊廣度 + 1M Alpha")
print(f"  {'日期':<12} {'強勢板塊':>8} {'A 基準':>8} {'B 過濾':>8} {'C 加權':>8}  最佳")
print("  " + "-" * 66)

by_date = {}
for key in ["A", "B", "C"]:
    for row in results[key]:
        d = row["date"]
        if d not in by_date:
            by_date[d] = {"n_strong": row.get("n_strong"), "n_picks_C": row.get("n_picks") if key == "C" else None}
        by_date[d][key] = row["alpha_1m"]
        if key == "C":
            by_date[d]["n_picks_C"] = row.get("n_picks")

def fmt(v):
    return f"{v:+.1f}%" if v is not None else "  N/A "

win_count = {"A": 0, "B": 0, "C": 0}
for d in sorted(by_date.keys()):
    vals = {k: by_date[d].get(k) for k in ["A", "B", "C"]}
    valid = {k: v for k, v in vals.items() if v is not None}
    best = max(valid, key=valid.get) if valid else "-"
    if best in win_count:
        win_count[best] += 1
    ns = by_date[d].get("n_strong", "?")
    nc = by_date[d].get("n_picks_C", "?")
    print(f"  {d:<12} {str(ns)+'/11':>8} {fmt(vals['A']):>8} {fmt(vals['B']):>8} "
          f"{fmt(vals['C'])+'('+str(nc)+')':>10}  {best}")

print()
print(f"  各策略最佳月數：" +
      "  ".join(f"{k}={v}" for k, v in win_count.items()))
