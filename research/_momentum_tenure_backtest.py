"""
動能在榜時間（Tenure）回測

追蹤每支股票在 top-20 動能榜的「連續在榜月數」（tenure），
測試「新進榜股票」vs「長期在榜股票」的後續報酬差異。

假設：
  - 新進榜（tenure=1）= 剛突破，可能有更多動能空間
  - 久在榜（tenure≥4）= 動能可能趨緩（該 ROTATE？）

四種策略：
  A（基準）    ：top5 mom_mixed，不考慮在榜時間
  B（新鮮優先）：從 top20 優先選 tenure ≤ 2，不足再補動能最高
  C（老兵優先）：從 top20 優先選 tenure ≥ 3，不足再補動能最高
  D（衰減懲罰）：調整分數 = mom × max(0.7, 1 - 0.05*(tenure-1))，重排取 top5

附加分析：按 tenure 分組（1, 2-3, 4-6, 7+），看各組平均後續報酬
→ 找出最佳 ROTATE 時機

資料：S&P500 2021-2025，月度再平衡
快取：data/_protection_bt_prices.pkl
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_PATH = "data/_protection_bt_prices.pkl"
MOM_SHORT  = 21
MOM_LONG   = 252
TOP_N      = 5
TOP_POOL   = 20

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


# ── 第一遍：建立每月 top-20 榜單（按時間順序）─────────────────────────
rebalance_dates = pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")
monthly_top20   = {}  # ref -> list of (sym, mom)

print("第一遍：建立每月 top-20 榜單...")
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG:
        continue
    pool = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, ti)
        if m is None or m <= 0:
            continue
        pool.append((sym, m))
    if len(pool) < TOP_N:
        continue
    pool.sort(key=lambda x: -x[1])
    monthly_top20[ref] = pool[:TOP_POOL]

# ── 第二遍：按時間追蹤在榜月數（tenure）──────────────────────────────
sorted_months = sorted(monthly_top20.keys())
current_tenure = {}  # sym -> 目前連續在榜月數

# tenure_at[ref] = {sym: tenure}
tenure_at = {}
for ref in sorted_months:
    top_this = {sym for sym, _ in monthly_top20[ref]}
    new_tenure = {}
    for sym in top_this:
        new_tenure[sym] = current_tenure.get(sym, 0) + 1
    current_tenure = new_tenure
    tenure_at[ref] = {sym: new_tenure[sym] for sym in top_this}

print(f"完成在榜時間計算：{len(sorted_months)} 個月\n")


# ── 主回測循環 ─────────────────────────────────────────────────────────
results       = {k: [] for k in ["A", "B", "C", "D"]}
tenure_groups = []  # 用於分組分析：{tenure_bucket, sym, fwd_1m, fwd_3m, alpha_1m, alpha_3m}

print("回測中...")
for ref in sorted_months:
    ti = tidx(str(ref.date()))
    pool_raw  = monthly_top20[ref]              # list of (sym, mom)
    tenure_mp = tenure_at[ref]                  # {sym: tenure}

    spy_1m = spy_fwd(ref.date(), 21)
    spy_3m = spy_fwd(ref.date(), 63)

    # ─ 策略 A：純動能 top5 ──────────────────────────────────────────
    picks_A = [sym for sym, _ in pool_raw[:TOP_N]]

    # ─ 策略 B：新鮮優先（tenure ≤ 2）─────────────────────────────────
    fresh   = [(sym, m) for sym, m in pool_raw if tenure_mp.get(sym, 1) <= 2]
    veteran = [(sym, m) for sym, m in pool_raw if tenure_mp.get(sym, 1) > 2]
    # 先取 fresh，不足補 veteran（按動能順序）
    combined_B = fresh + veteran
    picks_B    = [sym for sym, _ in combined_B[:TOP_N]]

    # ─ 策略 C：老兵優先（tenure ≥ 3）─────────────────────────────────
    combined_C = veteran + fresh
    picks_C    = [sym for sym, _ in combined_C[:TOP_N]]

    # ─ 策略 D：衰減懲罰 ───────────────────────────────────────────────
    def decay(tenure):
        return max(0.70, 1.0 - 0.05 * (tenure - 1))

    pool_D = [(sym, m * decay(tenure_mp.get(sym, 1))) for sym, m in pool_raw]
    pool_D.sort(key=lambda x: -x[1])
    picks_D = [sym for sym, _ in pool_D[:TOP_N]]

    # ─ 計算報酬 ──────────────────────────────────────────────────────
    for key, picks in [("A", picks_A), ("B", picks_B), ("C", picks_C), ("D", picks_D)]:
        fwd_1m = [x for x in [fwd_return_sym(s, ti, 21) for s in picks] if x is not None]
        fwd_3m = [x for x in [fwd_return_sym(s, ti, 63) for s in picks] if x is not None]
        if not fwd_1m:
            continue
        avg_1m = np.mean(fwd_1m)
        avg_3m = np.mean(fwd_3m) if fwd_3m else None
        results[key].append({
            "date":     str(ref.date()),
            "avg_1m":   avg_1m,
            "avg_3m":   avg_3m,
            "alpha_1m": avg_1m - spy_1m if spy_1m is not None else None,
            "alpha_3m": avg_3m - spy_3m if (avg_3m and spy_3m) else None,
            "n_picks":  len(picks),
        })

    # ─ tenure 分組分析 ───────────────────────────────────────────────
    for sym, mom in pool_raw:
        ten = tenure_mp.get(sym, 1)
        f1  = fwd_return_sym(sym, ti, 21)
        f3  = fwd_return_sym(sym, ti, 63)
        if f1 is None:
            continue
        if ten == 1:
            bucket = "1（新進榜）"
        elif ten <= 3:
            bucket = "2-3"
        elif ten <= 6:
            bucket = "4-6"
        else:
            bucket = "7+（老兵）"
        tenure_groups.append({
            "bucket":   bucket,
            "sym":      sym,
            "tenure":   ten,
            "fwd_1m":   f1,
            "fwd_3m":   f3,
            "alpha_1m": f1 - spy_1m if spy_1m is not None else None,
            "alpha_3m": f3 - spy_3m if (f3 and spy_3m) else None,
        })

print()


# ── 彙總輸出 ──────────────────────────────────────────────────────────
print("=" * 72)
print("  動能在榜時間（Tenure）回測  2021-2025  S&P500 月度")
print("=" * 72)

desc = {
    "A": "A. 純動能基準               top5 mom_mixed",
    "B": "B. 新鮮優先                 tenure≤2 優先，不足補動能最高",
    "C": "C. 老兵優先                 tenure≥3 優先，不足補動能最高",
    "D": "D. 衰減懲罰                 分數×max(0.7,1-0.05×(t-1))，重排取top5",
}

summary = {}
for key in ["A", "B", "C", "D"]:
    rows = results[key]
    if not rows:
        continue
    df = pd.DataFrame(rows).dropna(subset=["alpha_1m"])
    summary[key] = {
        "n":        len(df),
        "med_1m":   df["alpha_1m"].median(),
        "avg_1m":   df["alpha_1m"].mean(),
        "med_3m":   df["alpha_3m"].dropna().median() if "alpha_3m" in df.columns else None,
        "wins_vs_A": None,
    }

# 計算 B/C/D vs A 的月勝次數
df_A = pd.DataFrame(results["A"]).dropna(subset=["alpha_1m"])[["date","alpha_1m"]].rename(columns={"alpha_1m":"alpha_A"})
for key in ["B","C","D"]:
    df_x = pd.DataFrame(results[key]).dropna(subset=["alpha_1m"])[["date","alpha_1m"]].rename(columns={"alpha_1m":f"alpha_{key}"})
    merged = df_A.merge(df_x, on="date", how="inner")
    if len(merged) > 0:
        summary[key]["wins_vs_A"] = (merged[f"alpha_{key}"] > merged["alpha_A"]).sum()
        summary[key]["total_vs_A"] = len(merged)

print(f"\n  策略比較（月中位 alpha vs SPY）\n")
print(f"  {'策略':<46} {'1M中位':>8} {'1M均值':>8} {'3M中位':>8} {'勝A月數':>8}")
print(f"  {'-'*44} {'-------':>8} {'-------':>8} {'-------':>8} {'-------':>8}")

for key in ["A","B","C","D"]:
    s = summary[key]
    wins_str = f"{s['wins_vs_A']}/{s.get('total_vs_A','?')}" if s["wins_vs_A"] is not None else "  —"
    m3 = f"{s['med_3m']:+.2f}%" if s["med_3m"] is not None else "  N/A"
    print(f"  {desc[key]:<46} {s['med_1m']:+.2f}%   {s['avg_1m']:+.2f}%   {m3:>8}   {wins_str:>7}")

# ── vs 基準改善量 ─────────────────────────────────────────────────────
print(f"\n  相對基準 A 的改善量")
print(f"  {'策略':<20} {'1M alpha 差':>12} {'3M alpha 差':>12}")
print(f"  {'-'*20} {'-------':>12} {'-------':>12}")

for key in ["B","C","D"]:
    diff_1m = summary[key]["med_1m"] - summary["A"]["med_1m"]
    diff_3m = (summary[key]["med_3m"] - summary["A"]["med_3m"]
               if summary[key]["med_3m"] and summary["A"]["med_3m"] else None)
    d3 = f"{diff_3m:+.2f}%" if diff_3m is not None else "  N/A"
    print(f"  {desc[key][:20]:<20} {diff_1m:+.2f}%        {d3:>12}")


# ── tenure 分組分析 ────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  Tenure 分組分析（top-20 中，按在榜時間分組的後續報酬）")
print(f"  驗證：在榜越久的股票，報酬是否越差？最佳 ROTATE 時機？")
print(f"{'='*72}\n")

df_tg = pd.DataFrame(tenure_groups).dropna(subset=["alpha_1m"])
order = ["1（新進榜）", "2-3", "4-6", "7+（老兵）"]
print(f"  {'在榜時間':<14} {'次數':>6} {'1M alpha 中位':>14} {'1M alpha 均值':>14} {'3M alpha 中位':>14}")
print(f"  {'-'*14} {'------':>6} {'-------------':>14} {'-------------':>14} {'-------------':>14}")

for bucket in order:
    sub = df_tg[df_tg["bucket"] == bucket]
    if len(sub) == 0:
        continue
    m1  = sub["alpha_1m"].median()
    a1  = sub["alpha_1m"].mean()
    m3  = sub["alpha_3m"].dropna().median() if "alpha_3m" in sub.columns else None
    m3s = f"{m3:+.2f}%" if m3 is not None else "  N/A"
    print(f"  {bucket:<14} {len(sub):>6} {m1:>+12.2f}%   {a1:>+12.2f}%   {m3s:>14}")

# 統計 tenure 分布
print(f"\n  Tenure 分布（top-20 所有出現紀錄）")
ten_counts = df_tg.groupby("tenure").size()
print(f"  {dict(ten_counts.head(10))}")

print(f"\n{'='*72}")
