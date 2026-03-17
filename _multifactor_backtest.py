"""
多因子組合回測

結合已驗證有效的訊號：
  - 市場廣度加權（breadth regime 回測：+1%/+4.5% alpha）
  - 趨勢狀態（轉強↗/轉弱↘/盤整→，+2.14% 月差，但未與廣度交叉驗證）

四種策略：
  A（基準）        ：top5 mom_mixed
  B（廣度加權）    ：n_picks = round(breadth×5)  [與廣度回測 D 相同，作對照用]
  C（廣度+趨勢過濾）：n_picks = round(breadth×5)，從 top20 中排除 ↘️ 轉弱股票
  D（廣度+趨勢加分）：↗️ 轉強的股票動能分數加 10%，重排後取廣度加權數量

趨勢狀態定義（與 src/momentum.py 一致）：
  bounce_pct  = (current / low_40d  - 1) × 100
  from_high%  = (current / high_40d - 1) × 100
  轉強 ↗️：bounce_pct > 20 AND from_high% > -5
  轉弱 ↘️：from_high% < -15
  盤整 →：其他

資料：S&P500 2021-2025，月度再平衡
快取：data/_protection_bt_prices.pkl
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

CACHE_PATH = "data/_protection_bt_prices.pkl"
MOM_SHORT  = 21
MOM_LONG   = 252
MA_WIN     = 50
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


# ── 預計算廣度序列 ─────────────────────────────────────────────────────
print("計算每日市場廣度（% 股票 > 50MA）...")
ma50        = prices.rolling(MA_WIN, min_periods=30).mean()
above_ma50  = (prices > ma50).astype(int)
valid_count = (prices.notna() & ma50.notna()).sum(axis=1)
above_count = above_ma50.sum(axis=1)
breadth_series = (above_count / valid_count).where(valid_count > 100)

# ── 預計算趨勢狀態（向量化）──────────────────────────────────────────
print("計算每日趨勢狀態（rolling 40日高低點）...")
rolling_min40 = prices.rolling(40, min_periods=20).min()
rolling_max40 = prices.rolling(40, min_periods=20).max()
# bounce_pct_df[date][sym] = (current / low_40d - 1) * 100
bounce_pct_df  = (prices / rolling_min40 - 1) * 100
from_high_df   = (prices / rolling_max40 - 1) * 100
print("預計算完成\n")


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

def get_trend_states(ti) -> dict:
    """回傳 {sym: "轉強"/"轉弱"/"盤整"}（向量化查詢）"""
    bp  = bounce_pct_df.iloc[ti]
    fh  = from_high_df.iloc[ti]
    valid = bp.notna() & fh.notna()
    states = {}
    for sym in prices.columns:
        if not valid.get(sym, False):
            states[sym] = "盤整"
        elif bp[sym] > 20 and fh[sym] > -5:
            states[sym] = "轉強"
        elif fh[sym] < -15:
            states[sym] = "轉弱"
        else:
            states[sym] = "盤整"
    return states

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
results         = {k: [] for k in ["A", "B", "C", "D"]}
trend_log       = []   # 記錄趨勢狀態分布

print("回測中...")
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG + 40:
        continue

    breadth = get_breadth(ti)
    if breadth is None:
        continue

    # top-20 動能池
    pool = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, ti)
        if m is None or m <= 0:
            continue
        pool.append({"sym": sym, "mom": m})
    if len(pool) < TOP_N:
        continue

    df_pool = pd.DataFrame(pool).nlargest(TOP_POOL, "mom").copy()

    # 趨勢狀態
    states = get_trend_states(ti)
    df_pool["state"] = df_pool["sym"].map(states).fillna("盤整")

    # 廣度加權 n_picks
    n_breadth = max(1, min(TOP_N, round(breadth * TOP_N)))

    spy_1m = spy_fwd(ref.date(), 21)
    spy_3m = spy_fwd(ref.date(), 63)

    # ─ 策略 A：純動能 top5 ──────────────────────────────────────────
    picks_A = df_pool["sym"].head(TOP_N).tolist()

    # ─ 策略 B：廣度加權 n_picks ──────────────────────────────────────
    picks_B = df_pool["sym"].head(n_breadth).tolist()

    # ─ 策略 C：廣度加權 + 排除 ↘️ ───────────────────────────────────
    df_no_down = df_pool[df_pool["state"] != "轉弱"]
    # 若排除後不足，仍從原池補（否則過度空手）
    if len(df_no_down) >= n_breadth:
        picks_C = df_no_down["sym"].head(n_breadth).tolist()
    else:
        picks_C = df_pool["sym"].head(n_breadth).tolist()

    # ─ 策略 D：廣度加權 + 趨勢加分 ──────────────────────────────────
    BONUS = 10.0   # ↗️ 加 10%
    df_pool_D = df_pool.copy()
    df_pool_D["adj_mom"] = df_pool_D.apply(
        lambda r: r["mom"] + BONUS if r["state"] == "轉強" else r["mom"], axis=1
    )
    df_pool_D = df_pool_D.sort_values("adj_mom", ascending=False)
    picks_D = df_pool_D["sym"].head(n_breadth).tolist()

    # ─ 計算報酬 ──────────────────────────────────────────────────────
    for key, picks in [("A", picks_A), ("B", picks_B), ("C", picks_C), ("D", picks_D)]:
        if not picks:
            continue
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
            "breadth":  breadth,
            "n_picks":  len(picks),
        })

    # 記錄趨勢狀態分布（top-20 中）
    state_counts = df_pool["state"].value_counts().to_dict()
    trend_log.append({
        "date":    str(ref.date()),
        "轉強":    state_counts.get("轉強", 0),
        "盤整":    state_counts.get("盤整", 0),
        "轉弱":    state_counts.get("轉弱", 0),
        "breadth": breadth,
    })

print()


# ── 彙總輸出 ──────────────────────────────────────────────────────────
print("=" * 72)
print("  多因子組合回測  2021-2025  S&P500 月度")
print("=" * 72)

desc = {
    "A": "A. 純動能基準               top5",
    "B": "B. 廣度加權                 n=round(breadth×5)",
    "C": "C. 廣度+排除↘️              廣度加權，排除轉弱",
    "D": "D. 廣度+趨勢加分            ↗️加10%，廣度加權重排",
}

summary = {}
for key in ["A", "B", "C", "D"]:
    rows = results[key]
    if not rows:
        continue
    df = pd.DataFrame(rows).dropna(subset=["alpha_1m"])
    summary[key] = {
        "n":      len(df),
        "med_1m": df["alpha_1m"].median(),
        "avg_1m": df["alpha_1m"].mean(),
        "med_3m": df["alpha_3m"].dropna().median() if "alpha_3m" in df.columns else None,
        "wins_vs_A": None,
    }

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
    wins_str = f"{s['wins_vs_A']}/{s.get('total_vs_A','?')}" if s.get("wins_vs_A") is not None else "  —"
    m3 = f"{s['med_3m']:+.2f}%" if s.get("med_3m") is not None else "  N/A"
    print(f"  {desc[key]:<46} {s['med_1m']:+.2f}%   {s['avg_1m']:+.2f}%   {m3:>8}   {wins_str:>7}")

# vs 基準改善量
print(f"\n  相對基準 A 的改善量")
print(f"  {'策略':<20} {'1M alpha 差':>12} {'3M alpha 差':>12}")
print(f"  {'-'*20} {'-------':>12} {'-------':>12}")

for key in ["B","C","D"]:
    diff_1m = summary[key]["med_1m"] - summary["A"]["med_1m"]
    diff_3m = (summary[key]["med_3m"] - summary["A"]["med_3m"]
               if summary[key].get("med_3m") and summary["A"].get("med_3m") else None)
    d3 = f"{diff_3m:+.2f}%" if diff_3m is not None else "  N/A"
    print(f"  {desc[key][:20]:<20} {diff_1m:+.2f}%        {d3:>12}")


# ── 趨勢狀態分布分析 ──────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  top-20 中的趨勢狀態分布（月度平均）")
print(f"{'='*72}\n")

df_tl = pd.DataFrame(trend_log)
avg_up   = df_tl["轉強"].mean()
avg_flat = df_tl["盤整"].mean()
avg_down = df_tl["轉弱"].mean()
print(f"  平均每月 top-20 中：↗️ 轉強 {avg_up:.1f} 支 / → 盤整 {avg_flat:.1f} 支 / ↘️ 轉弱 {avg_down:.1f} 支")

# 高廣度 vs 低廣度 趨勢狀態
high_b = df_tl[df_tl["breadth"] >= 0.60]
low_b  = df_tl[df_tl["breadth"] <  0.45]
if len(high_b) > 0:
    print(f"\n  廣度≥60%（{len(high_b)}個月）：↗️ {high_b['轉強'].mean():.1f} / → {high_b['盤整'].mean():.1f} / ↘️ {high_b['轉弱'].mean():.1f}")
if len(low_b) > 0:
    print(f"  廣度<45%（{len(low_b)}個月）：↗️ {low_b['轉強'].mean():.1f} / → {low_b['盤整'].mean():.1f} / ↘️ {low_b['轉弱'].mean():.1f}")

# 策略 C 實際排除了多少 ↘️
total_excluded = sum(
    max(0, round(r["breadth"]*5)) - len([s for s in results["C"] if s["date"] == r["date"]])
    for r in trend_log
    if any(s["date"] == r["date"] for s in results["C"])
)

print(f"\n{'='*72}")
