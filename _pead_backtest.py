"""
PEAD（財報後慣性）回測

財報驚喜後股價通常在 1-3 個月內持續漂移，是學術上最穩固的 anomaly 之一。
問題：在動能候選池中，加入「最近財報 beat」的偏好，能否提升報酬？

測試四種策略：
  A（基準）：top5 mom_mixed（現行做法）
  B（PEAD 優先）：從 top15 mom，優先選 45 天內有財報 beat 的 5 支
  C（複合分數）：top5 by (mom_mixed + 0.5 × surprise_pct)，限最近 45 天
  D（排除 miss）：top5 mom，過濾掉 45 天內有財報 miss 的標的

資料：S&P500 2021-2025，月度再平衡
快取：
  - data/_protection_bt_prices.pkl（價格，與其他回測共用）
  - data/_earnings_cache.pkl（財報日 + 驚喜 %，首次需下載約 5-10 分鐘）
"""
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from src.data_loader import get_sp500_tickers

PRICE_CACHE    = "data/_protection_bt_prices.pkl"
EARNINGS_CACHE = "data/_earnings_cache.pkl"
MOM_SHORT      = 21
MOM_LONG       = 252
TOP_N          = 5
POOL_B         = 15   # 策略 B/C/D 的候選池大小
BEAT_WINDOW    = 45   # 財報 beat 的時效（天）
SURPRISE_CAP   = 50   # surprise% 上限（避免極端值主導）
BATCH_SIZE     = 50


# ── 1. 載入價格資料 ───────────────────────────────────────────────────
if not os.path.exists(PRICE_CACHE):
    print(f"找不到快取 {PRICE_CACHE}，請先執行 _protection_period_backtest.py")
    raise SystemExit(1)

print(f"讀取價格快取：{PRICE_CACHE}")
prices = pd.read_pickle(PRICE_CACHE)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()
print(f"價格：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日")


# ── 2. 財報資料（下載或讀快取）───────────────────────────────────────
def download_earnings(tickers):
    """下載所有股票的財報驚喜資料，回傳 {sym: DataFrame}"""
    result = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for bi, batch in enumerate(batches):
        print(f"  財報批次 {bi+1}/{len(batches)}...", end=" ", flush=True)
        ok = 0
        for sym in batch:
            try:
                df = yf.Ticker(sym).get_earnings_dates(limit=40)
                if df is not None and not df.empty:
                    # 統一時區：轉為 tz-naive
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert(None)
                    result[sym] = df
                    ok += 1
            except Exception:
                pass
        print(f"OK ({ok}/{len(batch)} 檔)")
        time.sleep(0.3)
    return result

tickers = get_sp500_tickers()

if os.path.exists(EARNINGS_CACHE):
    print(f"讀取財報快取：{EARNINGS_CACHE}")
    earnings_data = pd.read_pickle(EARNINGS_CACHE)
else:
    print(f"首次下載財報資料（{len(tickers)} 檔）...")
    earnings_data = download_earnings(tickers)
    pd.to_pickle(earnings_data, EARNINGS_CACHE)
    print(f"已快取至 {EARNINGS_CACHE}")

print(f"財報資料：{len(earnings_data)} 檔有效\n")


# ── 3. SPY 基準 ───────────────────────────────────────────────────────
spy = yf.Ticker("SPY").history(start="2020-12-01", end="2026-03-01",
                               auto_adjust=True)["Close"]
spy.index = pd.to_datetime(spy.index).tz_localize(None)


# ── 4. 工具函式 ───────────────────────────────────────────────────────
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

def get_earnings_info(sym, ref_date):
    """
    回傳最近一次財報的 (days_ago, surprise_pct)。
    surprise_pct > 0 = beat，< 0 = miss，None = 無資料。
    """
    df = earnings_data.get(sym)
    if df is None or df.empty:
        return None, None
    ref_ts = pd.Timestamp(ref_date)
    # 只看 ref_date 之前的（避免未來資料），且需有 Surprise(%) 欄
    if "Surprise(%)" not in df.columns:
        return None, None
    past = df[df.index < ref_ts].dropna(subset=["Surprise(%)"])
    if past.empty:
        return None, None
    most_recent = past.iloc[0]
    days_ago = (ref_ts - most_recent.name).days
    surprise = float(most_recent["Surprise(%)"])
    return days_ago, surprise

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


# ── 5. 主回測循環 ─────────────────────────────────────────────────────
rebalance_dates = pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")
results = {k: [] for k in ["A", "B", "C", "D"]}

print("回測中...")
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG:
        continue

    # 計算全市場動能 + 財報資訊
    pool = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, ti)
        if m is None or m <= 0:
            continue
        days_ago, surprise = get_earnings_info(sym, ref.date())
        is_recent_beat  = (days_ago is not None and days_ago <= BEAT_WINDOW and surprise is not None and surprise > 0)
        is_recent_miss  = (days_ago is not None and days_ago <= BEAT_WINDOW and surprise is not None and surprise < 0)
        # 複合分數：只有最近 beat 才加分
        surprise_bonus = min(surprise, SURPRISE_CAP) * 0.5 if is_recent_beat else 0.0
        pool.append({
            "sym":         sym,
            "mom":         m,
            "days_ago":    days_ago,
            "surprise":    surprise,
            "recent_beat": is_recent_beat,
            "recent_miss": is_recent_miss,
            "composite":   m + surprise_bonus,
        })

    if len(pool) < POOL_B:
        continue

    df_pool = pd.DataFrame(pool)
    top15   = df_pool.nlargest(POOL_B, "mom")
    picks   = {}

    # A：純動能（基準）
    picks["A"] = df_pool.nlargest(TOP_N, "mom")["sym"].tolist()

    # B：從 top15 mom 中，優先選有 recent beat 的（不足則補）
    b_beat = top15[top15["recent_beat"]].nlargest(TOP_N, "mom")
    b_rest = top15[~top15["sym"].isin(b_beat["sym"])].nlargest(TOP_N - len(b_beat), "mom")
    picks["B"] = pd.concat([b_beat, b_rest]).head(TOP_N)["sym"].tolist()

    # C：top5 by composite（mom + surprise bonus）
    picks["C"] = top15.nlargest(TOP_N, "composite")["sym"].tolist()

    # D：top5 mom，排除 recent miss
    d_no_miss = df_pool[~df_pool["recent_miss"]]
    if len(d_no_miss) >= TOP_N:
        picks["D"] = d_no_miss.nlargest(TOP_N, "mom")["sym"].tolist()
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
        beat_count = int(sub["recent_beat"].sum())
        results[key].append({
            "date":       str(ref.date()),
            "picks":      syms,
            "avg_1m":     avg_1m,
            "avg_3m":     avg_3m,
            "alpha_1m":   avg_1m - spy_1m if spy_1m else None,
            "alpha_3m":   avg_3m - spy_3m if (avg_3m and spy_3m) else None,
            "beat_count": beat_count,
        })

print()

# ── 6. 彙總輸出 ───────────────────────────────────────────────────────
print("=" * 72)
print("  PEAD 財報驚喜回測  2021-2025  S&P500 月度")
print("=" * 72)

desc = {
    "A": "A. 純動能（基準）        top5 mom_mixed",
    "B": "B. PEAD 優先             top15 mom → beat 優先",
    "C": "C. 複合分數              top15 (mom + 0.5×surprise)",
    "D": "D. 排除近期 miss         top5 mom，濾掉 45天內miss",
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
    avg_beat = df["beat_count"].mean()
    summary[key] = {"alpha_1m": alpha_1m, "alpha_3m": alpha_3m}
    print(f"\n  {desc[key]}")
    print(f"    平均 1M 報酬: {avg_1m:+.2f}%   3M: {avg_3m:+.2f}%   (n={n})")
    print(f"    vs SPY alpha: 1M {alpha_1m:+.2f}%   3M {alpha_3m:+.2f}%")
    print(f"    月勝率(>0): {win_1m:.0f}%   月跑贏SPY: {win_a:.0f}%")
    print(f"    選入時平均 beat 數: {avg_beat:.1f}/5 支")

print()
print("=" * 72)
print("  相對基準 A 的改善量")
print(f"  {'策略':<42} {'1M alpha 差':>12} {'3M alpha 差':>12}")
print("  " + "-" * 68)
for key in ["B", "C", "D"]:
    if key in summary and "A" in summary:
        d1 = summary[key]["alpha_1m"] - summary["A"]["alpha_1m"]
        d3 = summary[key]["alpha_3m"] - summary["A"]["alpha_3m"]
        print(f"  {desc[key]:<42} {d1:>+11.2f}%  {d3:>+11.2f}%")

# ── 7. PEAD 核心分析：beat vs no-beat 的報酬差異 ─────────────────────
print()
print("=" * 72)
print("  PEAD 核心驗證：策略 A 選到的股票，有 recent beat vs 沒有")
print("  （排除「沒有財報資料」的個案）")
print()

all_events = []
for row in results["A"]:
    ti = tidx(row["date"])
    for sym in row["picks"]:
        days_ago, surprise = get_earnings_info(sym, row["date"])
        r1 = fwd_return_sym(sym, ti, 21)
        r3 = fwd_return_sym(sym, ti, 63)
        s1 = spy_fwd(row["date"], 21)
        s3 = spy_fwd(row["date"], 63)
        if None in (r1, s1):
            continue
        all_events.append({
            "date":      row["date"],
            "sym":       sym,
            "days_ago":  days_ago,
            "surprise":  surprise,
            "r1": r1, "r3": r3,
            "alpha_1m":  r1 - s1,
            "alpha_3m":  (r3 - s3) if (r3 and s3) else None,
        })

df_ev = pd.DataFrame(all_events)

# 分組
df_beat  = df_ev[df_ev["days_ago"].notna() & (df_ev["days_ago"] <= BEAT_WINDOW) & (df_ev["surprise"] > 0)]
df_miss  = df_ev[df_ev["days_ago"].notna() & (df_ev["days_ago"] <= BEAT_WINDOW) & (df_ev["surprise"] < 0)]
df_stale = df_ev[df_ev["days_ago"].notna() & (df_ev["days_ago"] > BEAT_WINDOW)]
df_none  = df_ev[df_ev["days_ago"].isna()]

print(f"  {'分組':<25} {'1M報酬':>9} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 60)
for label, sub in [
    (f"近期 beat（≤{BEAT_WINDOW}天）", df_beat),
    (f"近期 miss（≤{BEAT_WINDOW}天）", df_miss),
    (f"財報過期（>{BEAT_WINDOW}天）", df_stale),
    ("無財報資料", df_none),
]:
    if len(sub) == 0:
        continue
    r1 = sub["r1"].mean()
    a1 = sub["alpha_1m"].dropna().mean()
    a3 = sub["alpha_3m"].dropna().mean()
    print(f"  {label:<25} {r1:>+8.2f}%  {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")

# ── 8. Surprise 大小的影響 ────────────────────────────────────────────
print()
print("=" * 72)
print(f"  Surprise 大小分組（45 天內有財報資料的 {len(df_ev[df_ev['days_ago'].notna() & (df_ev['days_ago'] <= BEAT_WINDOW)])} 筆）")
print()

df_recent = df_ev[df_ev["days_ago"].notna() & (df_ev["days_ago"] <= BEAT_WINDOW) & df_ev["surprise"].notna()]
if len(df_recent) >= 10:
    bins   = [-999, -5, 0, 5, 15, 999]
    labels = ["大幅miss(<-5%)", "小幅miss(-5%~0%)", "小幅beat(0%~5%)", "中幅beat(5%~15%)", "大幅beat(>15%)"]
    df_recent = df_recent.copy()
    df_recent["bucket"] = pd.cut(df_recent["surprise"], bins=bins, labels=labels)
    print(f"  {'Surprise 區間':<22} {'1M alpha':>10} {'3M alpha':>10}  n")
    print("  " + "-" * 48)
    for label in labels:
        sub = df_recent[df_recent["bucket"] == label]
        if len(sub) < 3:
            continue
        a1 = sub["alpha_1m"].dropna().mean()
        a3 = sub["alpha_3m"].dropna().mean()
        print(f"  {label:<22} {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")

# ── 9. 逐月明細 ───────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"  逐月 1M Alpha（vs SPY）")
print(f"  {'日期':<12} {'A 基準':>8} {'B PEAD':>8} {'C 複合':>8} {'D 過濾':>8}  最佳")
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
