"""
主動出場訊號回測 v2：動能衰退偵測

v1 發現：「動能跌破 0」在 63 天持有期內幾乎不觸發，
因為 252 天動能是慢變量，在持有期內很難從正轉負。

問題核心：CF 的動能還是正的，只是在衰退。
需要偵測的是「強度下降」，不是「轉負」。

測試四種策略：
  A（基準）：純停損（固定 -15%，追蹤 -25%）
  B（動能衰退）：進場後動能分數下降超過 50%，下一再平衡點出場
  C（短線反轉）：持倉價格跌破 15 天均線，且距入場高點 > -8%
  D（組合）：B 或 C 任一觸發

分析重點：
  - 觸發率（這些訊號實際會發生多少次？）
  - False Exit：出場後 21 天標的是否繼續漲
  - 平均報酬 vs 策略 A
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf

# ── 參數 ──────────────────────────────────────────────────────────────────
OHLCV_PATH  = "data/_protection_bt_ohlcv.pkl"
CACHE_PATH  = "data/_protection_bt_prices.pkl"
MOM_SHORT   = 21
MOM_LONG    = 252
TOP_N       = 5
REBAL_FREQ  = 21
HOLD_MAX    = 63
FIXED_STOP  = 0.15
TRAIL_STOP  = 0.25

MOM_DECAY_THRESH = 0.50   # 策略 B：動能分數衰退超過此比例觸發
MA_WIN_SHORT     = 15     # 策略 C：短線均線天數
PEAK_DRAWDOWN    = 0.08   # 策略 C：距入場高點回落超過此值才確認

# ── 載入資料 ──────────────────────────────────────────────────────────────
print("載入資料...")
if os.path.exists(OHLCV_PATH):
    ohlcv = pd.read_pickle(OHLCV_PATH)
    prices = ohlcv["Close"]
else:
    prices = pd.read_pickle(CACHE_PATH)

prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
prices = prices.sort_index()
trading_days = prices.index
n_dates = len(trading_days)
symbols = list(prices.columns)
close_arr = prices.values.astype(float)
sym_idx = {s: i for i, s in enumerate(symbols)}
print(f"資料：{len(symbols)} 支 / {n_dates} 交易日")

spy_raw = yf.Ticker("SPY").history(
    start="2020-12-01", end="2026-03-01", auto_adjust=True
)["Close"]
spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)

START_TI = MOM_LONG + 5
rebal_tis = list(range(START_TI, n_dates - HOLD_MAX - 5, REBAL_FREQ))
print(f"再平衡點：{len(rebal_tis)} 個\n")


# ── 工具函式 ──────────────────────────────────────────────────────────────
def price_at(sym, ti):
    if ti < 0 or ti >= n_dates:
        return None
    v = close_arr[ti, sym_idx[sym]]
    return float(v) if np.isfinite(v) else None


def mom_mixed_at(sym, ti):
    if ti < MOM_LONG:
        return None
    si = sym_idx[sym]
    def p(off):
        idx = ti - off
        if idx < 0: return None
        v = close_arr[idx, si]
        return float(v) if np.isfinite(v) else None
    p0, p21, p252 = p(0), p(MOM_SHORT), p(MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
        return None
    return 0.5 * (p0/p21 - 1) * 100 + 0.5 * (p0/p252 - 1) * 100


def ma_at(sym, ti, window):
    """ti 當日的 window 天收盤均線"""
    si = sym_idx[sym]
    seg = close_arr[max(0, ti - window + 1): ti + 1, si]
    seg = seg[np.isfinite(seg)]
    if len(seg) < window // 2:
        return None
    return float(seg.mean())


def spy_ret(ti, offset):
    ref_date = trading_days[ti]
    idx = spy_raw.index.searchsorted(ref_date)
    if idx + offset >= len(spy_raw):
        return None
    return float(spy_raw.iloc[idx + offset]) / float(spy_raw.iloc[idx]) - 1


# ── 模擬持倉 ──────────────────────────────────────────────────────────────
def simulate(sym, ti_entry, strategy):
    p0 = price_at(sym, ti_entry)
    if not p0:
        return None

    mom_entry = mom_mixed_at(sym, ti_entry)  # 進場時的動能分數

    fixed_px = p0 * (1 - FIXED_STOP)
    peak = p0
    peak_gain = 0.0

    for day in range(1, HOLD_MAX + 1):
        ti = ti_entry + day
        if ti >= n_dates:
            break

        px = price_at(sym, ti)
        if px is None:
            continue

        if px > peak:
            peak = px
            peak_gain = (peak - p0) / p0

        # ── 停損（所有策略共用）──
        trail_px = peak * (1 - TRAIL_STOP)
        active_stop = max(fixed_px, trail_px)
        if px < active_stop:
            stop_type = "fixed" if fixed_px >= trail_px else "trail"
            return {
                "ret": (px - p0) / p0,
                "exit_day": day,
                "exit_type": f"stop_{stop_type}",
                "peak_gain": peak_gain,
                "ti_exit": ti,
            }

        # ── 主動出場訊號（B/C/D）──
        if strategy in ("B", "D") and mom_entry is not None and mom_entry > 0:
            m_now = mom_mixed_at(sym, ti)
            if m_now is not None:
                decay = (mom_entry - m_now) / abs(mom_entry)
                if decay > MOM_DECAY_THRESH:
                    return {
                        "ret": (px - p0) / p0,
                        "exit_day": day,
                        "exit_type": "mom_decay",
                        "peak_gain": peak_gain,
                        "ti_exit": ti,
                    }

        if strategy in ("C", "D"):
            ma15 = ma_at(sym, ti, MA_WIN_SHORT)
            from_peak = (px - peak) / peak
            if ma15 is not None and px < ma15 and from_peak < -PEAK_DRAWDOWN:
                return {
                    "ret": (px - p0) / p0,
                    "exit_day": day,
                    "exit_type": "ma_break",
                    "peak_gain": peak_gain,
                    "ti_exit": ti,
                }

    px_end = price_at(sym, min(ti_entry + HOLD_MAX, n_dates - 1))
    return {
        "ret": (px_end - p0) / p0 if px_end else 0.0,
        "exit_day": HOLD_MAX,
        "exit_type": "time",
        "peak_gain": peak_gain,
        "ti_exit": ti_entry + HOLD_MAX,
    }


# ── 執行回測 ──────────────────────────────────────────────────────────────
strategies = ["A", "B", "C", "D"]
records = {s: [] for s in strategies}

print("回測中...")
for ti in rebal_tis:
    scores = {}
    for sym in symbols:
        m = mom_mixed_at(sym, ti)
        if m is not None:
            scores[sym] = m
    top = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
    spy_r = spy_ret(ti, HOLD_MAX)

    for sym in top:
        for strat in strategies:
            r = simulate(sym, ti, strat)
            if r:
                r["sym"] = sym
                r["ti_entry"] = ti
                r["spy_ret"] = spy_r or 0.0
                r["alpha"] = r["ret"] - (spy_r or 0.0)
                records[strat].append(r)

dfs = {s: pd.DataFrame(records[s]) for s in strategies}
print("完成\n")


# ── False Exit 分析 ────────────────────────────────────────────────────────
def false_exit_analysis(df, exit_types):
    exits = df[df["exit_type"].isin(exit_types)].copy()
    if len(exits) == 0:
        return None
    post_rets = []
    for _, row in exits.iterrows():
        sym = row["sym"]
        ti_exit = int(row["ti_exit"])
        p0 = price_at(sym, ti_exit)
        p21 = price_at(sym, min(ti_exit + 21, n_dates - 1))
        if p0 and p21:
            post_rets.append((p21 - p0) / p0 * 100)
    if not post_rets:
        return None
    pr = pd.Series(post_rets)
    return {
        "n": len(exits),
        "continued_up_pct": (pr > 0).mean() * 100,
        "avg_post_ret": pr.mean(),
        "median_post_ret": pr.median(),
    }


# ── 輸出 ───────────────────────────────────────────────────────────────────
print("=" * 62)
print("  主動出場訊號回測 v2  S&P500 2021-2025")
print("=" * 62)
print("  A：純停損（現狀）")
print(f"  B：動能衰退 >{MOM_DECAY_THRESH*100:.0f}%（進場後動能分數下降）")
print(f"  C：跌破 MA{MA_WIN_SHORT} 且距入場高點 > -{PEAK_DRAWDOWN*100:.0f}%")
print("  D：B 或 C 任一觸發")
print()

summaries = []
for strat in strategies:
    df = dfs[strat]
    n = len(df)
    avg_ret  = df["ret"].mean() * 100
    med_ret  = df["ret"].median() * 100
    avg_alpha = df["alpha"].mean() * 100
    avg_hold = df["exit_day"].mean()
    win_rate = (df["ret"] > 0).mean() * 100

    ec = df["exit_type"].value_counts()
    stop_n  = ec.get("stop_fixed", 0) + ec.get("stop_trail", 0)
    active_n = ec.get("mom_decay", 0) + ec.get("ma_break", 0)
    time_n  = ec.get("time", 0)

    # Calmar 近似
    avg_peak = df["peak_gain"].mean() * 100
    mdd_proxy = abs((df["ret"] - df["peak_gain"]).mean() * 100)
    calmar = (avg_ret * (252 / max(avg_hold, 1))) / max(mdd_proxy, 0.01)

    print(f"  {'─'*58}")
    print(f"  策略 {strat}  交易 {n} 次  平均報酬 {avg_ret:+.2f}%  Alpha {avg_alpha:+.2f}%")
    print(f"  中位數 {med_ret:+.2f}%  勝率 {win_rate:.0f}%  平均持有 {avg_hold:.0f} 天  Calmar {calmar:.3f}")
    print(f"  出場分布 → 停損：{stop_n}({stop_n/n*100:.0f}%)  "
          f"主動：{active_n}({active_n/n*100:.0f}%)  "
          f"時間：{time_n}({time_n/n*100:.0f}%)")

    # False Exit
    fe = false_exit_analysis(df, ["mom_decay", "ma_break"])
    if fe:
        print(f"  False Exit → 主動出場後 21 天：avg {fe['avg_post_ret']:+.2f}%，"
              f"繼續上漲 {fe['continued_up_pct']:.0f}%（共 {fe['n']} 次）")

    summaries.append({
        "strat": strat, "avg_ret": avg_ret, "avg_alpha": avg_alpha,
        "win_rate": win_rate, "avg_hold": avg_hold, "calmar": calmar,
        "stop_rate": stop_n/n, "active_rate": active_n/n,
    })

# 比較表
print(f"\n  {'─'*62}")
print(f"  {'策略':<6} {'平均報酬':>8} {'Alpha':>8} {'勝率':>6} {'持有天':>7} {'主動出場率':>10} {'Calmar':>8}")
print(f"  {'─'*62}")
for s in summaries:
    print(f"  {s['strat']:<6} {s['avg_ret']:>+8.2f}% {s['avg_alpha']:>+8.2f}% "
          f"{s['win_rate']:>5.0f}% {s['avg_hold']:>7.0f} "
          f"{s['active_rate']:>9.0%}  {s['calmar']:>8.3f}")
print(f"  {'─'*62}")

# ── 進階：動能衰退的分布（策略 B 觸發時的動能分數狀況）─────────────────
print(f"\n  {'='*62}")
print("  策略 B 觸發案例詳細分析（動能衰退出場）")
print(f"  {'='*62}")
b_mom_exits = dfs["B"][dfs["B"]["exit_type"] == "mom_decay"]
if len(b_mom_exits) > 0:
    print(f"  觸發次數：{len(b_mom_exits)}")
    print(f"  平均報酬：{b_mom_exits['ret'].mean()*100:+.2f}%")
    print(f"  平均峰值獲利：{b_mom_exits['peak_gain'].mean()*100:+.1f}%")
    print(f"  出場時點分布（exit_day）：")
    bins = [0, 10, 21, 42, 63]
    for i in range(len(bins)-1):
        cnt = ((b_mom_exits["exit_day"] > bins[i]) & (b_mom_exits["exit_day"] <= bins[i+1])).sum()
        print(f"    {bins[i]+1:>2}~{bins[i+1]:>2} 天：{cnt} 次")
else:
    print("  無觸發案例")

print(f"\n  {'='*62}")
print("  策略 C 觸發案例詳細分析（MA break 出場）")
print(f"  {'='*62}")
c_ma_exits = dfs["C"][dfs["C"]["exit_type"] == "ma_break"]
if len(c_ma_exits) > 0:
    print(f"  觸發次數：{len(c_ma_exits)}")
    print(f"  平均報酬：{c_ma_exits['ret'].mean()*100:+.2f}%")
    print(f"  平均峰值獲利：{c_ma_exits['peak_gain'].mean()*100:+.1f}%")
    fe_c = false_exit_analysis(dfs["C"], ["ma_break"])
    if fe_c:
        print(f"  出場後 21 天：avg {fe_c['avg_post_ret']:+.2f}%，"
              f"繼續上漲 {fe_c['continued_up_pct']:.0f}%")
else:
    print("  無觸發案例")
