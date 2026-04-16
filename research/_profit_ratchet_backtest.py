"""
_profit_ratchet_backtest.py

比較「利潤棘輪（Profit Ratchet）」vs 標準追蹤停損。

問題：進場後漲幅不大（+5~15%）的標的，標準 -25% 追蹤停損
      無法保護利潤，最後以虧損出場。若在達到某個浮盈門檻後
      縮緊追蹤停損，能否改善整體報酬？

策略：
  A  標準        固定 -15%（成本）+ 追蹤 -25%（高點）
  B  緊棘輪      浮盈 ≥+10% → 追蹤 -12%，≥+25% → -10%，≥+50% → -8%
  C  中棘輪      浮盈 ≥+15% → 追蹤 -15%，≥+30% → -12%，≥+50% → -10%
  D  寬棘輪      浮盈 ≥+20% → 追蹤 -15%，≥+40% → -12%，≥+60% → -10%
  E  只縮大贏家  浮盈 ≥+30% → 追蹤 -12%，≥+60% → -10%（不碰小贏家）

執行：
    conda run -n qt_env python _profit_ratchet_backtest.py
"""

import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ── 資料 ─────────────────────────────────────────────────────────────────────
OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
if not os.path.exists(OHLCV_PATH):
    print(f"❌ 找不到 {OHLCV_PATH}")
    sys.exit(1)

print("載入 OHLCV 快取...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
df_high  = ohlcv["High"]

df_close.index = pd.to_datetime(df_close.index)
df_high.index  = pd.to_datetime(df_high.index)
if df_close.index.tz is not None:
    df_close.index = df_close.index.tz_convert(None)
    df_high.index  = df_high.index.tz_convert(None)

trading_days = df_close.index.sort_values()
n_dates      = len(trading_days)
symbols      = list(df_close.columns)
print(f"  {len(symbols)} 支  /  {n_dates} 交易日  "
      f"({trading_days[0].date()} ~ {trading_days[-1].date()})")

# SPY 基準
print("取得 SPY 報酬...")
spy_raw = yf.Ticker("SPY").history(
    start="2020-12-01", end="2026-03-01", auto_adjust=True)["Close"]
spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)

def spy_ret(t0, t1):
    try:
        p0 = spy_raw.asof(trading_days[t0])
        p1 = spy_raw.asof(trading_days[t1])
        if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
            return None
        return float(p1 / p0 - 1)
    except Exception:
        return None

# ── 參數 ─────────────────────────────────────────────────────────────────────
MOM_SHORT  = 21
MOM_LONG   = 252
TOP_N      = 5
HOLD_MAX   = 63
REBAL_FREQ = 21

START_TI  = MOM_LONG + 5
rebal_tis = list(range(START_TI, n_dates - HOLD_MAX - 5, REBAL_FREQ))

# ── 棘輪策略定義 ─────────────────────────────────────────────────────────────
# 每個策略 = [(gain_threshold, trailing_pct), ...]（由低到高排列）
# 若無門檻觸發 → 使用標準 -25%

RATCHET_STRATEGIES = {
    "A 標準":   [],  # 空 = 純標準 -25%
    "B 緊棘輪": [(0.10, 0.12), (0.25, 0.10), (0.50, 0.08)],
    "C 中棘輪": [(0.15, 0.15), (0.30, 0.12), (0.50, 0.10)],
    "D 寬棘輪": [(0.20, 0.15), (0.40, 0.12), (0.60, 0.10)],
    "E 只大贏": [(0.30, 0.12), (0.60, 0.10)],
}

FIXED_STOP = -0.15    # 固定停損（成本以下 -15%）
STD_TRAIL  =  0.25    # 標準追蹤停損幅度（從高點）

def get_trail_pct(ratchet_rules, current_gain):
    """根據當前浮盈決定追蹤停損幅度（從高點算）"""
    trail = STD_TRAIL
    for gain_thr, new_trail in sorted(ratchet_rules, reverse=True):
        if current_gain >= gain_thr:
            trail = new_trail
            break
    return trail

# ── 工具函式 ─────────────────────────────────────────────────────────────────
def price_at(sym, ti):
    if ti < 0 or ti >= n_dates:
        return None
    v = df_close[sym].iloc[ti]
    return float(v) if pd.notna(v) else None

def mom_mixed_at(sym, ti):
    if ti < MOM_LONG:
        return None
    p0   = price_at(sym, ti)
    p21  = price_at(sym, ti - MOM_SHORT)
    p252 = price_at(sym, ti - MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
        return None
    return 0.5 * (p0/p21 - 1) + 0.5 * (p0/p252 - 1)

def simulate_hold_ratchet(sym, ti_entry, ratchet_rules):
    """
    模擬持有：每日更新高水位，根據當前浮盈選擇追蹤停損幅度。
    回傳 (exit_ti, exit_pct, stop_triggered)
    """
    p0 = price_at(sym, ti_entry)
    if not p0:
        return None, None, False

    fixed_px = p0 * (1 + FIXED_STOP)
    peak     = p0
    stop_hit = False
    exit_ti  = min(ti_entry + HOLD_MAX, n_dates - 1)
    exit_px  = None

    for ti in range(ti_entry + 1, min(ti_entry + HOLD_MAX + 1, n_dates)):
        px = price_at(sym, ti)
        if px is None:
            continue

        # 更新高水位
        if px > peak:
            peak = px

        # 計算當前浮盈（從成本計算，用於決定棘輪門檻）
        current_gain = (peak - p0) / p0   # 高水位浮盈

        # 動態追蹤停損
        trail_pct  = get_trail_pct(ratchet_rules, current_gain)
        trail_px   = peak * (1 - trail_pct)

        # 觸發停損？
        active_stop = max(fixed_px, trail_px)
        if px < active_stop:
            exit_ti  = ti
            exit_px  = px
            stop_hit = True
            break
    else:
        exit_px = price_at(sym, exit_ti)

    if exit_px is None:
        return None, None, False

    return exit_ti, (exit_px - p0) / p0, stop_hit

# ── 投組模擬 ─────────────────────────────────────────────────────────────────
def run_sim(strategy_name, ratchet_rules):
    """月度動能選股 + 停損模擬，回傳 equity curve"""
    equity      = 1.0
    equity_hist = [equity]
    stop_count  = 0
    trade_count = 0
    monthly_alpha = []

    for ri, ti in enumerate(rebal_tis):
        # 動能排名
        scores = {}
        for sym in symbols:
            m = mom_mixed_at(sym, ti)
            if m is not None:
                scores[sym] = m
        top = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
        if not top:
            continue

        # 持有期：到下次再平衡
        ti_end = rebal_tis[ri + 1] if ri + 1 < len(rebal_tis) else ti + REBAL_FREQ
        ti_end = min(ti_end, n_dates - 1)

        slot_rets  = []
        slot_alpha = []
        spy_r = spy_ret(ti, ti_end)

        for sym in top:
            exit_ti, ret, stopped = simulate_hold_ratchet(sym, ti, ratchet_rules)
            if ret is None:
                continue
            slot_rets.append(ret)
            if spy_r is not None:
                slot_alpha.append(ret - spy_r)
            trade_count += 1
            if stopped:
                stop_count += 1

        if slot_rets:
            period_ret = np.mean(slot_rets)
            equity    *= (1 + period_ret)
            equity_hist.append(equity)
            if slot_alpha:
                monthly_alpha.append(np.mean(slot_alpha))

    n_years = len(rebal_tis) * REBAL_FREQ / 252
    cagr    = equity ** (1 / n_years) - 1

    # MDD
    eq = np.array(equity_hist)
    peak_eq = np.maximum.accumulate(eq)
    dd      = (eq - peak_eq) / peak_eq
    mdd     = float(dd.min())

    calmar    = cagr / abs(mdd) if mdd != 0 else 0
    stop_rate = stop_count / trade_count if trade_count else 0
    med_alpha = float(np.median(monthly_alpha)) if monthly_alpha else 0

    return {
        "name":       strategy_name,
        "cagr":       cagr,
        "mdd":        mdd,
        "calmar":     calmar,
        "stop_rate":  stop_rate,
        "med_alpha":  med_alpha,
        "trades":     trade_count,
    }

# ── 執行所有策略 ─────────────────────────────────────────────────────────────
print(f"\n月度再平衡點：{len(rebal_tis)} 個  /  TOP-{TOP_N}  /  最長持有 {HOLD_MAX} 日\n")

results = []
for name, rules in RATCHET_STRATEGIES.items():
    print(f"  執行 {name}...", end="", flush=True)
    r = run_sim(name, rules)
    results.append(r)
    print(f"  CAGR {r['cagr']*100:+.1f}%  MDD {r['mdd']*100:.1f}%  "
          f"Calmar {r['calmar']:.3f}  停損率 {r['stop_rate']*100:.0f}%")

# ── 輸出 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  利潤棘輪回測  S&P500 ~{n_dates//252:.0f}Y  TOP-{TOP_N}  月度再平衡")
print(f"{'='*72}")
print(f"  {'策略':<18}  {'CAGR':>7}  {'MDD':>7}  {'Calmar':>7}  "
      f"{'停損率':>7}  {'月中位alpha':>10}")
print(f"  {'-'*65}")

best_calmar = max(r["calmar"] for r in results)
for r in results:
    marker = " ✅" if r["calmar"] == best_calmar else ""
    print(f"  {r['name']:<18}  {r['cagr']*100:>+6.1f}%  "
          f"{r['mdd']*100:>6.1f}%  {r['calmar']:>7.3f}  "
          f"{r['stop_rate']*100:>6.0f}%  "
          f"{r['med_alpha']*100:>+9.2f}%{marker}")

print(f"\n  棘輪規則說明：")
for name, rules in RATCHET_STRATEGIES.items():
    if not rules:
        print(f"    {name}：無棘輪，純標準 -25% 追蹤")
    else:
        tiers = "  ".join([f"浮盈≥{int(g*100)}%→追蹤-{int(t*100)}%" for g, t in rules])
        print(f"    {name}：{tiers}")

print(f"\n  ⚠️  注意：Survivorship bias（只含現有 S&P500 成份股）")
