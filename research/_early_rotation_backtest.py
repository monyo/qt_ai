"""
_early_rotation_backtest.py

問題：CF 型標的（進場後漲一點就還回去）能否透過「提早識別並輪換」改善報酬？

核心假設：
  CF 型 = 進場後動能快速消退，21 天後已不在 top-N
  SNDK 型 = 持續強勢，每次再平衡仍在 top-N

比較策略（持倉最多 5 槽，每 21 天再平衡）：
  A  標準       持有至停損或最長 63 天，不做中期輪換
  B  積極輪換   每 21 天重新評估：不在 top-N 就賣出，補進新 top-N
  C  寬鬆輪換   每 21 天重新評估：跌出 top-15 才賣（給一點緩衝）
  D  延遲輪換   持有 42 天後再開始輪換評估（折衷方案）

執行：
    conda run -n qt_env python _early_rotation_backtest.py
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
    print(f"❌ 找不到 {OHLCV_PATH}"); sys.exit(1)

print("載入 OHLCV 快取...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]

df_close.index = pd.to_datetime(df_close.index)
if df_close.index.tz is not None:
    df_close.index = df_close.index.tz_convert(None)

trading_days = df_close.index.sort_values()
n_dates      = len(trading_days)
symbols      = list(df_close.columns)
print(f"  {len(symbols)} 支  /  {n_dates} 交易日  "
      f"({trading_days[0].date()} ~ {trading_days[-1].date()})")

print("取得 SPY...")
spy_raw = yf.Ticker("SPY").history(
    start="2020-12-01", end="2026-03-01", auto_adjust=True)["Close"]
spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)

# ── 參數 ─────────────────────────────────────────────────────────────────────
MOM_SHORT  = 21
MOM_LONG   = 252
N_SLOTS    = 5
REBAL_FREQ = 21
FIXED_STOP = 0.15     # -15% 固定停損
TRAIL_STOP = 0.25     # -25% 追蹤停損

START_TI  = MOM_LONG + 5
rebal_tis = list(range(START_TI, n_dates - REBAL_FREQ - 5, REBAL_FREQ))

# ── 工具 ─────────────────────────────────────────────────────────────────────
def price_at(sym, ti):
    if ti < 0 or ti >= n_dates: return None
    v = df_close[sym].iloc[ti]
    return float(v) if pd.notna(v) else None

def mom_mixed_at(sym, ti):
    if ti < MOM_LONG: return None
    p0   = price_at(sym, ti)
    p21  = price_at(sym, ti - MOM_SHORT)
    p252 = price_at(sym, ti - MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0): return None
    return 0.5 * (p0/p21 - 1) + 0.5 * (p0/p252 - 1)

def get_ranked(ti, top_k=None):
    """回傳動能排名後的 [(sym, score), ...]"""
    scores = {}
    for sym in symbols:
        m = mom_mixed_at(sym, ti)
        if m is not None:
            scores[sym] = m
    ranked = sorted(scores, key=scores.get, reverse=True)
    if top_k:
        return ranked[:top_k]
    return ranked

def spy_px(ti):
    try:
        return float(spy_raw.asof(trading_days[ti]))
    except Exception:
        return None

# ── 投組模擬 ─────────────────────────────────────────────────────────────────
# position: {sym: {entry_ti, entry_px, peak_px, hold_days}}

def run_portfolio_sim(strategy):
    """
    strategy: 'standard' | 'aggressive' | 'loose' | 'delayed'
    """
    portfolio   = {}   # {sym: {entry_ti, entry_px, peak_px}}
    equity      = 1.0
    eq_hist     = [1.0]
    spy_hist    = [1.0]
    rotate_out  = 0    # 因排名輪換出場次數
    stop_out    = 0    # 因停損出場次數
    time_out    = 0    # 因持滿時間出場次數
    spy0        = spy_px(rebal_tis[0])

    for ri, ti in enumerate(rebal_tis[:-1]):
        ti_next = rebal_tis[ri + 1]

        # ── 日內停損檢查（ti 到 ti_next）────────────────────────────────────
        exited = []
        for sym, pos in list(portfolio.items()):
            entry_px = pos["entry_px"]
            peak_px  = pos["peak_px"]
            exit_px  = None
            stop_hit = False

            for day in range(ti, ti_next):
                px = price_at(sym, day)
                if px is None: continue
                if px > peak_px:
                    portfolio[sym]["peak_px"] = px
                    peak_px = px
                fixed_line = entry_px * (1 - FIXED_STOP)
                trail_line = peak_px  * (1 - TRAIL_STOP)
                if px < max(fixed_line, trail_line):
                    exit_px  = px
                    stop_hit = True
                    break

            if stop_hit:
                exited.append((sym, exit_px, "stop"))

        for sym, exit_px, reason in exited:
            if sym in portfolio:
                ret = (exit_px - portfolio[sym]["entry_px"]) / portfolio[sym]["entry_px"]
                equity *= (1 + ret / N_SLOTS)
                del portfolio[sym]
                stop_out += 1

        # ── 再平衡時的輪換邏輯 ───────────────────────────────────────────────
        hold_days_map = {sym: (ti - pos["entry_ti"]) for sym, pos in portfolio.items()}

        if strategy == "standard":
            # 不做主動輪換，只補空缺
            pass

        elif strategy == "aggressive":
            # 每 21 天：不在 top-N 就賣
            top_n_set = set(get_ranked(ti, top_k=N_SLOTS))
            for sym in list(portfolio.keys()):
                if sym not in top_n_set:
                    exit_px = price_at(sym, ti)
                    if exit_px:
                        ret = (exit_px - portfolio[sym]["entry_px"]) / portfolio[sym]["entry_px"]
                        equity *= (1 + ret / N_SLOTS)
                    del portfolio[sym]
                    rotate_out += 1

        elif strategy == "loose":
            # 每 21 天：跌出 top-15 才賣（緩衝）
            top15_set = set(get_ranked(ti, top_k=15))
            for sym in list(portfolio.keys()):
                if sym not in top15_set:
                    exit_px = price_at(sym, ti)
                    if exit_px:
                        ret = (exit_px - portfolio[sym]["entry_px"]) / portfolio[sym]["entry_px"]
                        equity *= (1 + ret / N_SLOTS)
                    del portfolio[sym]
                    rotate_out += 1

        elif strategy == "delayed":
            # 持有 42 天後才開始評估
            top_n_set = set(get_ranked(ti, top_k=N_SLOTS))
            for sym in list(portfolio.keys()):
                if hold_days_map.get(sym, 0) >= 42 and sym not in top_n_set:
                    exit_px = price_at(sym, ti)
                    if exit_px:
                        ret = (exit_px - portfolio[sym]["entry_px"]) / portfolio[sym]["entry_px"]
                        equity *= (1 + ret / N_SLOTS)
                    del portfolio[sym]
                    rotate_out += 1

        # ── 時間到期出場（standard：滿 63 天）────────────────────────────────
        if strategy == "standard":
            for sym in list(portfolio.keys()):
                if hold_days_map.get(sym, 0) >= 63:
                    exit_px = price_at(sym, ti)
                    if exit_px:
                        ret = (exit_px - portfolio[sym]["entry_px"]) / portfolio[sym]["entry_px"]
                        equity *= (1 + ret / N_SLOTS)
                    del portfolio[sym]
                    time_out += 1
        else:
            # 非 standard：持有至停損或被輪換，無硬性時間上限
            # 但設 126 天防止永久持有
            for sym in list(portfolio.keys()):
                if hold_days_map.get(sym, 0) >= 126:
                    exit_px = price_at(sym, ti)
                    if exit_px:
                        ret = (exit_px - portfolio[sym]["entry_px"]) / portfolio[sym]["entry_px"]
                        equity *= (1 + ret / N_SLOTS)
                    del portfolio[sym]
                    time_out += 1

        # ── 補進新部位 ───────────────────────────────────────────────────────
        n_empty = N_SLOTS - len(portfolio)
        if n_empty > 0:
            ranked = get_ranked(ti, top_k=N_SLOTS * 4)
            added  = 0
            for sym in ranked:
                if sym in portfolio:
                    continue
                entry_px = price_at(sym, ti)
                if entry_px is None:
                    continue
                portfolio[sym] = {
                    "entry_ti": ti,
                    "entry_px": entry_px,
                    "peak_px":  entry_px,
                }
                added += 1
                if added >= n_empty:
                    break

        # 記錄本期 equity 與 SPY
        eq_hist.append(equity)
        spy_cur = spy_px(ti_next)
        if spy0 and spy_cur:
            spy_hist.append(spy_cur / spy0)
        else:
            spy_hist.append(spy_hist[-1])

    # ── 計算指標 ─────────────────────────────────────────────────────────────
    n_years = len(rebal_tis) * REBAL_FREQ / 252
    cagr    = equity ** (1 / n_years) - 1

    eq = np.array(eq_hist)
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk).min())

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    total_exits = rotate_out + stop_out + time_out
    return {
        "cagr":       cagr,
        "mdd":        mdd,
        "calmar":     calmar,
        "rotate_out": rotate_out,
        "stop_out":   stop_out,
        "time_out":   time_out,
        "total_exits":total_exits,
    }

# ── 執行 ─────────────────────────────────────────────────────────────────────
STRATEGIES = [
    ("A 標準 (持63天)",    "standard"),
    ("B 積極輪換 (top-5)", "aggressive"),
    ("C 寬鬆輪換 (top-15)","loose"),
    ("D 延遲輪換 (42天後)","delayed"),
]

print(f"\n再平衡點：{len(rebal_tis)} 個  /  {N_SLOTS} 槽  /  固定停損 -{FIXED_STOP*100:.0f}%  追蹤停損 -{TRAIL_STOP*100:.0f}%\n")

results = []
for label, strat in STRATEGIES:
    print(f"  執行 {label}...", end="", flush=True)
    r = run_portfolio_sim(strat)
    r["name"] = label
    results.append(r)
    print(f"  CAGR {r['cagr']*100:+.1f}%  MDD {r['mdd']*100:.1f}%  Calmar {r['calmar']:.3f}"
          f"  輪換:{r['rotate_out']}次  停損:{r['stop_out']}次")

# ── 輸出 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  提早輪換回測  S&P500 ~{n_dates//252:.0f}Y  {N_SLOTS}槽  月度再平衡")
print(f"{'='*72}")
print(f"  {'策略':<22}  {'CAGR':>7}  {'MDD':>7}  {'Calmar':>7}  "
      f"{'輪換出':>7}  {'停損出':>7}")
print(f"  {'-'*68}")

best_calmar = max(r["calmar"] for r in results)
for r in results:
    mark = " ✅" if r["calmar"] == best_calmar else ""
    print(f"  {r['name']:<22}  {r['cagr']*100:>+6.1f}%  "
          f"{r['mdd']*100:>6.1f}%  {r['calmar']:>7.3f}  "
          f"{r['rotate_out']:>7}  {r['stop_out']:>7}{mark}")

print(f"\n  策略說明：")
print(f"    A 標準       ：持有至停損或 63 天，不主動輪換")
print(f"    B 積極輪換   ：每 21 天，若不在 top-5 即換出")
print(f"    C 寬鬆輪換   ：每 21 天，若跌出 top-15 才換出（給緩衝）")
print(f"    D 延遲輪換   ：持有滿 42 天後，若不在 top-5 才換出")
print(f"\n  ⚠️  Survivorship bias：僅含現有 S&P500 成份股")
