"""
_holdperiod_backtest.py

回測問題：停損保護期（suppress stop-loss for first N days）是否改善績效？

比較策略：
  A. 標準：停損從第 1 天起就有效（現有系統）
  B. 保護30天：入場後前 30 交易日停損抑制
  C. 保護60天：入場後前 60 交易日停損抑制

其他條件完全相同：
  - 動能選股 top-5（21d+252d 混合）
  - 固定停損 -15%，追蹤停損 -25%
  - BULL/BEAR 體制（SPY MA200）
  - ROTATE 冷卻期 30 天（三策略相同）

使用：
    conda run -n qt_env python _holdperiod_backtest.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import get_sp500_tickers, fetch_stock_data
from portfolio_backtest import build_aligned_prices, calc_mixed_momentum

# ── 參數 ──────────────────────────────────────────────────────────────────────
SLOT_SIZE       = 8_667.0
MAX_SLOTS       = 30
FIXED_STOP      = -0.15
TRAILING_PCT    = 0.25
REBALANCE_DAYS  = 5
MOM_WARMUP      = 252
ROTATE_GAP      = 0.10
ROTATE_MIN_DAYS = 30
MA200_WINDOW    = 200
TOP_N           = 5

PROTECT_DAYS_LIST = [0, 30, 60]   # 0 = 現有系統（無保護）

# ── 載入價格 ──────────────────────────────────────────────────────────────────
print("載入價格資料...")
tickers = get_sp500_tickers() + ["SPY"]
raw = {}
for t in tickers:
    df = fetch_stock_data(t, period="10y")
    if df is not None and len(df) > 200:
        raw[t] = df["Close"]

aligned, common_dates = build_aligned_prices(raw)
syms = [s for s in aligned if s != "SPY"]
n = len(common_dates)
print(f"  {len(syms)} 支標的  /  {n} 個交易日  ({common_dates[0].date()} ~ {common_dates[-1].date()})")

# ── SPY 體制 ──────────────────────────────────────────────────────────────────
def spy_bull_regime(aligned, n):
    spy = aligned.get("SPY", np.zeros(n))
    bull = np.ones(n, dtype=bool)
    for i in range(n):
        if i >= MA200_WINDOW - 1:
            ma = np.mean(spy[i - MA200_WINDOW + 1: i + 1])
            bull[i] = spy[i] >= ma
    return bull

spy_bull = spy_bull_regime(aligned, n)

# ── 核心模擬 ──────────────────────────────────────────────────────────────────
def run_sim(protect_days: int) -> dict:
    """
    protect_days: 入場後幾個交易日內停損不觸發
    0 = 現有系統（無保護）
    """
    slots = []          # {sym, shares, avg_price, high_price, entry_idx}
    realized = 0.0
    total_invested = 0.0
    daily_pnl = []
    daily_market_val = []
    stop_loss_count = 0
    protected_stop_skip = 0   # 被保護期擋下的停損次數

    for d_idx in range(n):
        px = {s: aligned[s][d_idx] for s in aligned}

        # BEAR：全部出場
        if not spy_bull[d_idx]:
            for sl in slots:
                p = px.get(sl["sym"], np.nan)
                if not np.isnan(p) and p > 0:
                    realized += sl["shares"] * p
            slots = []
            daily_pnl.append(realized - total_invested)
            daily_market_val.append(0.0)
            continue

        # 更新最高點
        for sl in slots:
            p = px.get(sl["sym"], np.nan)
            if not np.isnan(p) and p > 0 and p > sl["high_price"]:
                sl["high_price"] = p

        # 出場：固定停損 + 追蹤停損
        surviving = []
        for sl in slots:
            p = px.get(sl["sym"], np.nan)
            if np.isnan(p) or p <= 0:
                surviving.append(sl)
                continue
            pnl     = (p - sl["avg_price"]) / sl["avg_price"]
            from_hi = (p - sl["high_price"]) / sl["high_price"]
            triggered = (pnl <= FIXED_STOP or from_hi <= -TRAILING_PCT)

            if triggered:
                days_held = d_idx - sl["entry_idx"]
                if protect_days > 0 and days_held < protect_days:
                    # 在保護期內：抑制停損，繼續持有
                    protected_stop_skip += 1
                    surviving.append(sl)
                else:
                    realized += sl["shares"] * p
                    stop_loss_count += 1
            else:
                surviving.append(sl)
        slots = surviving

        # 每週重平衡
        if d_idx % REBALANCE_DAYS == 0 and d_idx >= MOM_WARMUP:
            moms = {}
            for s in syms:
                m = calc_mixed_momentum(aligned[s], d_idx)
                if not np.isnan(m) and m > 0:
                    moms[s] = m
            ranked = sorted(moms.items(), key=lambda x: -x[1])

            # ROTATE：冷卻期 ROTATE_MIN_DAYS（保護期不影響 ROTATE）
            eligible = [(i, sl) for i, sl in enumerate(slots)
                        if d_idx - sl["entry_idx"] >= ROTATE_MIN_DAYS]
            if eligible:
                worst_i, worst_sl = min(eligible,
                    key=lambda x: moms.get(x[1]["sym"], -999.0))
                worst_mom = moms.get(worst_sl["sym"], -999.0)
                held_syms = {sl["sym"] for sl in slots}
                for sym, cand_mom in ranked:
                    if sym in held_syms:
                        continue
                    if cand_mom - worst_mom <= ROTATE_GAP:
                        break
                    p_sell = px.get(worst_sl["sym"], np.nan)
                    p_buy  = px.get(sym, np.nan)
                    if np.isnan(p_sell) or np.isnan(p_buy):
                        break
                    realized += worst_sl["shares"] * p_sell
                    slots.pop(worst_i)
                    shares = SLOT_SIZE / p_buy
                    total_invested += SLOT_SIZE
                    slots.append({"sym": sym, "shares": shares,
                                  "avg_price": p_buy, "high_price": p_buy,
                                  "entry_idx": d_idx})
                    break

            # 補空槽
            held_syms = {sl["sym"] for sl in slots}
            for sym, _ in ranked:
                if len(slots) >= MAX_SLOTS:
                    break
                if sym in held_syms:
                    continue
                p_buy = px.get(sym, np.nan)
                if np.isnan(p_buy) or p_buy <= 0:
                    continue
                shares = SLOT_SIZE / p_buy
                total_invested += SLOT_SIZE
                slots.append({"sym": sym, "shares": shares,
                              "avg_price": p_buy, "high_price": p_buy,
                              "entry_idx": d_idx})
                held_syms.add(sym)
                if len(slots) >= TOP_N:
                    break

        # 當日市值
        mv = sum(
            sl["shares"] * px.get(sl["sym"], sl["avg_price"])
            for sl in slots
        )
        daily_market_val.append(mv)
        daily_pnl.append(realized + mv - total_invested)

    # 結算
    final_mv = sum(
        sl["shares"] * aligned[sl["sym"]][-1]
        for sl in slots if sl["sym"] in aligned
    )
    total_return = (realized + final_mv - total_invested) / max(total_invested, 1)

    pnl_arr = np.array(daily_pnl)
    peak = np.maximum.accumulate(pnl_arr)
    dd = np.where(peak > 0, (pnl_arr - peak) / (total_invested + 1e-9), 0)
    mdd = float(np.min(dd))

    years = n / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    calmar = abs(cagr / mdd) if mdd < 0 else 0

    return {
        "total_return": total_return * 100,
        "cagr": cagr * 100,
        "mdd": mdd * 100,
        "calmar": calmar,
        "stop_loss_count": stop_loss_count,
        "protected_skips": protected_stop_skip,
        "total_invested": total_invested,
    }

# ── 執行比較 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  停損保護期回測  ({common_dates[0].date()} ~ {common_dates[-1].date()})")
print(f"  固定停損 {FIXED_STOP*100:.0f}%  追蹤停損 {TRAILING_PCT*100:.0f}%  ROTATE冷卻 {ROTATE_MIN_DAYS}天")
print(f"{'='*65}")
print(f"  ⚠️  Survivorship bias：所有策略絕對報酬偏高，相對差距有參考價值")
print(f"{'='*65}\n")

results = {}
for pd_days in PROTECT_DAYS_LIST:
    label = f"停損保護 {pd_days:2d}天" if pd_days > 0 else "標準（無保護）"
    print(f"模擬中：{label}...", flush=True)
    r = run_sim(pd_days)
    results[pd_days] = r
    skip_note = f"  （被保護擋下 {r['protected_skips']} 次）" if pd_days > 0 else ""
    print(f"  CAGR {r['cagr']:+.1f}%  MDD {r['mdd']:+.1f}%  Calmar {r['calmar']:.3f}"
          f"  停損出場 {r['stop_loss_count']} 次{skip_note}")

# 彙總表
print(f"\n{'='*65}")
print(f"  {'策略':<16} {'CAGR':>8} {'MDD':>8} {'Calmar':>8} {'停損次數':>8} {'被擋次數':>8}")
print(f"-"*65)
base = results[0]
for pd_days in PROTECT_DAYS_LIST:
    r = results[pd_days]
    label = f"保護{pd_days:2d}天" if pd_days > 0 else "標準（無保護）"
    delta = f"  Δ Calmar {r['calmar']-base['calmar']:+.3f}" if pd_days > 0 else ""
    print(f"  {label:<16} {r['cagr']:>+7.1f}% {r['mdd']:>+7.1f}% {r['calmar']:>8.3f}"
          f" {r['stop_loss_count']:>8} {r['protected_skips']:>8}{delta}")

print(f"\n結論：")
best = max(results.values(), key=lambda x: x["calmar"])
best_days = [k for k, v in results.items() if v == best][0]
if best_days == 0:
    print("  ✅ 標準策略（無保護期）Calmar 最高 → 停損應該從第 1 天就生效")
else:
    print(f"  ⚠️  保護 {best_days} 天 Calmar 最高 → 考慮實作停損保護期")
