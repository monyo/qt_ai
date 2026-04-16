"""
_bear_trailing_backtest.py

回測問題：BEAR 模式下收緊追蹤停損，能否更好保住獲利？

比較策略（與現實系統一致：BEAR 不新增、不 ROTATE，但繼續持倉）：
  A. 標準：追蹤停損固定 -25%（BULL/BEAR 相同）
  B. BEAR 收緊 -20%：BEAR 期間追蹤停損改為 -20%
  C. BEAR 收緊 -15%：BEAR 期間追蹤停損改為 -15%
  D. BEAR 收緊 -10%：BEAR 期間追蹤停損改為 -10%

其他條件相同：固定停損 -15%，ROTATE 冷卻期 30 天，BULL 追蹤 -25%

使用：
    conda run -n qt_env python _bear_trailing_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data_loader import get_sp500_tickers, fetch_stock_data
from portfolio_backtest import build_aligned_prices, calc_mixed_momentum

# ── 參數 ──────────────────────────────────────────────────────────────────────
SLOT_SIZE       = 8_667.0
MAX_SLOTS       = 30
FIXED_STOP      = -0.15
BULL_TRAILING   = 0.25      # BULL 模式追蹤停損（不變）
REBALANCE_DAYS  = 5
MOM_WARMUP      = 252
ROTATE_GAP      = 0.10
ROTATE_MIN_DAYS = 30
MA200_WINDOW    = 200
TOP_N           = 5

# BEAR 模式追蹤停損（要比較的變體）
BEAR_TRAILING_LIST = [0.25, 0.20, 0.15, 0.10]

# ── 載入價格 ──────────────────────────────────────────────────────────────────
print("載入價格資料（有快取則略過）...")
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
bear_days = int(np.sum(~spy_bull[MOM_WARMUP:]))
bull_days = int(np.sum(spy_bull[MOM_WARMUP:]))
print(f"  BULL 天數: {bull_days}  BEAR 天數: {bear_days}  "
      f"（BEAR 佔比 {bear_days/(bull_days+bear_days)*100:.1f}%）\n")

# ── 核心模擬 ──────────────────────────────────────────────────────────────────
def run_sim(bear_trailing: float) -> dict:
    """
    BEAR 模式：持倉繼續，但追蹤停損收緊為 bear_trailing
               不新增、不 ROTATE
    BULL 模式：正常動能選股 + ROTATE，追蹤停損 BULL_TRAILING
    """
    slots = []
    realized = 0.0
    total_invested = 0.0
    daily_pnl = []
    stop_count_bull = 0
    stop_count_bear = 0
    bear_trailing_exits = 0  # 因收緊追蹤而額外出場的次數

    for d_idx in range(n):
        px = {s: aligned[s][d_idx] for s in aligned}
        is_bull = spy_bull[d_idx]

        # 更新最高點（無論 BULL/BEAR 都更新）
        for sl in slots:
            p = px.get(sl["sym"], np.nan)
            if not np.isnan(p) and p > 0 and p > sl["high_price"]:
                sl["high_price"] = p

        # 出場：固定停損 + 追蹤停損（依體制選不同追蹤比例）
        trailing = BULL_TRAILING if is_bull else bear_trailing
        surviving = []
        for sl in slots:
            p = px.get(sl["sym"], np.nan)
            if np.isnan(p) or p <= 0:
                surviving.append(sl)
                continue
            pnl     = (p - sl["avg_price"]) / sl["avg_price"]
            from_hi = (p - sl["high_price"]) / sl["high_price"]
            if pnl <= FIXED_STOP or from_hi <= -trailing:
                realized += sl["shares"] * p
                if is_bull:
                    stop_count_bull += 1
                else:
                    stop_count_bear += 1
                    # 如果用標準 -25% 不會觸發，但用 bear_trailing 觸發 → 額外出場
                    if from_hi > -BULL_TRAILING and bear_trailing < BULL_TRAILING:
                        bear_trailing_exits += 1
            else:
                surviving.append(sl)
        slots = surviving

        # BEAR 模式：不新增、不 ROTATE
        if not is_bull:
            mv = sum(sl["shares"] * px.get(sl["sym"], sl["avg_price"]) for sl in slots)
            daily_pnl.append(realized + mv - total_invested)
            continue

        # BULL 模式：每週重平衡
        if d_idx % REBALANCE_DAYS == 0 and d_idx >= MOM_WARMUP:
            moms = {}
            for s in syms:
                m = calc_mixed_momentum(aligned[s], d_idx)
                if not np.isnan(m) and m > 0:
                    moms[s] = m
            ranked = sorted(moms.items(), key=lambda x: -x[1])

            # ROTATE
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
                if len(slots) >= TOP_N:
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

        mv = sum(sl["shares"] * px.get(sl["sym"], sl["avg_price"]) for sl in slots)
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
        "cagr": cagr * 100,
        "mdd": mdd * 100,
        "calmar": calmar,
        "stop_bull": stop_count_bull,
        "stop_bear": stop_count_bear,
        "bear_extra_exits": bear_trailing_exits,
    }

# ── 執行比較 ──────────────────────────────────────────────────────────────────
print(f"{'='*68}")
print(f"  BEAR 模式追蹤停損收緊回測  ({common_dates[0].date()} ~ {common_dates[-1].date()})")
print(f"  BULL 追蹤停損固定 {BULL_TRAILING*100:.0f}%，固定停損 {FIXED_STOP*100:.0f}%")
print(f"  BEAR 模式：持倉繼續，不新增，不 ROTATE")
print(f"{'='*68}")
print(f"  ⚠️  Survivorship bias：相對差距有參考價值\n")

results = {}
for bt in BEAR_TRAILING_LIST:
    label = f"BEAR 追蹤 -{bt*100:.0f}%" if bt != BULL_TRAILING else "標準（BEAR/BULL 同 -25%）"
    print(f"模擬中：{label}...", flush=True)
    r = run_sim(bt)
    results[bt] = r
    print(f"  CAGR {r['cagr']:+.1f}%  MDD {r['mdd']:+.1f}%  Calmar {r['calmar']:.3f}"
          f"  BEAR停損 {r['stop_bear']}次  其中因收緊多出場 {r['bear_extra_exits']}次")

# 彙總
base = results[BULL_TRAILING]
print(f"\n{'='*68}")
print(f"  {'策略':<22} {'CAGR':>7} {'MDD':>7} {'Calmar':>8} {'BEAR停損':>8} {'收緊多出場':>10}")
print(f"-"*68)
for bt in BEAR_TRAILING_LIST:
    r = results[bt]
    label = f"標準（BEAR -25%）" if bt == BULL_TRAILING else f"BEAR 追蹤 -{bt*100:.0f}%"
    delta = f"  Δ{r['calmar']-base['calmar']:+.3f}" if bt != BULL_TRAILING else ""
    print(f"  {label:<22} {r['cagr']:>+6.1f}% {r['mdd']:>+6.1f}% {r['calmar']:>8.3f}"
          f" {r['stop_bear']:>8} {r['bear_extra_exits']:>10}{delta}")

print(f"\n結論：")
best_bt = max(results, key=lambda k: results[k]["calmar"])
if best_bt == BULL_TRAILING:
    print("  ✅ 標準追蹤停損（不分 BULL/BEAR）Calmar 最高 → 不需要收緊")
else:
    r_best = results[best_bt]
    print(f"  ✅ BEAR 追蹤 -{best_bt*100:.0f}% Calmar {r_best['calmar']:.3f} 最高"
          f" → 考慮在 BEAR 模式收緊追蹤停損至 -{best_bt*100:.0f}%")
