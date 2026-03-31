"""
_scaleout_backtest.py

回測問題：分批停利（Scale-out）能否改善「閃崩時把獲利全還回去」的問題？

比較策略：
  A. 標準：追蹤停損 -25%，固定停損 -15%（現有系統）
  B. 獲利 +20% 賣一半：達 +20% 時賣出 50% 部位，剩餘繼續追蹤停損 -25%
  C. 獲利 +30% 賣一半：達 +30% 時賣出 50% 部位，剩餘繼續追蹤停損 -25%
  D. 動態收緊追蹤：達 +25% 時追蹤停損從 -25% 收緊至 -15%（不賣，只調緊）

其他條件完全相同：BULL/BEAR 體制、ROTATE 30天冷卻、動能 top-5

使用：
    conda run -n qt_env python _scaleout_backtest.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np

from src.data_loader import get_sp500_tickers, fetch_stock_data
from portfolio_backtest import build_aligned_prices, calc_mixed_momentum

# ── 參數 ──────────────────────────────────────────────────────────────────────
SLOT_SIZE        = 8_667.0
FIXED_STOP       = -0.15
TRAILING_PCT     = 0.25
REBALANCE_DAYS   = 5
MOM_WARMUP       = 252
ROTATE_GAP       = 0.10
ROTATE_MIN_DAYS  = 30
MA200_WINDOW     = 200
TOP_N            = 5

# 比較的策略變體
VARIANTS = {
    "A": {"mode": "standard",          "scaleout_pct": None, "tighten_pct": None},
    "B": {"mode": "scaleout",          "scaleout_pct": 0.20,  "tighten_pct": None},
    "C": {"mode": "scaleout",          "scaleout_pct": 0.30,  "tighten_pct": None},
    "D": {"mode": "tighten_trailing",  "scaleout_pct": None,  "tighten_pct": 0.25},
}

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
print(f"  BULL 天數: {bull_days}  BEAR 天數: {bear_days} "
      f"（BEAR 佔比 {bear_days/(bull_days+bear_days)*100:.1f}%）\n")

# ── 核心模擬 ──────────────────────────────────────────────────────────────────
def run_sim(mode, scaleout_pct, tighten_pct):
    """
    mode:
      "standard"          — 純追蹤停損
      "scaleout"          — 達到 scaleout_pct 獲利時賣出 50% 部位
      "tighten_trailing"  — 達到 tighten_pct 獲利時把追蹤停損從 -25% 收緊至 -15%

    每個 slot 結構：
      sym, shares, avg_price, high_price, entry_idx,
      scaled_out (bool)    — 是否已做過停利
      trailing_override    — 若已動態收緊，記錄新的追蹤停損比例
    """
    slots = []
    realized = 0.0
    total_invested = 0.0
    daily_pnl = []
    scaleout_count = 0   # 停利觸發次數

    for d_idx in range(n):
        px = {s: aligned[s][d_idx] for s in aligned}
        is_bull = spy_bull[d_idx]

        # 更新最高點
        for sl in slots:
            p = px.get(sl["sym"], np.nan)
            if not np.isnan(p) and p > 0 and p > sl["high_price"]:
                sl["high_price"] = p

        # ── 出場 / 停利邏輯 ──
        surviving = []
        for sl in slots:
            p = px.get(sl["sym"], np.nan)
            if np.isnan(p) or p <= 0:
                surviving.append(sl)
                continue

            pnl      = (p - sl["avg_price"]) / sl["avg_price"]
            from_hi  = (p - sl["high_price"]) / sl["high_price"]

            # 有效追蹤停損比例（可能被動態收緊）
            eff_trailing = sl.get("trailing_override", TRAILING_PCT)

            # 1. 分批停利（scale-out）：尚未停利 + 達到獲利門檻 → 賣出一半
            if (mode == "scaleout" and scaleout_pct is not None
                    and not sl.get("scaled_out", False)
                    and pnl >= scaleout_pct):
                half = max(1, sl["shares"] // 2)
                realized += half * p
                sl["shares"] -= half
                sl["scaled_out"] = True
                scaleout_count += 1
                if sl["shares"] > 0:
                    surviving.append(sl)
                continue

            # 2. 動態收緊追蹤停損：達到門檻後收緊（不賣）
            if (mode == "tighten_trailing" and tighten_pct is not None
                    and not sl.get("scaled_out", False)
                    and pnl >= tighten_pct):
                sl["trailing_override"] = 0.15   # 收緊至 -15%
                sl["scaled_out"] = True           # 標記已觸發，不重複調整

            # 3. 一般停損出場（固定停損 or 追蹤停損）
            if pnl <= FIXED_STOP or from_hi <= -eff_trailing:
                realized += sl["shares"] * p
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
                                  "entry_idx": d_idx,
                                  "scaled_out": False, "trailing_override": TRAILING_PCT})
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
                              "entry_idx": d_idx,
                              "scaled_out": False, "trailing_override": TRAILING_PCT})
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
        "scaleout_count": scaleout_count,
    }

# ── 執行比較 ──────────────────────────────────────────────────────────────────
labels = {
    "A": "標準（純追蹤 -25%）",
    "B": "Scale-out +20% 賣一半",
    "C": "Scale-out +30% 賣一半",
    "D": "動態收緊：+25%→追蹤改 -15%",
}

print(f"{'='*65}")
print(f"  分批停利回測  ({common_dates[0].date()} ~ {common_dates[-1].date()})")
print(f"  固定停損 {FIXED_STOP*100:.0f}%  標準追蹤 {TRAILING_PCT*100:.0f}%  ROTATE冷卻 {ROTATE_MIN_DAYS}天")
print(f"{'='*65}")
print(f"  ⚠️  Survivorship bias：相對差距有參考價值\n")

results = {}
for key, cfg in VARIANTS.items():
    label = labels[key]
    print(f"模擬中：{label}...", flush=True)
    r = run_sim(cfg["mode"], cfg["scaleout_pct"], cfg["tighten_pct"])
    results[key] = r
    so_note = f"  停利觸發 {r['scaleout_count']} 次" if r["scaleout_count"] > 0 else ""
    print(f"  CAGR {r['cagr']:+.1f}%  MDD {r['mdd']:+.1f}%  Calmar {r['calmar']:.3f}{so_note}")

base = results["A"]
print(f"\n{'='*65}")
print(f"  {'策略':<28} {'CAGR':>7} {'MDD':>7} {'Calmar':>8} {'停利次數':>8}")
print(f"-"*65)
for key in VARIANTS:
    r = results[key]
    delta = f"  Δ{r['calmar']-base['calmar']:+.3f}" if key != "A" else ""
    print(f"  {labels[key]:<28} {r['cagr']:>+6.1f}% {r['mdd']:>+6.1f}% "
          f"{r['calmar']:>8.3f} {r['scaleout_count']:>8}{delta}")

print(f"\n結論：")
best = max(results, key=lambda k: results[k]["calmar"])
if best == "A":
    print("  ✅ 標準策略 Calmar 最高 → 純追蹤停損已是最佳，不需要分批停利")
else:
    r_best = results[best]
    print(f"  ✅ {labels[best]}")
    print(f"     Calmar {r_best['calmar']:.3f}（vs 標準 {base['calmar']:.3f}，"
          f"Δ{r_best['calmar']-base['calmar']:+.3f}）")
    print(f"     停利觸發 {r_best['scaleout_count']} 次")
