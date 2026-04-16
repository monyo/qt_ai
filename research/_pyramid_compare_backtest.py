"""
_pyramid_compare_backtest.py

公平比較：金字塔加碼 vs 標準動能分散（無資本上限版）

設計原則：
  - 兩邊相同的槽位數（MAX_SLOTS=150）、相同每槽資本（SLOT_SIZE）
  - 資金無限：每次開槽直接注入 SLOT_SIZE，不受現金餘額限制
  - 最後結算：比較總投入資本、最終價值、淨利潤、報酬率
  - 出場/ROTATE/停損機制與主系統一致

差別只在「空缺槽要填哪個標的」：
  A（標準）: 永遠買最強的「尚未持有」標的
  B/C/D/E（金字塔）: 優先對動能最強且尚未達上限的標的加碼；
              若無此機會，行為與 A 相同

最後比：CAGR / MDD / Calmar（均基於總投入資本）
"""

import os
import sys
import warnings

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data_loader import get_sp500_tickers, fetch_stock_data
from portfolio_backtest import build_aligned_prices, calc_mixed_momentum

# ── 參數 ──────────────────────────────────────────────────────────────────────
SLOT_SIZE        = 8_667.0        # 每個投注單位的資本（美元）
MAX_SLOTS        = 30             # 與現實系統一致（30 槽）
TRAILING_PCT     = 0.25           # 追蹤停損（從各槽位自身最高點）
FIXED_STOP       = -0.15          # 固定停損
CASH_SAFETY      = 0.85           # ROTATE 賣出後保留 15% 避免價差
REBALANCE_DAYS   = 5              # 每週重平衡
MOM_WARMUP       = 252            # 動能暖機期（交易日）
ROTATE_GAP       = 0.10           # ROTATE 動能差門檻（>10%）
ROTATE_MIN_DAYS  = 30             # 槽位最短持有天數，才可被 ROTATE
MA200_WINDOW     = 200            # SPY 體制判斷均線

# 差異停損：依批次設定不同的固定停損、追蹤停損與保護期
# 第 3 批以上沿用第 3 批設定
TRANCHE_PARAMS = {
    1: {"fixed": -0.15, "trailing": 0.25, "protect": 30},
    2: {"fixed": -0.10, "trailing": 0.15, "protect": 15},
    3: {"fixed": -0.07, "trailing": 0.10, "protect":  7},
}


def _tranche_p(tranche: int) -> dict:
    return TRANCHE_PARAMS.get(tranche, TRANCHE_PARAMS[3])


# ── 工具函數 ──────────────────────────────────────────────────────────────────

def _spy_bull_regime(aligned: dict, n: int) -> np.ndarray:
    """True = BULL（SPY > MA200），False = BEAR"""
    spy = aligned.get("SPY", np.zeros(n))
    bull = np.ones(n, dtype=bool)
    for i in range(n):
        if i >= MA200_WINDOW - 1:
            ma = np.mean(spy[i - MA200_WINDOW + 1: i + 1])
            bull[i] = spy[i] >= ma
    return bull


# ── 核心模擬 ──────────────────────────────────────────────────────────────────

def run_sim(aligned: dict, common_dates, use_pyramid: bool,
            max_pyramid: int = 1, graduated_stops: bool = False) -> dict:
    """
    無資本上限模擬。

    slots: list of {sym, shares, avg_price, high_price, entry_idx}

    資本追蹤（取代 cash 變數）：
      total_invested : 累計所有買入金額（開槽時注入新資本）
      realized       : 累計所有賣出收益（賣出時流回）

    日常 P&L = realized + 浮動市值 - total_invested（起點 0，漲跌反映損益）
    avg_concurrent  = 平均每日實際在場市值（作為 CAGR / MDD 基準）
    MDD 基於 daily_pnl 序列，除以 avg_concurrent 換算百分比
    CAGR = (1 + net_profit / avg_concurrent)^(252/n) - 1

    ROTATE 為自給自足：賣出收益直接用於買入替換，不注入新資本。
    ADD 開槽才注入 SLOT_SIZE 新資本。
    """
    syms   = [s for s in aligned if s != "SPY"]
    n      = len(common_dates)

    total_invested  = 0.0    # 累計資本注入
    realized        = 0.0    # 累計賣出收益
    slots           = []
    daily_pnl       = []     # 每日淨損益：realized + 市值 - total_invested
    daily_market_val = []    # 每日實際在場市值（用於計算 avg_concurrent）

    spy_bull = _spy_bull_regime(aligned, n)

    for d_idx in range(n):
        px = {s: aligned[s][d_idx] for s in aligned}

        # ── SPY BEAR：全部出場，等待 BULL 恢復 ───────────────
        if not spy_bull[d_idx]:
            for sl in slots:
                p = px.get(sl["sym"], np.nan)
                if not np.isnan(p) and p > 0:
                    realized += sl["shares"] * p
            slots = []
            daily_pnl.append(realized - total_invested)
            daily_market_val.append(0.0)
            continue

        # ── 更新各槽位最高點 ──────────────────────────────────
        for sl in slots:
            p = px.get(sl["sym"], np.nan)
            if not np.isnan(p) and p > 0 and p > sl["high_price"]:
                sl["high_price"] = p

        # ── 出場：固定停損 + 追蹤停損（每槽獨立，差異停損依批次）─
        surviving = []
        for sl in slots:
            p = px.get(sl["sym"], np.nan)
            if np.isnan(p) or p <= 0:
                surviving.append(sl)
                continue
            if graduated_stops:
                tp = _tranche_p(sl.get("tranche", 1))
                f_stop, t_pct = tp["fixed"], tp["trailing"]
            else:
                f_stop, t_pct = FIXED_STOP, TRAILING_PCT
            pnl     = (p - sl["avg_price"]) / sl["avg_price"]
            from_hi = (p - sl["high_price"]) / sl["high_price"]
            if pnl <= f_stop or from_hi <= -t_pct:
                realized += sl["shares"] * p
            else:
                surviving.append(sl)
        slots = surviving

        # ── 每週重平衡 ────────────────────────────────────────
        if d_idx % REBALANCE_DAYS == 0 and d_idx >= MOM_WARMUP:

            # 動能排名（只取正動能）
            moms = {}
            for s in syms:
                m = calc_mixed_momentum(aligned[s], d_idx)
                if not np.isnan(m) and m > 0:
                    moms[s] = m
            ranked = sorted(moms.items(), key=lambda x: -x[1])

            # ── ROTATE：汰弱留強（每週最多 1 次）─────────────
            eligible = []
            for i, sl in enumerate(slots):
                prot = _tranche_p(sl.get("tranche", 1))["protect"] \
                       if graduated_stops else ROTATE_MIN_DAYS
                if d_idx - sl.get("entry_idx", 0) >= prot:
                    eligible.append((i, sl))
            if eligible:
                worst_i, worst_sl = min(
                    eligible,
                    key=lambda x: moms.get(x[1]["sym"], -999.0)
                )
                worst_mom = moms.get(worst_sl["sym"], -999.0)
                held_syms = {sl["sym"] for sl in slots}

                for sym, cand_mom in ranked:
                    if sym in held_syms:
                        continue
                    if cand_mom - worst_mom <= ROTATE_GAP:
                        break
                    p_sell = px.get(worst_sl["sym"], np.nan)
                    p_buy  = px.get(sym, np.nan)
                    if np.isnan(p_sell) or np.isnan(p_buy) or p_buy <= 0:
                        continue
                    # ROTATE 自給自足：賣出收益用於買入，不注入新資本
                    sell_proceeds = worst_sl["shares"] * p_sell
                    realized += sell_proceeds
                    slots.pop(worst_i)
                    new_sh = int(sell_proceeds * CASH_SAFETY // p_buy)
                    if new_sh > 0:
                        total_invested += new_sh * p_buy
                        slots.append({
                            "sym": sym, "shares": new_sh,
                            "avg_price": p_buy, "high_price": p_buy,
                            "entry_idx": d_idx, "tranche": 1,
                        })
                    break

            # ── ADD：填補空缺槽位（注入新資本，無限制）──────────
            cnt = {}
            for sl in slots:
                cnt[sl["sym"]] = cnt.get(sl["sym"], 0) + 1

            open_n = MAX_SLOTS - len(slots)
            for _ in range(open_n):
                chosen = None

                if use_pyramid:
                    # 金字塔：優先對動能最強且未達上限的標的加碼
                    for sym, _ in ranked:
                        c = cnt.get(sym, 0)
                        if 0 < c < max_pyramid:
                            chosen = sym
                            break
                    if chosen is None:
                        for sym, _ in ranked:
                            if cnt.get(sym, 0) == 0:
                                chosen = sym
                                break
                else:
                    # 標準：只買尚未持有的標的
                    for sym, _ in ranked:
                        if cnt.get(sym, 0) == 0:
                            chosen = sym
                            break

                if chosen is None:
                    break

                p_buy = px.get(chosen, np.nan)
                if np.isnan(p_buy) or p_buy <= 0:
                    continue

                shares = int(SLOT_SIZE // p_buy)
                if shares <= 0:
                    continue

                # 注入新資本開槽（記錄批次編號供差異停損使用）
                tranche = cnt.get(chosen, 0) + 1
                total_invested += shares * p_buy
                slots.append({
                    "sym": chosen, "shares": shares,
                    "avg_price": p_buy, "high_price": p_buy,
                    "entry_idx": d_idx, "tranche": tranche,
                })
                cnt[chosen] = tranche

        # ── 記錄當日 P&L 與市值 ───────────────────────────────
        market_val = sum(
            sl["shares"] * max(px.get(sl["sym"], sl["avg_price"]), 0.01)
            for sl in slots
        )
        daily_pnl.append(realized + market_val - total_invested)
        daily_market_val.append(market_val)

    # ── 計算績效指標 ──────────────────────────────────────────────────────────
    final_pnl   = daily_pnl[-1]
    final_value = total_invested + final_pnl   # = realized + market_val at last day

    # 平均在場資本 = 每日實際持倉市值的時間加權平均
    # 自然涵蓋暖機期（市值=0）和 BEAR 期（市值=0），不偏不倚
    avg_concurrent = float(np.mean(daily_market_val)) if daily_market_val else 1.0
    if avg_concurrent <= 0:
        avg_concurrent = SLOT_SIZE  # 防止除以零

    # CAGR：以平均在場資本為基準年化（使結果與固定資本回測可比）
    if final_pnl > -avg_concurrent:
        hpr  = final_pnl / avg_concurrent
        cagr = float((1 + hpr) ** (252 / n) - 1)
    else:
        hpr = cagr = -1.0

    # MDD：P&L 序列的峰谷回撤，除以平均在場資本
    pnl_arr  = np.array(daily_pnl)
    peak_pnl = np.maximum.accumulate(pnl_arr)
    mdd_abs  = float(np.min(pnl_arr - peak_pnl))   # 負值，$
    mdd      = mdd_abs / avg_concurrent

    calmar = cagr / abs(mdd) if mdd != 0 else 0.0

    return {
        "AvgConcurrent$": round(avg_concurrent),
        "TotalDeployed$": round(total_invested),
        "FinalVal$"     : round(final_value),
        "NetProfit$"    : round(final_pnl),
        "Return%"       : round(final_pnl / total_invested * 100, 1) if total_invested > 0 else 0.0,
        "CAGR%"         : round(cagr * 100, 1),
        "MDD%"          : round(mdd * 100, 1),
        "Calmar"        : round(calmar, 3),
    }


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    print("=== 金字塔加碼深度掃描 | 無資本上限公平比較 ===\n")
    print(f"  每槽資本 ${SLOT_SIZE:,.0f}  |  最大槽位 {MAX_SLOTS}")
    print(f"  固定停損 -{abs(FIXED_STOP)*100:.0f}%  |  追蹤停損 -{TRAILING_PCT*100:.0f}%")
    print(f"  ROTATE 門檻 >{ROTATE_GAP*100:.0f}%  |  保護期 {ROTATE_MIN_DAYS}天")
    print(f"  SPY MA{MA200_WINDOW} 體制過濾\n")

    print("  取得 S&P 500 標的...")
    tickers  = [t for t in get_sp500_tickers() if t not in ("BRK.B", "BF.B")]
    universe = tickers[:503] + ["SPY"]
    print(f"  下載 {len(universe)} 支股票歷史價格（有快取則略過）...")

    raw = {}
    for t in universe:
        df = fetch_stock_data(t, period="10y")
        if df is not None and len(df) > 200:
            raw[t] = df["Close"]

    aligned, common_dates = build_aligned_prices(raw)
    n = len(common_dates)
    print(f"  期間：{str(common_dates[0])[:10]} → {str(common_dates[-1])[:10]}（{n} 交易日）")
    print(f"  有效標的：{len(aligned) - 1} 支\n")

    # (label, use_pyramid, max_pyramid, graduated_stops)
    configs = [
        ("A 標準分散",          False, 1,         False),
        ("B 金字塔×3 統一停損",  True,  3,         False),
        ("C 金字塔×3 差異停損",  True,  3,         True),
        ("D 金字塔×5 統一停損",  True,  5,         False),
        ("E 金字塔×5 差異停損",  True,  5,         True),
        ("F 金字塔無限 統一停損", True,  MAX_SLOTS, False),
        ("G 金字塔無限 差異停損", True,  MAX_SLOTS, True),
    ]

    results = {}
    for label, use_pyr, mp, grad in configs:
        tag = "差異" if grad else "統一"
        desc = f"×{mp}" if mp < MAX_SLOTS else "無限"
        print(f"  模擬 {label}...", end=" ", flush=True)
        res = run_sim(aligned, common_dates, use_pyramid=use_pyr,
                      max_pyramid=mp, graduated_stops=grad)
        results[label] = res
        print(
            f"CAGR {res['CAGR%']:+.1f}%  MDD {res['MDD%']:.1f}%  Calmar {res['Calmar']:.3f}"
            f"  淨利 ${res['NetProfit$']/1e3:.0f}K"
        )

    # ── 結果表 ────────────────────────────────────────────────────────────────
    col_labels = [lbl for lbl, *_ in configs]
    col_w = 22

    metrics_map = [
        ("AvgConcurrent$", "平均在場資本 $"),
        ("TotalDeployed$", "累計投入 $"),
        ("FinalVal$",      "最終價值 $"),
        ("NetProfit$",     "淨利潤 $"),
        ("Return%",        "累計投入報酬 %"),
        ("CAGR%",          "年化報酬 %"),
        ("MDD%",           "最大回撤 %"),
        ("Calmar",         "Calmar 比率"),
    ]

    total_width = 22 + col_w * len(configs)
    print()
    print("=" * total_width)
    header = f"  {'指標':<18}" + "".join(f"  {c:>{col_w-2}}" for c in col_labels)
    print(header)
    print("  " + "-" * (total_width - 2))

    base_label = col_labels[0]
    for k, metric_name in metrics_map:
        row = f"  {metric_name:<18}"
        base_val = results[base_label][k]
        for lbl in col_labels:
            v = results[lbl][k]
            if lbl == base_label:
                row += f"  {str(v):>{col_w-2}}"
            else:
                diff = v - base_val
                sign = "+" if diff >= 0 else ""
                cell = f"{v}({sign}{diff:.1f})"
                row += f"  {cell:>{col_w-2}}"
        print(row)

    print("=" * total_width)

    # ── 結論 ──────────────────────────────────────────────────────────────────
    print()
    best_calmar_lbl = max(col_labels, key=lambda l: results[l]["Calmar"])
    best_cagr_lbl   = max(col_labels, key=lambda l: results[l]["CAGR%"])
    print(f"  最高 Calmar：{best_calmar_lbl}  ({results[best_calmar_lbl]['Calmar']:.3f})")
    print(f"  最高 CAGR  ：{best_cagr_lbl}  ({results[best_cagr_lbl]['CAGR%']:+.1f}%)")

    if best_calmar_lbl == best_cagr_lbl and best_calmar_lbl != base_label:
        print(f"\n  ✅ 金字塔加碼雙優：選 {best_calmar_lbl}")
    elif best_cagr_lbl != base_label:
        diff_mdd = results[best_cagr_lbl]["MDD%"] - results[base_label]["MDD%"]
        print(f"\n  ⚠️  CAGR 最高為 {best_cagr_lbl}，但回撤增加 {diff_mdd:+.1f}%")
    else:
        print(f"\n  ❌ 金字塔加碼各深度均未超越標準分散策略")


if __name__ == "__main__":
    main()
