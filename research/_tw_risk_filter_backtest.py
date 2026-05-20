"""
_tw_risk_filter_backtest.py

台股風險過濾條件回測：測試哪些條件能有效排除炒作股（連續漲停後崩跌）。

問題：系統推薦聯致 +765% 等高動能股，但這類股票往往是主力炒作，
      一旦進場很可能追到高點後崩跌（190→70 兩天）。

測試過濾條件（price-based，使用現有快取）：
  A（基準）：無過濾（現行系統）
  B：連續漲停 ≥ 2 天（日漲幅 ≥ 9.5%）→ 排除
  C：連續漲停 ≥ 3 天 → 排除
  D：5 日報酬 > 40% → 排除
  E：5 日報酬 > 60% → 排除
  F：混合動能 > 200% → 排除
  G：B + D 組合
  H：C + E 組合

執行：
    conda run -n qt_env python research/_tw_risk_filter_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CACHE_PATH  = "data/_tw_bt_prices.pkl"
TOP_N       = 5
REBAL_FREQ  = 21
FIXED_STOP  = 0.15
TRAIL_STOP  = 0.25
TX_TAX      = 0.003
INIT_CASH   = 100_000.0
LIMIT_UP    = 0.095   # 台股漲停門檻

print("載入台股價格快取...")
prices = pd.read_pickle(CACHE_PATH)
prices.index = pd.to_datetime(prices.index).tz_localize(None)
close = prices.copy()
dates = close.index
syms  = list(close.columns)
print(f"  {len(syms)} 支標的，{dates[0].date()} ~ {dates[-1].date()}")

# 預先計算動能（整個矩陣一次算，避免重複）
ret21  = close.pct_change(21)
ret252 = close.pct_change(252)
mom    = 0.5 * ret21 + 0.5 * ret252

# 預先計算 5 日報酬
ret5 = close.pct_change(5)

# 預先計算連續漲停天數（向量化）
daily_ret = close.pct_change()

def count_consecutive_limit_up(ti, sym, n_days=7):
    """計算 ti 時點之前最多 n_days 天的連續漲停天數"""
    if ti < 1:
        return 0
    start = max(0, ti - n_days)
    rets = daily_ret[sym].iloc[start:ti].values
    count = 0
    for r in reversed(rets):
        if np.isnan(r):
            break
        if r >= LIMIT_UP:
            count += 1
        else:
            break
    return count


def run_backtest(filter_fn=None, filter_name="A（基準）"):
    """
    filter_fn: function(ti, sym, mom_val, ret5_val) -> True = 排除
    """
    portfolio = {}  # sym -> {ep, hi, entry_ti}
    cash = INIT_CASH
    nav_series = []
    trades = []

    rebal_indices = list(range(300, len(dates) - REBAL_FREQ, REBAL_FREQ))

    for ri, ti in enumerate(rebal_indices):
        if ti + REBAL_FREQ >= len(dates):
            break
        next_ti = ti + REBAL_FREQ

        # ── 再平衡日：先日日停損檢查，然後換股 ──
        to_exit = []
        for sym, pos in portfolio.items():
            stopped = False
            for di in range(ti, next_ti):
                p = close[sym].iloc[di] if sym in close.columns else np.nan
                if np.isnan(p):
                    continue
                pos['hi'] = max(pos['hi'], p)
                stop = max(pos['ep'] * (1 - FIXED_STOP), pos['hi'] * (1 - TRAIL_STOP))
                if p <= stop:
                    net = p * (1 - TX_TAX)
                    cash += net
                    trades.append(net / pos['ep'] - 1)
                    to_exit.append(sym)
                    stopped = True
                    break
            if not stopped:
                # 再平衡日賣出
                p = close[sym].iloc[next_ti] if sym in close.columns else np.nan
                if not np.isnan(p):
                    net = p * (1 - TX_TAX)
                    cash += net
                    trades.append(net / pos['ep'] - 1)
                to_exit.append(sym)

        for s in to_exit:
            portfolio.pop(s, None)

        # ── 選新標的 ──
        mom_today = mom.iloc[ti].dropna()
        ret5_today = ret5.iloc[ti]

        ranked = mom_today.sort_values(ascending=False)
        selected = []
        for sym in ranked.index:
            if len(selected) >= TOP_N:
                break
            if sym in portfolio:
                continue
            mom_val  = float(ranked[sym])
            r5_val   = float(ret5_today.get(sym, np.nan))

            # 套用過濾條件
            if filter_fn and filter_fn(ti, sym, mom_val, r5_val):
                continue

            selected.append(sym)

        # 買入
        if selected:
            per_slot = cash / len(selected)
            for sym in selected:
                ep = float(close[sym].iloc[ti]) if sym in close.columns else np.nan
                if np.isnan(ep) or ep <= 0:
                    continue
                shares = per_slot / ep
                cash -= per_slot
                portfolio[sym] = {'ep': ep, 'hi': ep, 'shares': shares, 'entry_ti': ti}

        # NAV
        pos_val = sum(
            float(close[sym].iloc[ti]) * pos['shares']
            for sym, pos in portfolio.items()
            if sym in close.columns and not np.isnan(float(close[sym].iloc[ti]))
        )
        nav_series.append(cash + pos_val)

    if not trades:
        return None

    nav = pd.Series(nav_series)
    cagr = (nav.iloc[-1] / INIT_CASH) ** (252 / (len(nav) * REBAL_FREQ)) - 1
    roll_max = nav.cummax()
    mdd = ((nav - roll_max) / roll_max).min()

    return {
        'name': filter_name,
        'n_trades': len(trades),
        'avg': np.mean(trades) * 100,
        'med': np.median(trades) * 100,
        'win': sum(t > 0 for t in trades) / len(trades) * 100,
        'p25': np.percentile(trades, 25) * 100,
        'p75': np.percentile(trades, 75) * 100,
        'cagr': cagr * 100,
        'mdd': mdd * 100,
    }


# ── 定義各過濾函數 ─────────────────────────────────────────────────────────

def filter_B(ti, sym, mom_val, r5_val):
    """連續漲停 ≥ 2 天"""
    return count_consecutive_limit_up(ti, sym) >= 2

def filter_C(ti, sym, mom_val, r5_val):
    """連續漲停 ≥ 3 天"""
    return count_consecutive_limit_up(ti, sym) >= 3

def filter_D(ti, sym, mom_val, r5_val):
    """5日報酬 > 40%"""
    return not np.isnan(r5_val) and r5_val > 0.40

def filter_E(ti, sym, mom_val, r5_val):
    """5日報酬 > 60%"""
    return not np.isnan(r5_val) and r5_val > 0.60

def filter_F(ti, sym, mom_val, r5_val):
    """混合動能 > 200%"""
    return mom_val > 2.00

def filter_G(ti, sym, mom_val, r5_val):
    """B + D"""
    return filter_B(ti, sym, mom_val, r5_val) or filter_D(ti, sym, mom_val, r5_val)

def filter_H(ti, sym, mom_val, r5_val):
    """C + E"""
    return filter_C(ti, sym, mom_val, r5_val) or filter_E(ti, sym, mom_val, r5_val)


# ── 執行所有回測 ───────────────────────────────────────────────────────────

configs = [
    (None,     "A（基準，無過濾）"),
    (filter_B, "B（連續漲停≥2天）"),
    (filter_C, "C（連續漲停≥3天）"),
    (filter_D, "D（5日>40%）"),
    (filter_E, "E（5日>60%）"),
    (filter_F, "F（動能>200%）"),
    (filter_G, "G（B+D）"),
    (filter_H, "H（C+E）"),
]

print("\n執行回測（8 組合）...")
results = []
for fn, name in configs:
    r = run_backtest(fn, name)
    if r:
        results.append(r)
        print(f"  {name} 完成（{r['n_trades']} 筆）")

print(f"""
{'='*85}
  台股風險過濾條件回測
  策略：Top5 混合動能（21+252日），月度再平衡，固定-15%/追蹤-25%停損，含0.3%證交稅
{'='*85}
  過濾條件             筆數  平均報酬  中位報酬   勝率    P25     P75    CAGR    MDD
  {'-'*75}""")

baseline = results[0]
for r in results:
    mark = " ← 現行" if r['name'].startswith("A") else ""
    diff_med = r['med'] - baseline['med']
    diff_str = f"({diff_med:+.1f}%)" if not r['name'].startswith("A") else ""
    print(f"  {r['name']:<20} {r['n_trades']:>4}  "
          f"{r['avg']:>+7.2f}%  {r['med']:>+7.2f}% {diff_str:<8}"
          f"{r['win']:>5.1f}%  {r['p25']:>+6.2f}%  {r['p75']:>+6.2f}%"
          f"  {r['cagr']:>+6.1f}%  {r['mdd']:>+6.1f}%{mark}")

best = max(results[1:], key=lambda r: r['med'])
print(f"\n  中位報酬最佳過濾：{best['name']}  → 中位 {best['med']:+.2f}%（vs 基準 {baseline['med']:+.2f}%）")
print(f"\n  ⚠️  Survivorship bias，不含台股停損滑價成本\n")
