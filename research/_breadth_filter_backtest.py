"""
_breadth_filter_backtest.py

回測：市場廣度警告（ADD ≤ 3）是否縮小 alpha？

比較兩種策略：
  A（基準）：無廣度限制，每次再平衡最多買滿 MAX_SLOTS 槽
  B（廣度限制）：廣度 < BREADTH_THRESH 時，該次 ADD 上限改為 BREADTH_LIMIT

資料：data/_protection_bt_prices.pkl（501 支 S&P500，2019-2026）
廣度定義：再平衡當日，501 支中有多少 % 的股票站在 50 日均線之上

執行：
    conda run -n qt_env python research/_breadth_filter_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CACHE_PATH     = "data/_protection_bt_prices.pkl"
MAX_SLOTS      = 5         # 最大持倉槽數
REBAL_FREQ     = 21        # 再平衡頻率（交易日）
FIXED_STOP     = 0.15      # 固定停損
TRAIL_STOP     = 0.25      # 追蹤停損
INIT_CASH      = 130_000.0
BREADTH_THRESH = 0.60      # 廣度門檻
BREADTH_LIMIT  = 3         # 廣度不足時 ADD 上限

print("載入 S&P500 價格快取...")
prices = pd.read_pickle(CACHE_PATH)
prices.index = pd.to_datetime(prices.index).tz_localize(None)
close = prices.copy()
dates = close.index
syms  = list(close.columns)
print(f"  {len(syms)} 支標的，{dates[0].date()} ~ {dates[-1].date()}")

# ── 預先計算動能（50% 短期21天 + 50% 長期252天）──────────────────────
ret21  = close.pct_change(21)
ret252 = close.pct_change(252)
mom    = 0.5 * ret21 + 0.5 * ret252

# ── 預先計算 MA50（廣度用）──────────────────────────────────────────
ma50 = close.rolling(50).mean()

def calc_breadth(ti):
    """計算 ti 時點廣度：股票站在 MA50 之上的比例"""
    c = close.iloc[ti]
    m = ma50.iloc[ti]
    valid = (~c.isna()) & (~m.isna())
    if valid.sum() == 0:
        return 1.0
    return (c[valid] > m[valid]).sum() / valid.sum()


def run_backtest(use_breadth_filter=False, label="A"):
    portfolio = {}   # sym -> {ep, hi, shares}
    cash = INIT_CASH
    nav_series = []
    trades = []
    breadth_log = []  # (date, breadth, add_limit)

    rebal_indices = list(range(300, len(dates) - REBAL_FREQ, REBAL_FREQ))

    for ti in rebal_indices:
        if ti + REBAL_FREQ >= len(dates):
            break
        next_ti = ti + REBAL_FREQ

        # ── 停損檢查 + 再平衡賣出 ──────────────────────────────────────
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
                    cash += p * pos['shares']
                    trades.append(p / pos['ep'] - 1)
                    to_exit.append(sym)
                    stopped = True
                    break
            if not stopped:
                p = close[sym].iloc[next_ti] if sym in close.columns else np.nan
                if not np.isnan(p):
                    cash += p * pos['shares']
                    trades.append(p / pos['ep'] - 1)
                to_exit.append(sym)
        for s in to_exit:
            portfolio.pop(s, None)

        # ── 廣度計算 → 決定 ADD 上限 ────────────────────────────────────
        breadth = calc_breadth(ti)
        if use_breadth_filter and breadth < BREADTH_THRESH:
            add_limit = BREADTH_LIMIT
        else:
            add_limit = MAX_SLOTS
        breadth_log.append((dates[ti].date(), round(breadth, 3), add_limit))

        # ── 選新標的 ─────────────────────────────────────────────────
        mom_today = mom.iloc[ti].dropna()
        ranked = mom_today.sort_values(ascending=False)
        empty_slots = MAX_SLOTS - len(portfolio)
        can_add = min(empty_slots, add_limit - 0)  # 這次能 ADD 幾個
        # 修正：add_limit 是「本次再平衡最多 ADD 幾個新倉」
        can_add = min(empty_slots, add_limit)

        selected = []
        for sym in ranked.index:
            if len(selected) >= can_add:
                break
            if sym in portfolio:
                continue
            selected.append(sym)

        if selected:
            per_slot = cash / (len(selected) + 1e-9)
            per_slot = min(per_slot, cash / max(1, len(selected)))
            for sym in selected:
                ep = float(close[sym].iloc[ti]) if sym in close.columns else np.nan
                if np.isnan(ep) or ep <= 0:
                    continue
                shares = per_slot / ep
                cash -= per_slot
                portfolio[sym] = {'ep': ep, 'hi': ep, 'shares': shares}

        # ── NAV ──────────────────────────────────────────────────────
        pos_val = sum(
            float(close[sym].iloc[ti]) * pos['shares']
            for sym, pos in portfolio.items()
            if sym in close.columns and not np.isnan(float(close[sym].iloc[ti]))
        )
        nav_series.append(cash + pos_val)

    nav = pd.Series(nav_series)
    years = len(nav) * REBAL_FREQ / 252
    cagr  = (nav.iloc[-1] / INIT_CASH) ** (1 / years) - 1
    roll_max = nav.cummax()
    mdd   = ((nav - roll_max) / roll_max).min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # 廣度限制啟動次數
    triggered = sum(1 for _, b, lim in breadth_log if lim < MAX_SLOTS)

    return {
        'label': label,
        'final': nav.iloc[-1],
        'cagr': cagr * 100,
        'mdd': mdd * 100,
        'calmar': calmar,
        'n_trades': len(trades),
        'win': sum(t > 0 for t in trades) / len(trades) * 100 if trades else 0,
        'med': np.median(trades) * 100 if trades else 0,
        'breadth_triggered': triggered,
        'total_rebal': len(breadth_log),
        'breadth_log': breadth_log,
    }


print("\n跑回測（2 組合）...")
r_a = run_backtest(use_breadth_filter=False, label="A 無廣度限制")
print(f"  A 完成（{r_a['n_trades']} 筆交易）")
r_b = run_backtest(use_breadth_filter=True,  label="B 有廣度限制")
print(f"  B 完成（{r_b['n_trades']} 筆交易）")

print(f"""
{'='*65}
  市場廣度警告回測（ADD ≤ {BREADTH_LIMIT} 當廣度 < {int(BREADTH_THRESH*100)}%）
  資料：501支 S&P500，{dates[0].date()} ~ {dates[-1].date()}
  策略：Top{MAX_SLOTS} 混合動能，月度再平衡，固定-15%/追蹤-25%停損
{'='*65}
  策略              最終值       CAGR     MDD    Calmar  中位報酬   勝率
  {'-'*55}""")

for r in [r_a, r_b]:
    mark = " ← 現行" if "有廣度" in r['label'] else ""
    print(f"  {r['label']:<18}  ${r['final']:>10,.0f}  "
          f"{r['cagr']:>+6.1f}%  {r['mdd']:>+6.1f}%  "
          f"{r['calmar']:>6.3f}  {r['med']:>+6.2f}%  {r['win']:>5.1f}%{mark}")

diff_cagr   = r_b['cagr'] - r_a['cagr']
diff_calmar = r_b['calmar'] - r_a['calmar']
triggered   = r_b['breadth_triggered']
total       = r_b['total_rebal']

print(f"""
  廣度限制啟動次數：{triggered}/{total} 次再平衡（{triggered/total*100:.0f}%）
  CAGR 差距：{diff_cagr:+.2f}%（B vs A）
  Calmar 差距：{diff_calmar:+.3f}（B vs A）
""")

# ── 廣度低時的 ADD 被限制，損失/獲得了多少？ ──────────────────────────
print("  廣度 < 60% 的再平衡點（前10筆）：")
print(f"  {'日期':<12} {'廣度':>6}  {'ADD上限':>6}")
low_breadth = [(d, b, l) for d, b, l in r_b['breadth_log'] if l < MAX_SLOTS]
for d, b, l in low_breadth[:10]:
    print(f"  {str(d):<12} {b:>5.1%}  {l:>6}")
if len(low_breadth) > 10:
    print(f"  ... 共 {len(low_breadth)} 次")

print(f"""
{'='*65}
""")
