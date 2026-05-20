"""
_tighten_slots_backtest.py

同時測試兩個未驗證參數：
  【Part 1】動態收緊門檻（獲利達多少%才收緊追蹤停損）
    測試：+15 / +20 / +25 / +30 / +40 %
    現行：+25%

  【Part 2】最大持倉槽數
    測試：15 / 20 / 25 / 30 槽
    現行：30 槽

執行：
    conda run -n qt_env python research/_tighten_slots_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

PRICES_PATH = "data/_protection_bt_prices.pkl"

print("載入資料...")
prices = pd.DataFrame(pd.read_pickle(PRICES_PATH))
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None: prices.index = prices.index.tz_localize(None)
close = prices.copy()
dates = close.index

rebal_dates = []
prev = None
for d in dates[252:]:
    m = (d.year, d.month)
    if m != prev: rebal_dates.append(d); prev = m

MAX_HOLD = 63
STOP_FIX = -0.15
STOP_TRL = -0.25  # 收緊前
STOP_TRL_TIGHT = -0.15  # 收緊後（standard tranche 邏輯）

mom = 0.5 * close.pct_change(21) + 0.5 * close.pct_change(252)

def run_tighten(tighten_thresh_pct, top_n=5):
    """tighten_thresh_pct: 獲利百分比門檻，如 25 表示 +25%"""
    tighten_thresh = tighten_thresh_pct / 100.0
    portfolio = {}  # sym -> {ei, ep, hi, tightened}
    trades = []

    for ri, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[ri+1]
        ti = dates.get_loc(rd)
        ranked = mom.iloc[ti].dropna().sort_values(ascending=False)
        held = set(portfolio)

        for sym in ranked.index:
            if len(portfolio) >= top_n: break
            if sym in held: continue
            ep = close.iloc[ti].get(sym, np.nan)
            if np.isnan(ep) or ep <= 0: continue
            portfolio[sym] = {'ei': ti, 'ep': ep, 'hi': ep, 'tightened': False}
            held.add(sym)

        next_ti = dates.get_loc(next_rd)
        to_rm = []
        for sym, pos in portfolio.items():
            stopped = False
            for di in range(ti+1, next_ti+1):
                p = close.iloc[di].get(sym, np.nan)
                if np.isnan(p): continue
                pos['hi'] = max(pos['hi'], p)
                # 收緊邏輯
                gain = (p - pos['ep']) / pos['ep']
                if gain >= tighten_thresh and not pos['tightened']:
                    pos['tightened'] = True
                trl = STOP_TRL_TIGHT if pos['tightened'] else STOP_TRL
                st = max(pos['ep']*(1+STOP_FIX), pos['hi']*(1+trl))
                if p <= st:
                    trades.append({'ret': (p-pos['ep'])/pos['ep']})
                    to_rm.append(sym); stopped = True; break
            if not stopped and (next_rd - dates[pos['ei']]).days >= MAX_HOLD:
                ep2 = close.iloc[next_ti].get(sym, np.nan)
                if not np.isnan(ep2):
                    trades.append({'ret': (ep2-pos['ep'])/pos['ep']})
                    to_rm.append(sym)
        for s in to_rm: portfolio.pop(s, None)

    if not trades: return None
    rets = [t['ret'] for t in trades]
    return {
        'thresh': tighten_thresh_pct,
        'n': len(rets),
        'avg': np.mean(rets)*100,
        'med': np.median(rets)*100,
        'win': sum(r>0 for r in rets)/len(rets)*100,
        'p25': np.percentile(rets,25)*100,
        'p75': np.percentile(rets,75)*100,
    }

def run_slots(top_n):
    portfolio = {}
    trades = []
    for ri, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[ri+1]
        ti = dates.get_loc(rd)
        ranked = mom.iloc[ti].dropna().sort_values(ascending=False)
        held = set(portfolio)
        for sym in ranked.index:
            if len(portfolio) >= top_n: break
            if sym in held: continue
            ep = close.iloc[ti].get(sym, np.nan)
            if np.isnan(ep) or ep <= 0: continue
            portfolio[sym] = {'ei': ti, 'ep': ep, 'hi': ep}; held.add(sym)

        next_ti = dates.get_loc(next_rd)
        to_rm = []
        for sym, pos in portfolio.items():
            stopped = False
            for di in range(ti+1, next_ti+1):
                p = close.iloc[di].get(sym, np.nan)
                if np.isnan(p): continue
                pos['hi'] = max(pos['hi'], p)
                st = max(pos['ep']*(1+STOP_FIX), pos['hi']*(1+STOP_TRL))
                if p <= st:
                    trades.append({'ret': (p-pos['ep'])/pos['ep']})
                    to_rm.append(sym); stopped = True; break
            if not stopped and (next_rd - dates[pos['ei']]).days >= MAX_HOLD:
                ep2 = close.iloc[next_ti].get(sym, np.nan)
                if not np.isnan(ep2):
                    trades.append({'ret': (ep2-pos['ep'])/pos['ep']})
                    to_rm.append(sym)
        for s in to_rm: portfolio.pop(s, None)

    if not trades: return None
    rets = [t['ret'] for t in trades]
    return {
        'slots': top_n, 'n': len(rets),
        'avg': np.mean(rets)*100, 'med': np.median(rets)*100,
        'win': sum(r>0 for r in rets)/len(rets)*100,
        'p25': np.percentile(rets,25)*100, 'p75': np.percentile(rets,75)*100,
    }

# ── Part 1: 收緊門檻 ──────────────────────────────────────────────────────
print("\n【Part 1】收緊門檻...")
tighten_results = []
for t in [15, 20, 25, 30, 40]:
    r = run_tighten(t)
    if r: tighten_results.append(r)

print(f"""
{'='*65}
  動態收緊門檻（追蹤停損 -25% → -15% 的觸發獲利%）
  現行：+25%
{'='*65}
  收緊門檻   N    平均報酬  中位報酬   勝率   P25      P75
  {'-'*55}""")
for r in tighten_results:
    mark = " ← 現行" if r['thresh'] == 25 else ""
    print(f"  +{r['thresh']:>3}%     {r['n']:>4}  {r['avg']:>+7.2f}%  {r['med']:>+7.2f}%  "
          f"{r['win']:>5.1f}%  {r['p25']:>+7.2f}%  {r['p75']:>+7.2f}%{mark}")

# ── Part 2: 槽數 ──────────────────────────────────────────────────────────
print(f"\n【Part 2】最大持倉槽數...")
slot_results = []
for s in [5, 10, 15, 20, 25, 30]:
    r = run_slots(s)
    if r: slot_results.append(r)

print(f"""
{'='*65}
  最大持倉槽數（月度換股，top-N 動能）
  現行：30 槽（實際持倉約 20-25 支）
{'='*65}
  槽數   N    平均報酬  中位報酬   勝率   P25      P75
  {'-'*55}""")
for r in slot_results:
    mark = " ← 現行" if r['slots'] == 30 else ""
    print(f"  {r['slots']:>4}   {r['n']:>4}  {r['avg']:>+7.2f}%  {r['med']:>+7.2f}%  "
          f"{r['win']:>5.1f}%  {r['p25']:>+7.2f}%  {r['p75']:>+7.2f}%{mark}")

print(f"\n  ⚠️  Survivorship bias，不含交易成本\n")
