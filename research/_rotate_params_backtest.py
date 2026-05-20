"""
_rotate_params_backtest.py

ROTATE 兩個核心參數的 grid search：
  X 軸：動能差距門檻 MOM_GAP（5 / 8 / 10 / 12 / 15 %）
  Y 軸：最少持有天數 PROTECT_DAYS（20 / 30 / 45 / 60 天）
  → 共 20 個組合，找最優配置

執行：
    conda run -n qt_env python research/_rotate_params_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

PRICES_PATH = "data/_protection_bt_prices.pkl"
OHLCV_PATH  = "data/_protection_bt_ohlcv.pkl"

print("載入資料...")
prices  = pd.DataFrame(pd.read_pickle(PRICES_PATH))
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None: prices.index = prices.index.tz_localize(None)

ohlcv   = pd.read_pickle(OHLCV_PATH)
close   = prices.copy()

print("計算特徵...")
mom21       = close.pct_change(21)
mom252      = close.pct_change(252)
mom         = 0.5 * mom21 + 0.5 * mom252
rolling_max40 = close.rolling(40).max()
rolling_min40 = close.rolling(40).min()
from_high_df  = (close / rolling_max40 - 1) * 100
bounce_df     = (close / rolling_min40  - 1) * 100

dates = close.index
rebal_dates = []
prev = None
for d in dates[252:]:
    m = (d.year, d.month)
    if m != prev:
        rebal_dates.append(d); prev = m

TOP_N     = 5
MAX_HOLD  = 63
STOP_FIX  = -0.15
STOP_TRL  = -0.25
# 轉強：已更新為 -15%
BOUNCE_TH = 20.0
FH_TH     = -15.0

def is_strong(ti, sym):
    bp = bounce_df.iloc[ti].get(sym, np.nan)
    fh = from_high_df.iloc[ti].get(sym, np.nan)
    return (not np.isnan(bp)) and (not np.isnan(fh)) and bp > BOUNCE_TH and fh > FH_TH

def run(gap_pct, protect_days):
    gap = gap_pct / 100.0
    portfolio = {}
    trades = []
    for ri, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[ri+1]
        ti = dates.get_loc(rd)
        mom_today = mom.iloc[ti]
        ranked = mom_today.dropna().sort_values(ascending=False)
        held = set(portfolio)
        candidates = [s for s in ranked.index if s not in held and ranked[s] > 0]
        rot_out, rot_in = set(), set()

        for sym, pos in list(portfolio.items()):
            if sym in rot_out: continue
            if (rd - dates[pos['ei']]).days < protect_days: continue
            if is_strong(ti, sym): continue
            pm = mom_today.get(sym, np.nan)
            if np.isnan(pm): continue
            for c in candidates:
                if c in rot_in: continue
                if ranked[c] - pm > gap:
                    sp = close.iloc[ti].get(sym, np.nan)
                    ep = close.iloc[pos['ei']].get(sym, np.nan)
                    if np.isnan(sp) or np.isnan(ep): break
                    trades.append({'ret': (sp-ep)/ep, 'type': 'rot'})
                    del portfolio[sym]; portfolio[c] = {'ei': ti, 'ep': close.iloc[ti].get(c,np.nan), 'hi': close.iloc[ti].get(c,np.nan)}
                    rot_out.add(sym); rot_in.add(c); break

        held = set(portfolio)
        for sym in ranked.index:
            if len(portfolio) >= TOP_N: break
            if sym in held or sym in rot_in: continue
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
                    trades.append({'ret': (p-pos['ep'])/pos['ep'], 'type': 'stop'})
                    to_rm.append(sym); stopped = True; break
            if not stopped and (next_rd - dates[pos['ei']]).days >= MAX_HOLD:
                ep2 = close.iloc[next_ti].get(sym, np.nan)
                if not np.isnan(ep2):
                    trades.append({'ret': (ep2-pos['ep'])/pos['ep'], 'type': 'time'})
                    to_rm.append(sym)
        for s in to_rm: portfolio.pop(s, None)

    if not trades: return None
    rets = [t['ret'] for t in trades]
    return {
        'gap': gap_pct, 'days': protect_days, 'n': len(rets),
        'avg': np.mean(rets)*100, 'med': np.median(rets)*100,
        'win': sum(r>0 for r in rets)/len(rets)*100,
        'rots': sum(1 for t in trades if t['type']=='rot'),
        'p25': np.percentile(rets,25)*100, 'p75': np.percentile(rets,75)*100,
    }

gaps  = [5, 8, 10, 12, 15]
days  = [20, 30, 45, 60]

print("\n執行 grid search（20 組合）...")
results = []
for g in gaps:
    for d in days:
        r = run(g, d)
        if r: results.append(r)

print(f"""
{'='*72}
  ROTATE 參數 Grid Search
  轉強保護：bounce>20% AND from_high>-15%（已更新）
{'='*72}
  動能差距  持有天  N    平均報酬  中位報酬   勝率  ROTATE  P25      P75
  {'-'*68}""")

current = (10, 30)
for r in results:
    mark = " ← 現行" if (r['gap'], r['days']) == current else ""
    print(f"  {r['gap']:>5}%  {r['days']:>5}天  {r['n']:>4}  {r['avg']:>+7.2f}%  "
          f"{r['med']:>+7.2f}%  {r['win']:>5.1f}%  {r['rots']:>5}  "
          f"{r['p25']:>+7.2f}%  {r['p75']:>+7.2f}%{mark}")

# 找最佳
best = max(results, key=lambda r: r['med'])
print(f"\n  最佳中位報酬：動能差距 {best['gap']}%，持有天 {best['days']}天  →  中位 {best['med']:+.2f}%")
print(f"\n  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股，不含交易成本\n")
