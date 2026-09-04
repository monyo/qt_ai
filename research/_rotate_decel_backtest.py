"""
_rotate_decel_backtest.py

前三個修法（冷卻期 _rotate_whipsaw_backtest.py、連續確認 _rotate_confirm_backtest.py、
買回門檻提高）全部在賣後買回的「買回端」設防，backtest 顯示都會誤傷更多好交易或效果有限。

使用者指出問題根源可能在「賣出端」：現行邏輯只看「對手動能是否比我強 10%」，
沒有檢查「我自己是不是真的轉弱」——像 VLO 被 ECHO 曇花一現的單月衝刺比下去，
但 VLO 自身動能其實還在增強，賣出的理由並不成立。

本腳本測試：ROTATE 賣出前，額外要求「持股自身動能分數比 N 個月前更差（真的在退步）」，
否則即使有更強的候選者，也不賣。N=1、N=2 月分別測試，並重跑 whipsaw 診斷。

執行：
    /home/mark/miniconda3/envs/qt_env/bin/python research/_rotate_decel_backtest.py
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
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
close = prices.copy()

print("計算特徵...")
mom21 = close.pct_change(21)
mom252 = close.pct_change(252)
mom = 0.5 * mom21 + 0.5 * mom252
rolling_max40 = close.rolling(40).max()
rolling_min40 = close.rolling(40).min()
from_high_df = (close / rolling_max40 - 1) * 100
bounce_df = (close / rolling_min40 - 1) * 100

dates = close.index
rebal_dates = []
prev = None
for d in dates[252:]:
    m = (d.year, d.month)
    if m != prev:
        rebal_dates.append(d); prev = m

TOP_N = 5
MAX_HOLD = 63
STOP_FIX = -0.15
STOP_TRL = -0.25
BOUNCE_TH = 20.0
FH_TH = -15.0
GAP = 0.10
PROTECT_DAYS = 30
WHIPSAW_WINDOW = 90


def is_strong(ti, sym):
    bp = bounce_df.iloc[ti].get(sym, np.nan)
    fh = from_high_df.iloc[ti].get(sym, np.nan)
    return (not np.isnan(bp)) and (not np.isnan(fh)) and bp > BOUNCE_TH and fh > FH_TH


def run(require_decel=False, decel_lookback_months=2):
    """require_decel=True 時，賣出前額外要求：
    持股自身動能分數 < decel_lookback_months 個月前的自身動能分數（自己真的在退步）。
    若自己動能仍持平或上升，即使對手動能差距達標也不賣。"""
    portfolio = {}
    trades = []
    rotate_pairs = []
    entry_log = []
    exit_log = []

    for ri, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[ri + 1]
        ti = dates.get_loc(rd)
        mom_today = mom.iloc[ti]
        prev_ti = None
        if require_decel and ri >= decel_lookback_months:
            prev_rd = rebal_dates[ri - decel_lookback_months]
            prev_ti = dates.get_loc(prev_rd)

        ranked = mom_today.dropna().sort_values(ascending=False)
        held = set(portfolio)
        candidates = [s for s in ranked.index if s not in held and ranked[s] > 0]
        rot_out, rot_in = set(), set()

        for sym, pos in list(portfolio.items()):
            if sym in rot_out: continue
            if (rd - dates[pos['ei']]).days < PROTECT_DAYS: continue
            if is_strong(ti, sym): continue
            pm = mom_today.get(sym, np.nan)
            if np.isnan(pm): continue
            if require_decel and prev_ti is not None:
                pm_prev = mom.iloc[prev_ti].get(sym, np.nan)
                if not np.isnan(pm_prev) and pm >= pm_prev:
                    continue  # 自己動能還在加速/持平，不因對手一時衝刺就賣
            for c in candidates:
                if c in rot_in: continue
                if ranked[c] - pm > GAP:
                    sp = close.iloc[ti].get(sym, np.nan)
                    ep = close.iloc[pos['ei']].get(sym, np.nan)
                    cp = close.iloc[ti].get(c, np.nan)
                    if np.isnan(sp) or np.isnan(ep) or np.isnan(cp): break
                    trades.append({'ret': (sp - ep) / ep, 'type': 'rot'})
                    rotate_pairs.append({'sym_out': sym, 'ti_out': ti, 'price_out': sp,
                                          'sym_in': c, 'price_in': cp})
                    exit_log.append({'sym': sym, 'ti': ti, 'price': sp, 'via': 'rotate_out'})
                    entry_log.append({'sym': c, 'ti': ti, 'price': cp, 'via': 'rotate_in'})
                    del portfolio[sym]
                    portfolio[c] = {'ei': ti, 'ep': cp, 'hi': cp}
                    rot_out.add(sym); rot_in.add(c)
                    break

        held = set(portfolio)
        for sym in ranked.index:
            if len(portfolio) >= TOP_N: break
            if sym in held or sym in rot_in: continue
            ep = close.iloc[ti].get(sym, np.nan)
            if np.isnan(ep) or ep <= 0: continue
            portfolio[sym] = {'ei': ti, 'ep': ep, 'hi': ep}
            entry_log.append({'sym': sym, 'ti': ti, 'price': ep, 'via': 'new'})
            held.add(sym)

        next_ti = dates.get_loc(next_rd)
        to_rm = []
        for sym, pos in portfolio.items():
            stopped = False
            for di in range(ti + 1, next_ti + 1):
                p = close.iloc[di].get(sym, np.nan)
                if np.isnan(p): continue
                pos['hi'] = max(pos['hi'], p)
                st = max(pos['ep'] * (1 + STOP_FIX), pos['hi'] * (1 + STOP_TRL))
                if p <= st:
                    trades.append({'ret': (p - pos['ep']) / pos['ep'], 'type': 'stop'})
                    exit_log.append({'sym': sym, 'ti': di, 'price': p, 'via': 'stop'})
                    to_rm.append(sym); stopped = True; break
            if not stopped and (next_rd - dates[pos['ei']]).days >= MAX_HOLD:
                ep2 = close.iloc[next_ti].get(sym, np.nan)
                if not np.isnan(ep2):
                    trades.append({'ret': (ep2 - pos['ep']) / pos['ep'], 'type': 'time'})
                    exit_log.append({'sym': sym, 'ti': next_ti, 'price': ep2, 'via': 'time'})
                    to_rm.append(sym)
        for s in to_rm:
            portfolio.pop(s, None)

    return trades, rotate_pairs, entry_log, exit_log


def stats(trades):
    rets = [t['ret'] for t in trades]
    return {
        'n': len(rets), 'avg': np.mean(rets) * 100, 'med': np.median(rets) * 100,
        'win': sum(r > 0 for r in rets) / len(rets) * 100,
        'rots': sum(1 for t in trades if t['type'] == 'rot'),
    }


def whipsaw_stats(rotate_pairs, entry_log, exit_log):
    costs = []
    for rp in rotate_pairs:
        sym_out, ti_out, price_out = rp['sym_out'], rp['ti_out'], rp['price_out']
        sym_in, price_in = rp['sym_in'], rp['price_in']
        rebuys = [e for e in entry_log if e['sym'] == sym_out and e['ti'] > ti_out]
        if not rebuys:
            continue
        rebuy = min(rebuys, key=lambda e: e['ti'])
        days_to_rebuy = (dates[rebuy['ti']] - dates[ti_out]).days
        if days_to_rebuy > WHIPSAW_WINDOW:
            continue
        hold_return = (rebuy['price'] - price_out) / price_out
        exits_in_window = [e for e in exit_log
                            if e['sym'] == sym_in and ti_out < e['ti'] <= rebuy['ti']]
        if exits_in_window:
            exit_evt = min(exits_in_window, key=lambda e: e['ti'])
            y_return = (exit_evt['price'] - price_in) / price_in
        else:
            mtm_price = close.iloc[rebuy['ti']].get(sym_in, np.nan)
            if np.isnan(mtm_price):
                continue
            y_return = (mtm_price - price_in) / price_in
        costs.append((hold_return - y_return) * 100)
    return costs


print(f"""
{'='*84}
  ROTATE 賣出端加「自身動能須真的轉弱」條件 vs 現行（只比對手動能）
{'='*84}
  設定                          ROT次數  N筆   平均報酬  中位報酬   勝率    whipsaw次數  whipsaw中位成本
  {'-'*80}""")

configs = [
    ("現行（無自身轉弱要求）", dict(require_decel=False)),
    ("自身動能須比1個月前差", dict(require_decel=True, decel_lookback_months=1)),
    ("自身動能須比2個月前差", dict(require_decel=True, decel_lookback_months=2)),
    ("自身動能須比3個月前差", dict(require_decel=True, decel_lookback_months=3)),
]

for label, kwargs in configs:
    trades, rotate_pairs, entry_log, exit_log = run(**kwargs)
    s = stats(trades)
    costs = whipsaw_stats(rotate_pairs, entry_log, exit_log)
    med_cost = np.median(costs) if costs else float('nan')
    print(f"  {label:<26}  {s['rots']:>5}  {s['n']:>5}  {s['avg']:>+7.2f}%  "
          f"{s['med']:>+7.2f}%  {s['win']:>5.1f}%  {len(costs):>9}   {med_cost:>+7.2f}%")

print(f"\n  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股，不含交易成本，僅供相對比較參考\n")
