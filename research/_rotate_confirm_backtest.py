"""
_rotate_confirm_backtest.py

冷卻期（見 _rotate_whipsaw_backtest.py）backtest 顯示無差別擋買回會誤傷更多好交易。
換一個更貼近問題根源的解法：ROTATE 用的是「當月瞬間動能分數」，一次性衝刺
（如 ECHO 案例）會被誤判成「真的轉強」。仿照系統既有的停損兩日確認邏輯，
改成「動能差距要連續 N 個月都成立才真的換股」，用持續性取代單月快照，
篩掉曇花一現、保留真正持續轉強的候選。

比較 CONFIRM_MONTHS = 1（現行，無確認）/ 2 / 3 的整體表現，
並重跑 whipsaw 診斷確認頻率是否下降。

執行：
    conda run -n qt_env python research/_rotate_confirm_backtest.py
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


def run(confirm_months=1):
    """confirm_months=1 為現行（看到一次差距就換）。
    >=2 表示同一檔持股的換股訊號要連續 N 個月都成立（候選不必是同一檔）才真的執行。"""
    portfolio = {}
    trades = []
    rotate_pairs = []
    entry_log = []
    exit_log = []
    pending = {}  # sym_out -> 連續觸發次數

    for ri, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[ri + 1]
        ti = dates.get_loc(rd)
        mom_today = mom.iloc[ti]
        ranked = mom_today.dropna().sort_values(ascending=False)
        held = set(portfolio)
        candidates = [s for s in ranked.index if s not in held and ranked[s] > 0]
        rot_out, rot_in = set(), set()
        triggered_this_month = set()

        for sym, pos in list(portfolio.items()):
            if sym in rot_out: continue
            if (rd - dates[pos['ei']]).days < PROTECT_DAYS: continue
            if is_strong(ti, sym): continue
            pm = mom_today.get(sym, np.nan)
            if np.isnan(pm): continue
            best_c = None
            for c in candidates:
                if c in rot_in: continue
                if ranked[c] - pm > GAP:
                    best_c = c; break
            if best_c is None:
                pending.pop(sym, None)
                continue
            triggered_this_month.add(sym)
            cnt = pending.get(sym, 0) + 1
            if cnt < confirm_months:
                pending[sym] = cnt
                continue
            # 確認次數足夠，執行換股
            sp = close.iloc[ti].get(sym, np.nan)
            ep = close.iloc[pos['ei']].get(sym, np.nan)
            cp = close.iloc[ti].get(best_c, np.nan)
            if np.isnan(sp) or np.isnan(ep) or np.isnan(cp):
                pending.pop(sym, None); continue
            trades.append({'ret': (sp - ep) / ep, 'type': 'rot'})
            rotate_pairs.append({'sym_out': sym, 'ti_out': ti, 'price_out': sp,
                                  'sym_in': best_c, 'ti_in': ti, 'price_in': cp})
            exit_log.append({'sym': sym, 'ti': ti, 'price': sp, 'via': 'rotate_out'})
            entry_log.append({'sym': best_c, 'ti': ti, 'price': cp, 'via': 'rotate_in'})
            del portfolio[sym]
            portfolio[best_c] = {'ei': ti, 'ep': cp, 'hi': cp}
            rot_out.add(sym); rot_in.add(best_c)
            pending.pop(sym, None)

        # 未觸發的持股清掉 pending 計數（避免斷續觸發也累積）
        for sym in list(pending.keys()):
            if sym not in triggered_this_month:
                pending.pop(sym, None)

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
    if not trades:
        return None
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
{'='*78}
  ROTATE 「連續 N 個月確認」vs 現行「單月觸發即換」
{'='*78}
  確認月數  ROTATE次數  N筆(全部)  平均報酬  中位報酬   勝率   whipsaw次數  whipsaw中位成本
  {'-'*74}""")

for cm in [1, 2, 3]:
    trades, rotate_pairs, entry_log, exit_log = run(confirm_months=cm)
    s = stats(trades)
    costs = whipsaw_stats(rotate_pairs, entry_log, exit_log)
    med_cost = np.median(costs) if costs else float('nan')
    mark = " ← 現行" if cm == 1 else ""
    print(f"  {cm:>6}   {len(rotate_pairs):>8}   {s['n']:>7}   {s['avg']:>+7.2f}%  "
          f"{s['med']:>+7.2f}%  {s['win']:>5.1f}%  {len(costs):>9}   "
          f"{med_cost:>+7.2f}%{mark}")

print(f"\n  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股，不含交易成本，僅供相對比較參考\n")
