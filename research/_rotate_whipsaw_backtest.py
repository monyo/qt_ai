"""
_rotate_whipsaw_backtest.py

驗證使用者實際遇到的情境：ROTATE 賣出 X 買 Y，結果 Y 動能很快衰退，
之後又要用更高價把 X 買回來（VLO→ECHO→VLO 案例）。

兩階段：
  1. 量化問題：現行 ROTATE 邏輯（gap=10%, protect_days=30）跑歷史，找出所有
     「賣出 X 之後 N 天內又買回 X」的來回單，計算「早知道抱著不動」vs
     「真的照做换股」哪個報酬比較好，算出 whipsaw 的平均成本與發生頻率。
  2. 驗證解法：加一個「冷卻期」規則——剛換股賣掉的標的，N天內不能被選為
     ROTATE買進標的或新倉候選，除非現價已跌破賣出價一定幅度（撿便宜例外）。
     比較加冷卻期前後的整體換股報酬（平均/中位數/勝率），確認解法有效。

執行：
    conda run -n qt_env python research/_rotate_whipsaw_backtest.py
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
GAP = 0.10           # 現行門檻
PROTECT_DAYS = 30    # 現行門檻


def is_strong(ti, sym):
    bp = bounce_df.iloc[ti].get(sym, np.nan)
    fh = from_high_df.iloc[ti].get(sym, np.nan)
    return (not np.isnan(bp)) and (not np.isnan(fh)) and bp > BOUNCE_TH and fh > FH_TH


def run(cooldown_days=0, rebuy_discount=0.0):
    """cooldown_days=0 為現行邏輯（無冷卻期）。
    rebuy_discount>0 表示冷卻期內若現價已跌破賣出價 rebuy_discount 以上，允許例外買回。
    回傳 (trades, rotate_pairs, entry_log, exit_log)"""
    portfolio = {}
    trades = []
    rotate_pairs = []
    entry_log = []   # {sym, date, ti, price, via}
    exit_log = []    # {sym, date, ti, price, via}
    cooldown = {}    # sym -> (exit_ti, exit_price)

    def in_cooldown(sym, ti):
        if cooldown_days <= 0 or sym not in cooldown:
            return False
        exit_ti, exit_price = cooldown[sym]
        if (dates[ti] - dates[exit_ti]).days >= cooldown_days:
            return False
        cur_price = close.iloc[ti].get(sym, np.nan)
        if np.isnan(cur_price):
            return True
        if rebuy_discount > 0 and cur_price <= exit_price * (1 - rebuy_discount):
            return False  # 撿便宜例外，放行
        return True

    for ri, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[ri + 1]
        ti = dates.get_loc(rd)
        mom_today = mom.iloc[ti]
        ranked = mom_today.dropna().sort_values(ascending=False)
        held = set(portfolio)
        candidates = [s for s in ranked.index
                      if s not in held and ranked[s] > 0 and not in_cooldown(s, ti)]
        rot_out, rot_in = set(), set()

        for sym, pos in list(portfolio.items()):
            if sym in rot_out: continue
            if (rd - dates[pos['ei']]).days < PROTECT_DAYS: continue
            if is_strong(ti, sym): continue
            pm = mom_today.get(sym, np.nan)
            if np.isnan(pm): continue
            for c in candidates:
                if c in rot_in: continue
                if ranked[c] - pm > GAP:
                    sp = close.iloc[ti].get(sym, np.nan)
                    ep = close.iloc[pos['ei']].get(sym, np.nan)
                    cp = close.iloc[ti].get(c, np.nan)
                    if np.isnan(sp) or np.isnan(ep) or np.isnan(cp): break
                    trades.append({'ret': (sp - ep) / ep, 'type': 'rot'})
                    rotate_pairs.append({'sym_out': sym, 'ti_out': ti, 'price_out': sp,
                                          'sym_in': c, 'ti_in': ti, 'price_in': cp})
                    exit_log.append({'sym': sym, 'ti': ti, 'price': sp, 'via': 'rotate_out'})
                    entry_log.append({'sym': c, 'ti': ti, 'price': cp, 'via': 'rotate_in'})
                    cooldown[sym] = (ti, sp)
                    del portfolio[sym]
                    portfolio[c] = {'ei': ti, 'ep': cp, 'hi': cp}
                    rot_out.add(sym); rot_in.add(c); break

        held = set(portfolio)
        for sym in ranked.index:
            if len(portfolio) >= TOP_N: break
            if sym in held or sym in rot_in or in_cooldown(sym, ti): continue
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
                    cooldown[sym] = (di, p)
                    to_rm.append(sym); stopped = True; break
            if not stopped and (next_rd - dates[pos['ei']]).days >= MAX_HOLD:
                ep2 = close.iloc[next_ti].get(sym, np.nan)
                if not np.isnan(ep2):
                    trades.append({'ret': (ep2 - pos['ep']) / pos['ep'], 'type': 'time'})
                    exit_log.append({'sym': sym, 'ti': next_ti, 'price': ep2, 'via': 'time'})
                    cooldown[sym] = (next_ti, ep2)
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


# ============================================================
# 階段 1：量化問題 — 現行邏輯（無冷卻期）下的 whipsaw 統計
# ============================================================
print("\n執行基準回測（現行邏輯，無冷卻期）...")
base_trades, base_rotate_pairs, base_entry_log, base_exit_log = run(cooldown_days=0)
base_stats = stats(base_trades)

WHIPSAW_WINDOW = 90  # 賣出後幾天內重新買回，算作 whipsaw

whipsaw_costs = []
whipsaw_days = []
for rp in base_rotate_pairs:
    sym_out, ti_out, price_out = rp['sym_out'], rp['ti_out'], rp['price_out']
    sym_in, ti_in, price_in = rp['sym_in'], rp['ti_in'], rp['price_in']

    # 找 sym_out 之後最近一次重新進場（任何 via）
    rebuys = [e for e in base_entry_log if e['sym'] == sym_out and e['ti'] > ti_out]
    if not rebuys:
        continue
    rebuy = min(rebuys, key=lambda e: e['ti'])
    days_to_rebuy = (dates[rebuy['ti']] - dates[ti_out]).days
    if days_to_rebuy > WHIPSAW_WINDOW:
        continue

    hold_return = (rebuy['price'] - price_out) / price_out

    # 找換進的 sym_in 在這段期間的下場：若期間內出場用出場報酬，否則 mark-to-market
    exits_in_window = [e for e in base_exit_log
                        if e['sym'] == sym_in and ti_out < e['ti'] <= rebuy['ti']]
    if exits_in_window:
        exit_evt = min(exits_in_window, key=lambda e: e['ti'])
        y_return = (exit_evt['price'] - price_in) / price_in
    else:
        mtm_price = close.iloc[rebuy['ti']].get(sym_in, np.nan)
        if np.isnan(mtm_price):
            continue
        y_return = (mtm_price - price_in) / price_in

    whipsaw_costs.append((hold_return - y_return) * 100)
    whipsaw_days.append(days_to_rebuy)

print(f"""
{'='*72}
  階段 1：ROTATE Whipsaw 量化（現行邏輯 gap={GAP*100:.0f}% / protect={PROTECT_DAYS}天）
{'='*72}
  總 ROTATE 換股次數：{len(base_rotate_pairs)}
  其中 {WHIPSAW_WINDOW} 天內把賣掉的標的又買回來的次數：{len(whipsaw_costs)}
  （whipsaw 發生率：{len(whipsaw_costs)/max(len(base_rotate_pairs),1)*100:.1f}%）
""")

if whipsaw_costs:
    arr = np.array(whipsaw_costs)
    days_arr = np.array(whipsaw_days)
    print(f"  平均重新買回天數：{days_arr.mean():.0f} 天（中位數 {np.median(days_arr):.0f} 天）")
    print(f"  「早知道抱著不動」vs「真的換股再買回」的報酬差：")
    print(f"    平均：{arr.mean():+.2f}%（正值代表換股是白繞一圈，抱著不動比較好）")
    print(f"    中位數：{np.median(arr):+.2f}%")
    print(f"    正值比例（換股確實是虧的）：{(arr>0).sum()/len(arr)*100:.1f}%")
    print(f"    最慘一筆：{arr.max():+.2f}%    最賺一筆（換股真的值得）：{arr.min():+.2f}%")

# ============================================================
# 階段 2：驗證解法 — 加冷卻期後整體表現是否改善
# ============================================================
print(f"""
{'='*72}
  階段 2：加冷卻期規則後的整體 ROTATE 表現
{'='*72}
  冷卻天數  撿便宜門檻  N筆   平均報酬  中位報酬   勝率   ROTATE次數
  {'-'*68}""")

configs = [
    (0, 0.0),     # 現行（基準）
    (15, 0.0),
    (30, 0.0),
    (45, 0.0),
    (60, 0.0),
    (30, 0.05),
    (30, 0.10),
    (45, 0.10),
]

results = []
for cd, disc in configs:
    if cd == 0:
        t, s = base_trades, base_stats
    else:
        t, _, _, _ = run(cooldown_days=cd, rebuy_discount=disc)
        s = stats(t)
    if s is None:
        continue
    results.append((cd, disc, s))
    mark = " ← 現行" if cd == 0 else ""
    print(f"  {cd:>6}天  跌破{disc*100:>4.0f}%   {s['n']:>4}  {s['avg']:>+7.2f}%  "
          f"{s['med']:>+7.2f}%  {s['win']:>5.1f}%  {s['rots']:>6}{mark}")

best = max(results, key=lambda r: r[2]['med'])
print(f"\n  中位數報酬最佳組合：冷卻 {best[0]} 天 / 撿便宜門檻 {best[1]*100:.0f}%  →  中位 {best[2]['med']:+.2f}%（vs 現行 {base_stats['med']:+.2f}%）")
print(f"\n  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股，不含交易成本，僅供相對比較參考\n")
