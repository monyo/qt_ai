"""
_slots_rebal_backtest.py

比較槽位數 × 再平衡頻率的組合，純動能，等權重，無停損
目標：隔離「槽位多 vs 少」和「換得快 vs 慢」的效果

策略組合：
  A  TOP 5  / 再平衡 21 天（接近現行系統選股核心）
  B  TOP 10 / 再平衡 21 天
  C  TOP 20 / 再平衡 21 天
  D  TOP 30 / 再平衡 21 天
  E  TOP 50 / 再平衡 21 天
  F  TOP 50 / 再平衡  7 天（使用者提案）
  G  TOP 20 / 再平衡  7 天
  H  TOP 5  / 再平衡  7 天
  SPY B&H
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
print("載入 OHLCV 快取...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
symbols  = list(df_close.columns)
dates    = df_close.index
print(f"  {len(symbols)} 支  /  {len(dates)} 日  ({dates[0].date()} ~ {dates[-1].date()})")

MOM_SHORT = 21
MOM_LONG  = 252
START_T   = MOM_LONG + 10

close_arr = {s: df_close[s].values.astype(float) for s in symbols}
spy_c = close_arr.get("SPY")

def get_top_n(t, n):
    """取動能前 N 支（不含 SPY）"""
    scores = []
    for s in symbols:
        if s == "SPY":
            continue
        c = close_arr[s]
        if t < MOM_LONG or np.isnan(c[t]) or np.isnan(c[t-MOM_SHORT]) or np.isnan(c[t-MOM_LONG]):
            continue
        m_s = c[t] / c[t-MOM_SHORT] - 1
        m_l = c[t] / c[t-MOM_LONG]  - 1
        mom = 0.5*m_s + 0.5*m_l
        if mom > 0:
            scores.append((mom, s))
    scores.sort(reverse=True)
    return [s for _, s in scores[:n]]

def simulate(top_n, rebal_freq, label):
    """
    全期模擬：每 rebal_freq 天重新選 top_n，等權重持有到下次再平衡
    回傳：每個再平衡週期的組合報酬（等權平均）
    """
    portfolio_rets = []
    t = START_T
    while t + rebal_freq < len(dates):
        picks = get_top_n(t, top_n)
        if not picks:
            t += rebal_freq
            continue
        t_end = min(t + rebal_freq, len(dates) - 1)
        period_rets = []
        for s in picks:
            c = close_arr[s]
            if np.isnan(c[t]) or np.isnan(c[t_end]) or c[t] <= 0:
                continue
            period_rets.append(c[t_end] / c[t] - 1)
        if period_rets:
            portfolio_rets.append(np.mean(period_rets))
        t += rebal_freq

    arr = np.array(portfolio_rets)
    # 累積報酬（複利）
    cum = np.prod(1 + arr) - 1
    ann_periods = 252 / rebal_freq
    years = len(arr) / ann_periods
    cagr = (1 + cum) ** (1 / years) - 1 if years > 0 else 0
    # 最大回撤（用累積淨值序列）
    nav = np.cumprod(1 + arr)
    nav = np.insert(nav, 0, 1.0)
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    sharpe = (arr.mean() / arr.std() * np.sqrt(ann_periods)) if arr.std() > 0 else 0
    win = (arr > 0).mean() * 100
    return {
        "label": label, "n_periods": len(arr),
        "cum": cum*100, "cagr": cagr*100, "mdd": mdd*100,
        "calmar": calmar, "sharpe": sharpe,
        "mean_period": arr.mean()*100, "win": win,
    }

strategies = [
    (5,  21, "A  TOP5  / 21天"),
    (10, 21, "B  TOP10 / 21天"),
    (20, 21, "C  TOP20 / 21天"),
    (30, 21, "D  TOP30 / 21天"),
    (50, 21, "E  TOP50 / 21天"),
    (50,  7, "F  TOP50 /  7天  ← 使用者提案"),
    (20,  7, "G  TOP20 /  7天"),
    (5,   7, "H  TOP5  /  7天"),
]

print("\n回測中...")
results = []
for top_n, rebal, label in strategies:
    r = simulate(top_n, rebal, label)
    results.append(r)
    print(f"  {label} 完成（{r['n_periods']} 期）")

# SPY B&H
t = START_T
spy_periods = []
rebal = 21
while t + rebal < len(dates) and spy_c is not None:
    spy_periods.append(spy_c[t+rebal] / spy_c[t] - 1)
    t += rebal
spy_arr = np.array(spy_periods)
spy_cum  = np.prod(1 + spy_arr) - 1
spy_years = len(spy_arr) / (252/rebal)
spy_cagr  = (1+spy_cum)**(1/spy_years)-1 if spy_years > 0 else 0
nav = np.cumprod(1 + spy_arr); nav = np.insert(nav,0,1.)
peak = np.maximum.accumulate(nav)
spy_mdd = ((nav-peak)/peak).min()
spy_calmar = spy_cagr / abs(spy_mdd) if spy_mdd != 0 else 0
spy_sharpe = spy_arr.mean()/spy_arr.std()*np.sqrt(252/rebal) if spy_arr.std()>0 else 0

print()
print("=" * 80)
print(f"  {'策略':<26}  {'年化':>6}  {'MDD':>7}  {'Calmar':>7}  {'Sharpe':>7}  {'勝率':>6}")
print("-" * 80)
for r in results:
    print(f"  {r['label']:<26}  {r['cagr']:>+5.1f}%  {r['mdd']:>+6.1f}%  "
          f"{r['calmar']:>7.3f}  {r['sharpe']:>7.3f}  {r['win']:>5.1f}%")
print("-" * 80)
print(f"  {'Z  SPY B&H':<26}  {spy_cagr*100:>+5.1f}%  {spy_mdd*100:>+6.1f}%  "
      f"{spy_calmar:>7.3f}  {spy_sharpe:>7.3f}")
print("=" * 80)

print()
print("重點對照：")
print("  A vs E：槽位 5 vs 50（21天再平衡），看稀釋效果")
print("  E vs F：50槽，21天 vs 7天再平衡，看換得快有沒有幫助")
print("  A vs H：5槽，21天 vs 7天再平衡，看頻率效果")
print("  F vs C：使用者提案 vs TOP20/21天，哪個更好？")
