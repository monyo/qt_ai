"""
_trend_params_backtest.py

趨勢狀態（轉強）兩個參數的 grid search：
  X 軸：bounce 門檻（10 / 15 / 20 / 25 / 30 %）
  Y 軸：滾動窗口（20 / 30 / 40 / 60 天）

問題：「轉強 = 反彈 >X% AND 距高點 >-15%（已更新）」
      X 是 20%，窗口是 40 天，這兩個有沒有比其他值更好？

評估：轉強股的未來 21 日 alpha（vs SPY）分佈

執行：
    conda run -n qt_env python research/_trend_params_backtest.py
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
spy = close.get("SPY", None)
if spy is None:
    import yfinance as yf
    spy_hist = yf.Ticker("SPY").history(period="max")["Close"]
    spy_hist.index = pd.to_datetime(spy_hist.index).tz_localize(None)
    spy = spy_hist.reindex(close.index).ffill()
    print("  SPY 從 yfinance 補充")

dates = close.index
syms  = [s for s in close.columns if s != "SPY"]

# 月度再平衡點
rebal_dates = []
prev = None
for d in dates[300:]:
    m = (d.year, d.month)
    if m != prev: rebal_dates.append(d); prev = m

FWD_DAYS = 21
FH_THRESH = -15.0  # 已更新的 from_high 門檻

def run_trend_params(bounce_thresh, window):
    rolling_max = close.rolling(window).max()
    rolling_min = close.rolling(window).min()
    from_high   = (close / rolling_max - 1) * 100
    bounce_pct  = (close / rolling_min  - 1) * 100
    spy_fwd = spy.pct_change(FWD_DAYS).shift(-FWD_DAYS)

    strong_alphas  = []  # 轉強股的未來 alpha
    other_alphas   = []  # 非轉強股的未來 alpha（動能前 top-50 中）

    mom21  = close.pct_change(21)
    mom252 = close.pct_change(252)
    mom    = 0.5 * mom21 + 0.5 * mom252

    for rd in rebal_dates[:-2]:
        ti = dates.get_loc(rd)
        if ti + FWD_DAYS >= len(dates): continue
        spy_ret = spy_fwd.iloc[ti]
        if np.isnan(spy_ret): continue

        mom_today = mom.iloc[ti].dropna().sort_values(ascending=False)
        top50 = [s for s in mom_today.index[:50] if s in syms]

        for sym in top50:
            bp = bounce_pct.iloc[ti].get(sym, np.nan)
            fh = from_high.iloc[ti].get(sym, np.nan)
            fwd = close[sym].pct_change(FWD_DAYS).shift(-FWD_DAYS).iloc[ti]
            if np.isnan(bp) or np.isnan(fh) or np.isnan(fwd): continue
            alpha = (fwd - spy_ret) * 100
            is_str = bp > bounce_thresh and fh > FH_THRESH
            if is_str:
                strong_alphas.append(alpha)
            else:
                other_alphas.append(alpha)

    if len(strong_alphas) < 20: return None
    return {
        'bounce': bounce_thresh, 'window': window,
        'n_str': len(strong_alphas),
        'med_str': np.median(strong_alphas),
        'win_str': sum(a > 0 for a in strong_alphas) / len(strong_alphas) * 100,
        'med_oth': np.median(other_alphas),
        'gap': np.median(strong_alphas) - np.median(other_alphas),
    }

bounces = [10, 15, 20, 25, 30]
windows = [20, 30, 40, 60]

print("\n執行 grid search（20 組合）...")
results = []
for b in bounces:
    for w in windows:
        r = run_trend_params(b, w)
        if r: results.append(r)
        print(f"  bounce={b}% window={w}d 完成")

print(f"""
{'='*72}
  轉強參數 Grid Search（from_high 門檻固定 -15%）
  評估：轉強股未來 21 日 alpha（vs SPY）
{'='*72}
  bounce  窗口  轉強N  中位alpha  勝率   vs非轉強  gap
  {'-'*60}""")

current = (20, 40)
for r in sorted(results, key=lambda r: -r['gap']):
    mark = " ← 現行" if (r['bounce'], r['window']) == current else ""
    print(f"  {r['bounce']:>5}%  {r['window']:>4}天  {r['n_str']:>5}  "
          f"{r['med_str']:>+7.2f}%  {r['win_str']:>5.1f}%  "
          f"{r['med_oth']:>+7.2f}%  {r['gap']:>+6.2f}%{mark}")

best = max(results, key=lambda r: r['gap'])
print(f"\n  最大 gap（轉強 vs 非轉強差距）：bounce={best['bounce']}%，窗口={best['window']}天  →  gap {best['gap']:+.2f}%")
print(f"\n  ⚠️  Survivorship bias，不含交易成本\n")
