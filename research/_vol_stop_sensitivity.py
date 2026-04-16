"""
_vol_stop_sensitivity.py

掃描中波動層（35-60%）的固定停損門檻，從 -16% 到 -24%（每格 1%）
追蹤停損同比例縮放（維持 fixed:trailing ≈ 1:1.5）

執行：
    conda run -n qt_env python _vol_stop_sensitivity.py
"""

import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
df_close.index = pd.to_datetime(df_close.index)
if df_close.index.tz is not None:
    df_close.index = df_close.index.tz_convert(None)

trading_days = df_close.index.sort_values()
n_dates      = len(trading_days)
symbols      = list(df_close.columns)
print(f"載入：{len(symbols)} 支 / {n_dates} 日")

spy_raw = yf.Ticker("SPY").history(start="2020-12-01", end="2026-03-01", auto_adjust=True)["Close"]
spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)

MOM_SHORT=21; MOM_LONG=252; VOL_WIN=63; TOP_N=5; HOLD_MAX=63; REBAL_FREQ=21
VOL_LOW=0.35; VOL_HIGH=0.60
START_TI  = MOM_LONG + 5
rebal_tis = list(range(START_TI, n_dates - HOLD_MAX - 5, REBAL_FREQ))
close_arr = df_close.values.astype(float)
sym_idx   = {s: i for i, s in enumerate(symbols)}

def price_at(sym, ti):
    if ti < 0 or ti >= n_dates: return None
    v = close_arr[ti, sym_idx[sym]]
    return float(v) if np.isfinite(v) else None

def mom_mixed_at(sym, ti):
    if ti < MOM_LONG: return None
    si = sym_idx[sym]
    def p(off):
        idx = ti - off
        if idx < 0: return None
        v = close_arr[idx, si]
        return float(v) if np.isfinite(v) else None
    p0, p21, p252 = p(0), p(21), p(252)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0): return None
    return 0.5*(p0/p21-1) + 0.5*(p0/p252-1)

def vol_at(sym, ti):
    si  = sym_idx[sym]
    seg = close_arr[max(0,ti-VOL_WIN):ti+1, si]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 20: return None
    lr  = np.log(seg[1:]/seg[:-1])
    return float(lr.std()*np.sqrt(252)) if len(lr) >= 15 else None

def simulate_hold(sym, ti_entry, fixed_stop, trail_stop):
    p0 = price_at(sym, ti_entry)
    if not p0: return None
    fixed_px = p0 * (1 - fixed_stop)
    peak     = p0
    for ti in range(ti_entry+1, min(ti_entry+HOLD_MAX+1, n_dates)):
        px = price_at(sym, ti)
        if px is None: continue
        if px > peak: peak = px
        if px < max(fixed_px, peak*(1-trail_stop)):
            return (px-p0)/p0
    px = price_at(sym, min(ti_entry+HOLD_MAX, n_dates-1))
    return (px-p0)/p0 if px else None

def run_sim(mid_fixed, mid_trail):
    month_rets = []
    for ti in rebal_tis:
        scores = {}
        for sym in symbols:
            m = mom_mixed_at(sym, ti)
            if m is not None: scores[sym] = m
        top = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
        if not top: continue
        slot_rets = []
        for sym in top:
            v = vol_at(sym, ti)
            if v is None or v < VOL_LOW:
                fs, ts = 0.15, 0.25
            elif v < VOL_HIGH:
                fs, ts = mid_fixed, mid_trail
            else:
                fs, ts = 0.25, 0.35
            ret = simulate_hold(sym, ti, fs, ts)
            if ret is not None: slot_rets.append(ret)
        if slot_rets: month_rets.append(np.mean(slot_rets))
    if not month_rets: return None
    equity  = np.cumprod([1+r for r in month_rets])
    n_years = len(rebal_tis) * REBAL_FREQ / 252
    cagr    = float(equity[-1])**(1/n_years) - 1
    pk      = np.maximum.accumulate(equity)
    mdd     = float(((equity-pk)/pk).min())
    calmar  = cagr/abs(mdd) if mdd else 0
    return {"cagr": cagr, "mdd": mdd, "calmar": calmar}

print(f"\n掃描中波動層固定停損（低層 -15%/-25%，高層 -25%/-35% 不變）\n")
print(f"  {'中層固定':>8}  {'中層追蹤':>8}  {'CAGR':>8}  {'MDD':>8}  {'Calmar':>8}")
print(f"  {'-'*50}")

results = []
for fixed_pct in range(15, 26):
    mid_fixed = fixed_pct / 100
    mid_trail = round(mid_fixed * 1.5, 2)
    r = run_sim(mid_fixed, mid_trail)
    if r:
        results.append((mid_fixed, mid_trail, r))
        marker = " ←基準" if fixed_pct == 22 else ""
        print(f"  -{fixed_pct:>3}%      -{mid_trail*100:.0f}%  "
              f"{r['cagr']*100:>+7.1f}%  {r['mdd']*100:>7.1f}%  {r['calmar']:>8.3f}{marker}")

best_c = max(results, key=lambda x: x[2]["calmar"])
best_r = max(results, key=lambda x: x[2]["cagr"])
print(f"\n  {'='*50}")
print(f"  最佳 Calmar：-{best_c[0]*100:.0f}% / -{best_c[1]*100:.0f}%  "
      f"CAGR {best_c[2]['cagr']*100:+.1f}%  Calmar {best_c[2]['calmar']:.3f}")
print(f"  最佳 CAGR  ：-{best_r[0]*100:.0f}% / -{best_r[1]*100:.0f}%  "
      f"CAGR {best_r[2]['cagr']*100:+.1f}%  Calmar {best_r[2]['calmar']:.3f}")
