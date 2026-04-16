"""
_stophunt_resilience_backtest.py

回測：各種反停損獵殺對策的效果比較

基準 A：收盤 < 停損 → 隔日開盤出場（現行系統）

對策說明（基於日線收盤價）：
  B 兩日確認  —— 需 2 個連續收盤 < 停損才出場（避免單日假突破）
  C 成交量確認 —— 跌破停損當日成交量 < 20MA×0.8 → 再等一天
  D VIX 展寬  —— VIX>25 停損×1.5，VIX>35 停損×2.0
  E ATR 停損  —— 停損 = entry - 2×ATR（取代固定 -15%）
  F 組合      —— 兩日確認 + 成交量確認 + VIX 展寬

使用：
    conda run -n qt_env python _stophunt_resilience_backtest.py
"""
import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ── 載入 OHLCV 快取 ────────────────────────────────────────────────────────────
OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
if not os.path.exists(OHLCV_PATH):
    print(f"❌ 找不到 {OHLCV_PATH}")
    print("   請先執行：conda run -n qt_env python _build_ohlcv_cache.py")
    sys.exit(1)

print("載入 OHLCV 快取...")
ohlcv     = pd.read_pickle(OHLCV_PATH)
df_close  = ohlcv["Close"]
df_high   = ohlcv["High"]
df_low    = ohlcv["Low"]
df_volume = ohlcv["Volume"]

symbols = list(df_close.columns)
dates   = df_close.index
n_dates = len(dates)
print(f"  {len(symbols)} 支股票  /  {n_dates} 個交易日  ({dates[0].date()} ~ {dates[-1].date()})")

# ── 下載 VIX ──────────────────────────────────────────────────────────────────
print("下載 VIX...")
vix_raw = yf.download("^VIX", start=dates[0].strftime("%Y-%m-%d"),
                      end=(dates[-1] + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
vix_series = vix_raw["Close"].reindex(dates, method="ffill").fillna(20.0)
vix = vix_series.values.astype(float)

# ── 參數 ──────────────────────────────────────────────────────────────────────
FIXED_STOP     = 0.15
TRAIL_STOP     = 0.25
ATR_WINDOW     = 14
ATR_MULTIPLIER = 2.0
VOL_MA_WINDOW  = 20
VOL_THRESHOLD  = 0.8
HOLD_MAX       = 63
MOM_SHORT      = 21
MOM_LONG       = 252
TOP_N          = 5
REBAL_FREQ     = 21
START_T        = max(MOM_LONG, ATR_WINDOW) + 5

STRATEGIES = {
    "A": "A 基準（現行 -15%/-25%）",
    "B": "B 兩日確認",
    "C": "C 成交量確認",
    "D": "D VIX 展寬停損",
    "E": f"E ATR×{ATR_MULTIPLIER} 動態停損",
    "F": "F 組合（兩日+量+VIX）",
}

# ── 輔助函數 ──────────────────────────────────────────────────────────────────
def get_close(sym):
    return df_close[sym].values.astype(float)

def get_volume(sym):
    return df_volume[sym].values.astype(float)

def calc_atr_pct(c, h, l, t):
    if t < ATR_WINDOW + 1:
        return FIXED_STOP
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
           for i in range(t-ATR_WINDOW, t)]
    atr = np.nanmean(trs)
    return (atr / c[t]) if c[t] > 0 else FIXED_STOP

def calc_vol_ma(v, t):
    if t < VOL_MA_WINDOW:
        return np.nan
    return np.nanmean(v[t-VOL_MA_WINDOW:t])

def calc_momentum(c, t):
    if t < MOM_LONG:
        return None
    p0, ps, pl = c[t], c[t-MOM_SHORT], c[t-MOM_LONG]
    if any(np.isnan([p0, ps, pl])) or ps <= 0 or pl <= 0 or p0 <= 0:
        return None
    return 0.5*(p0/ps - 1) + 0.5*(p0/pl - 1)

# ── 單倉模擬 ──────────────────────────────────────────────────────────────────
def simulate(sym, entry_t, strategy):
    c = get_close(sym)
    h = df_high[sym].values.astype(float)
    l = df_low[sym].values.astype(float)
    v = get_volume(sym)

    entry_price = c[entry_t]
    if np.isnan(entry_price) or entry_price <= 0:
        return None, None, None

    high_wm   = entry_price
    atr_stop  = entry_price * (1 - ATR_MULTIPLIER * calc_atr_pct(c, h, l, entry_t))
    pending_exit = False   # 用於兩日確認：昨日已跌破，今日確認

    for dt in range(1, HOLD_MAX + 1):
        t = entry_t + dt
        if t >= n_dates:
            break

        ct = c[t]
        vt = v[t]
        vix_t = vix[t]

        if np.isnan(ct) or ct <= 0:
            continue

        high_wm = max(high_wm, ct)

        # 計算基準停損位
        stop = max(entry_price*(1-FIXED_STOP), high_wm*(1-TRAIL_STOP))

        # VIX 展寬（D / F）
        if strategy in ("D", "F"):
            mult = 2.0 if vix_t > 35 else (1.5 if vix_t > 25 else 1.0)
            if mult > 1.0:
                # 展寬 = 把停損往下移
                stop = entry_price - (entry_price - stop) * mult
                stop = max(stop, entry_price * 0.65)   # 最寬不超過 -35%

        # ATR 停損（E）
        if strategy == "E":
            stop = atr_stop

        triggered = (ct < stop)

        if strategy == "A":
            if triggered:
                return ct/entry_price - 1, dt, "stop"

        elif strategy == "B":
            # 兩日確認：需連續兩日收盤 < 停損才出場
            if triggered:
                if pending_exit:
                    return ct/entry_price - 1, dt, "stop"
                else:
                    pending_exit = True
            else:
                pending_exit = False

        elif strategy == "C":
            # 成交量確認：量不足視為假突破，延一天
            if triggered:
                vol_ma = calc_vol_ma(v, t)
                if not np.isnan(vol_ma) and vt < vol_ma * VOL_THRESHOLD:
                    # 低量，等一天
                    if pending_exit:
                        return ct/entry_price - 1, dt, "stop"
                    pending_exit = True
                else:
                    return ct/entry_price - 1, dt, "stop"
            else:
                pending_exit = False

        elif strategy in ("D", "E"):
            if triggered:
                return ct/entry_price - 1, dt, "stop"

        elif strategy == "F":
            # 組合：兩日確認 + 成交量確認 + VIX 展寬（stop 已調整）
            if triggered:
                vol_ma = calc_vol_ma(v, t)
                vol_ok = np.isnan(vol_ma) or (vt >= vol_ma * VOL_THRESHOLD)
                if vol_ok:
                    if pending_exit:
                        return ct/entry_price - 1, dt, "stop"
                    pending_exit = True
                else:
                    pending_exit = True   # 量不足也要先標記，但不立即出
            else:
                pending_exit = False

    # timeout
    t_exit = min(entry_t + HOLD_MAX, n_dates - 1)
    ct = c[t_exit]
    if np.isnan(ct) or ct <= 0:
        return None, None, None
    return ct/entry_price - 1, HOLD_MAX, "timeout"

# ── 月度回測 ──────────────────────────────────────────────────────────────────
print("\n執行月度動能回測（65 個再平衡點）...")
results = {s: {"rets": [], "hdays": [], "n_stop": 0, "n_timeout": 0} for s in STRATEGIES}

closes_cache = {sym: get_close(sym) for sym in symbols}
rebal_pts    = list(range(START_T, n_dates - HOLD_MAX - 5, REBAL_FREQ))

for ri, t in enumerate(rebal_pts):
    if ri % 15 == 0:
        print(f"  {ri+1}/{len(rebal_pts)}  {dates[t].date()}")

    moms = [(sym, calc_momentum(closes_cache[sym], t)) for sym in symbols]
    moms = [(s, m) for s, m in moms if m is not None]
    moms.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in moms[:TOP_N]]

    for sym in top:
        for s in STRATEGIES:
            ret, hd, reason = simulate(sym, t, s)
            if ret is None or np.isnan(ret):
                continue
            results[s]["rets"].append(ret)
            results[s]["hdays"].append(hd)
            if reason == "stop":
                results[s]["n_stop"] += 1
            else:
                results[s]["n_timeout"] += 1

# ── 輸出 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  停損獵殺對策回測結果")
print(f"{'='*72}")
print(f"  期間: {dates[START_T].date()} ~ {dates[-1].date()}  "
      f"再平衡: {len(rebal_pts)} 次  每次 top {TOP_N}\n")

print(f"  {'策略':<30} {'N':>5}  {'平均報酬':>9}  {'中位報酬':>9}  "
      f"{'勝率':>7}  {'停損率':>7}  {'平均持天':>8}")
print(f"  {'-'*74}")

base_stop = results["A"]["n_stop"]
summary = {}
for s, label in STRATEGIES.items():
    rets = np.array(results[s]["rets"])
    hd   = np.array(results[s]["hdays"])
    n    = len(rets)
    if n == 0:
        continue
    mean_r  = np.nanmean(rets) * 100
    med_r   = np.nanmedian(rets) * 100
    win     = np.nanmean(rets > 0) * 100
    n_stop  = results[s]["n_stop"]
    stop_rt = n_stop / n * 100
    mean_hd = np.nanmean(hd)
    summary[s] = (n, mean_r, med_r, win, n_stop, stop_rt, mean_hd)
    print(f"  {label:<30} {n:>5}  {mean_r:>+8.2f}%  {med_r:>+8.2f}%  "
          f"{win:>6.1f}%  {stop_rt:>6.1f}%  {mean_hd:>7.1f}天")

# 停損次數比較
print(f"\n{'='*72}")
print("  停損次數對比（vs 基準 A）")
print(f"{'='*72}")
print(f"  基準 A 停損次數：{base_stop}")
for s, label in list(STRATEGIES.items())[1:]:
    ns = results[s]["n_stop"]
    diff = base_stop - ns
    pct  = diff / base_stop * 100 if base_stop > 0 else 0
    print(f"  {label:<30}  停損 {ns:>3} 次  (少觸發 {diff:>+3} 次  {pct:>+.1f}%)")

# 報酬分位數（尾端風險）
print(f"\n{'='*72}")
print("  報酬分位數（P10=壞情況  P90=好情況）")
print(f"{'='*72}")
print(f"  {'策略':<30}  {'P10':>8}  {'P25':>8}  {'P50':>8}  {'P75':>8}  {'P90':>8}")
print(f"  {'-'*68}")
for s, label in STRATEGIES.items():
    rets = np.array(results[s]["rets"]) * 100
    if len(rets) == 0:
        continue
    vals = [np.nanpercentile(rets, p) for p in [10, 25, 50, 75, 90]]
    print(f"  {label:<30}  {vals[0]:>+7.1f}%  {vals[1]:>+7.1f}%  "
          f"{vals[2]:>+7.1f}%  {vals[3]:>+7.1f}%  {vals[4]:>+7.1f}%")

# 平均報酬 vs 基準
print(f"\n{'='*72}")
print("  vs 基準 A 的 alpha 差")
print(f"{'='*72}")
base_mean = np.nanmean(results["A"]["rets"]) * 100
for s, label in list(STRATEGIES.items())[1:]:
    mean_r = np.nanmean(results[s]["rets"]) * 100
    diff   = mean_r - base_mean
    print(f"  {label:<30}  平均報酬 {mean_r:>+6.2f}%  vs A: {diff:>+.2f}%")

print(f"\n  ⚠️  Survivorship bias：樣本為 S&P500 現有成份股")
print(f"  ⚠️  不含交易成本、滑點")
