"""
_factor_quality_backtest.py

問題：能否用「進場時的特徵」區分 CF 型（漲一點還回去）vs SNDK 型（大贏家）？

方法：在 top-20 動能候選池中，用各因子重新排序，取 top-5 進場。
      比較不同因子過濾後的 CAGR / MDD / Calmar。

因子：
  A  純動能        基準，top-20 取前 5
  B  動能一致性    5 個時間框（5d/21d/63d/126d/252d）全正的數量 → 越多越好
  C  動能加速比    21d_mom / 63d_mom → 高 = 近期加速，低 = 衰退中
  D  成交量確認    過去 21 天「上漲日均量 / 下跌日均量」→ 量價俱揚
  E  Vol調整動能   混合動能 / 63日波動率（Sharpe-like）
  F  趨勢狀態      V轉格局（40日低點反彈>20% 且距高<5%）優先
  G  多因子綜合    B+C+D+E 各自 rank → 加總排名

執行：
    conda run -n qt_env python _factor_quality_backtest.py
"""

import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ── 資料 ─────────────────────────────────────────────────────────────────────
OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
if not os.path.exists(OHLCV_PATH):
    print(f"❌ 找不到 {OHLCV_PATH}"); sys.exit(1)

print("載入 OHLCV 快取...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
df_high  = ohlcv["High"]
df_low   = ohlcv["Low"]
df_vol   = ohlcv["Volume"]

for df in [df_close, df_high, df_low, df_vol]:
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

trading_days = df_close.index.sort_values()
n_dates      = len(trading_days)
symbols      = list(df_close.columns)
print(f"  {len(symbols)} 支  /  {n_dates} 交易日  "
      f"({trading_days[0].date()} ~ {trading_days[-1].date()})")

print("取得 SPY...")
spy_raw = yf.Ticker("SPY").history(
    start="2020-12-01", end="2026-03-01", auto_adjust=True)["Close"]
spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)

def spy_ret(t0, t1):
    try:
        p0 = spy_raw.asof(trading_days[t0])
        p1 = spy_raw.asof(trading_days[t1])
        if pd.isna(p0) or pd.isna(p1) or p0 <= 0: return None
        return float(p1 / p0 - 1)
    except Exception:
        return None

# ── 參數 ─────────────────────────────────────────────────────────────────────
MOM_SHORT  = 21
MOM_LONG   = 252
TOP_POOL   = 20     # 動能候選池大小
TOP_N      = 5      # 最終選入數
HOLD_MAX   = 63
REBAL_FREQ = 21
FIXED_STOP = 0.15
TRAIL_STOP = 0.25

START_TI  = MOM_LONG + 10
rebal_tis = list(range(START_TI, n_dates - HOLD_MAX - 5, REBAL_FREQ))

# ── 因子計算（全向量化）──────────────────────────────────────────────────────
print("\n預計算因子矩陣...", flush=True)

close_arr = df_close.values.astype(float)   # (n_dates, n_sym)
vol_arr   = df_vol.values.astype(float)
sym_idx   = {s: i for i, s in enumerate(symbols)}

def price_at(sym, ti):
    if ti < 0 or ti >= n_dates: return None
    v = close_arr[ti, sym_idx[sym]]
    return float(v) if np.isfinite(v) else None

def compute_factors_at(ti):
    """
    在 ti 時間點，對所有股票計算各因子。
    回傳 dict {sym: {mom, consistency, accel, vol_confirm, sharpe, trend}}
    """
    results = {}
    for si, sym in enumerate(symbols):
        # ── 基礎動能 ──────────────────────────────────────────────────────
        def p(offset):
            idx = ti - offset
            if idx < 0 or idx >= n_dates: return None
            v = close_arr[idx, si]
            return float(v) if np.isfinite(v) else None

        p0   = p(0);   p5  = p(5)
        p21  = p(21);  p63 = p(63)
        p126 = p(126); p252 = p(252)

        if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
            continue

        mom = 0.5 * (p0/p21 - 1) + 0.5 * (p0/p252 - 1)

        # ── B: 動能一致性（幾個時間框是正值）──────────────────────────────
        consistency = sum([
            1 if (p5  and p0/p5  - 1 > 0) else 0,
            1 if (p21 and p0/p21 - 1 > 0) else 0,
            1 if (p63 and p0/p63 - 1 > 0) else 0,
            1 if (p126 and p0/p126 - 1 > 0) else 0,
            1 if (p252 and p0/p252 - 1 > 0) else 0,
        ])

        # ── C: 動能加速比（近期 vs 中期）──────────────────────────────────
        mom21 = (p0/p21 - 1) if p21 else None
        mom63 = (p0/p63 - 1) if p63 else None
        accel = (mom21 / abs(mom63)) if (mom63 and mom63 != 0) else 0.0

        # ── D: 成交量確認（上漲日均量 / 下跌日均量）──────────────────────
        win = 21
        seg_close = close_arr[max(0, ti-win):ti+1, si]
        seg_vol   = vol_arr[max(0, ti-win):ti+1, si]
        if len(seg_close) >= 5:
            daily_ret = np.diff(seg_close)
            up_mask   = daily_ret > 0
            dn_mask   = daily_ret < 0
            seg_vol_mid = seg_vol[1:]   # align with diff
            up_vol = seg_vol_mid[up_mask].mean() if up_mask.sum() > 0 else 1.0
            dn_vol = seg_vol_mid[dn_mask].mean() if dn_mask.sum() > 0 else 1.0
            vol_confirm = up_vol / dn_vol if dn_vol > 0 else 1.0
        else:
            vol_confirm = 1.0

        # ── E: Vol 調整動能（Sharpe-like）────────────────────────────────
        vol_win = 63
        seg_c = close_arr[max(0, ti-vol_win):ti+1, si]
        seg_c = seg_c[np.isfinite(seg_c)]
        if len(seg_c) >= 20:
            lr  = np.log(seg_c[1:] / seg_c[:-1])
            ann_vol = float(lr.std() * np.sqrt(252)) if len(lr) >= 15 else None
        else:
            ann_vol = None
        sharpe_mom = (mom / ann_vol) if (ann_vol and ann_vol > 0) else mom

        # ── F: 趨勢狀態（V轉格局）────────────────────────────────────────
        w40 = 40
        seg40 = close_arr[max(0, ti-w40):ti+1, si]
        if len(seg40) >= 20 and np.all(np.isfinite(seg40)):
            low40  = seg40.min()
            high40 = seg40.max()
            bounce = (p0 - low40) / low40 if low40 > 0 else 0
            from_high = (p0 - high40) / high40 if high40 > 0 else 0
            # ↗️ 轉強：反彈>20% 且距高點<5%
            trend_score = 1 if (bounce > 0.20 and from_high > -0.05) else 0
        else:
            trend_score = 0

        results[sym] = {
            "mom":         mom,
            "consistency": consistency,
            "accel":       accel,
            "vol_confirm": vol_confirm,
            "sharpe_mom":  sharpe_mom,
            "trend":       trend_score,
        }

    return results

# ── 模擬函式 ─────────────────────────────────────────────────────────────────
def simulate_hold(sym, ti_entry):
    p0 = price_at(sym, ti_entry)
    if not p0: return None, False
    fixed_px = p0 * (1 - FIXED_STOP)
    peak     = p0
    for ti in range(ti_entry + 1, min(ti_entry + HOLD_MAX + 1, n_dates)):
        px = price_at(sym, ti)
        if px is None: continue
        if px > peak: peak = px
        if px < max(fixed_px, peak * (1 - TRAIL_STOP)):
            return (px - p0) / p0, True
    px = price_at(sym, min(ti_entry + HOLD_MAX, n_dates - 1))
    if px is None: return None, False
    return (px - p0) / p0, False

def run_strategy(strategy_name, selector_fn):
    """
    selector_fn(factor_dict, pool) -> 選出的 top-N symbols
    """
    month_rets   = []
    month_alphas = []
    stop_count   = 0
    trade_count  = 0

    for ri, ti in enumerate(rebal_tis):
        ti_end = rebal_tis[ri + 1] if ri + 1 < len(rebal_tis) else ti + REBAL_FREQ
        ti_end = min(ti_end, n_dates - 1)

        factors = compute_factors_at(ti)

        # 先取動能 top-20 作為候選池
        pool = sorted(factors, key=lambda s: factors[s]["mom"], reverse=True)[:TOP_POOL]

        # 再由 selector 從 pool 中選 top-5
        selected = selector_fn(factors, pool)
        if not selected: continue

        slot_rets = []
        spy_r = spy_ret(ti, ti_end)

        for sym in selected:
            ret, stopped = simulate_hold(sym, ti)
            if ret is None: continue
            slot_rets.append(ret)
            trade_count += 1
            if stopped: stop_count += 1

        if slot_rets:
            avg_ret = np.mean(slot_rets)
            month_rets.append(avg_ret)
            if spy_r is not None:
                month_alphas.append(avg_ret - spy_r)

    if not month_rets:
        return {"cagr": 0, "mdd": 0, "calmar": 0, "stop_rate": 0, "med_alpha": 0}

    equity = np.cumprod([1 + r for r in month_rets])
    n_years = len(rebal_tis) * REBAL_FREQ / 252
    cagr = float(equity[-1]) ** (1 / n_years) - 1

    peak  = np.maximum.accumulate(equity)
    mdd   = float(((equity - peak) / peak).min())
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    stop_rate = stop_count / trade_count if trade_count else 0
    med_alpha = float(np.median(month_alphas)) if month_alphas else 0

    return {
        "cagr": cagr, "mdd": mdd, "calmar": calmar,
        "stop_rate": stop_rate, "med_alpha": med_alpha,
        "trades": trade_count,
    }

# ── 各策略的 selector ────────────────────────────────────────────────────────

def sel_pure_mom(factors, pool):
    """A: 純動能，直接取 pool 前 N"""
    return pool[:TOP_N]

def sel_consistency(factors, pool):
    """B: 在 pool 中，優先一致性高（多個時框都正）"""
    return sorted(pool, key=lambda s: (factors[s]["consistency"],
                                       factors[s]["mom"]), reverse=True)[:TOP_N]

def sel_accel(factors, pool):
    """C: 在 pool 中，優先動能加速中（近期 > 中期）"""
    return sorted(pool, key=lambda s: factors[s]["accel"], reverse=True)[:TOP_N]

def sel_vol_confirm(factors, pool):
    """D: 在 pool 中，優先量價俱揚"""
    return sorted(pool, key=lambda s: factors[s]["vol_confirm"], reverse=True)[:TOP_N]

def sel_sharpe(factors, pool):
    """E: 在 pool 中，優先 vol 調整後動能最高"""
    return sorted(pool, key=lambda s: factors[s]["sharpe_mom"], reverse=True)[:TOP_N]

def sel_trend(factors, pool):
    """F: 優先選 V轉格局（↗️轉強）；同為轉強時再比動能"""
    return sorted(pool, key=lambda s: (factors[s]["trend"],
                                       factors[s]["mom"]), reverse=True)[:TOP_N]

def sel_multi(factors, pool):
    """G: 多因子綜合排名（B+C+D+E 各自排名加總）"""
    n = len(pool)
    if n == 0: return []
    # 各因子在 pool 內排名（越高越好）
    def rank_by(key, reverse=True):
        vals = sorted(pool, key=lambda s: factors[s][key], reverse=reverse)
        return {s: i for i, s in enumerate(vals)}

    r_consist = rank_by("consistency")
    r_accel   = rank_by("accel")
    r_volconf = rank_by("vol_confirm")
    r_sharpe  = rank_by("sharpe_mom")
    r_trend   = rank_by("trend")

    combined = {s: (r_consist[s] + r_accel[s] + r_volconf[s] +
                    r_sharpe[s] + r_trend[s])
                for s in pool}
    return sorted(pool, key=lambda s: combined[s])[:TOP_N]  # 小 rank = 好

# ── 執行所有策略 ─────────────────────────────────────────────────────────────
STRATEGIES = [
    ("A 純動能",       sel_pure_mom),
    ("B 動能一致性",   sel_consistency),
    ("C 動能加速比",   sel_accel),
    ("D 成交量確認",   sel_vol_confirm),
    ("E Vol調整動能",  sel_sharpe),
    ("F 趨勢狀態↗️",  sel_trend),
    ("G 多因子綜合",   sel_multi),
]

print(f"\n候選池 top-{TOP_POOL} → 選 {TOP_N}  /  再平衡 {len(rebal_tis)} 期  /  持有最長 {HOLD_MAX} 日\n")

results = []
for name, sel_fn in STRATEGIES:
    print(f"  {name}...", end="", flush=True)
    r = run_strategy(name, sel_fn)
    r["name"] = name
    results.append(r)
    print(f"  CAGR {r['cagr']*100:+.1f}%  MDD {r['mdd']*100:.1f}%  "
          f"Calmar {r['calmar']:.3f}  月中位alpha {r['med_alpha']*100:+.2f}%")

# ── 輸出 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*76}")
print(f"  動能品質因子回測  S&P500 ~{n_dates//252:.0f}Y  pool-{TOP_POOL}→top-{TOP_N}  月度")
print(f"{'='*76}")
print(f"  {'策略':<18}  {'CAGR':>7}  {'MDD':>7}  {'Calmar':>7}  "
      f"{'停損率':>7}  {'月中位alpha':>10}")
print(f"  {'-'*70}")

best_calmar = max(r["calmar"] for r in results)
best_alpha  = max(r["med_alpha"] for r in results)
for r in results:
    marks = []
    if r["calmar"] == best_calmar: marks.append("Calmar✅")
    if r["med_alpha"] == best_alpha: marks.append("alpha✅")
    marker = "  " + " ".join(marks) if marks else ""
    print(f"  {r['name']:<18}  {r['cagr']*100:>+6.1f}%  "
          f"{r['mdd']*100:>6.1f}%  {r['calmar']:>7.3f}  "
          f"{r['stop_rate']*100:>6.0f}%  "
          f"{r['med_alpha']*100:>+9.2f}%{marker}")

print(f"\n  因子說明：")
print(f"    B 一致性：5d/21d/63d/126d/252d 中正值的數量（0-5）")
print(f"    C 加速比：21d動能 / 63d動能，>1 = 近期加速")
print(f"    D 量確認：過去21日上漲日均量 / 下跌日均量")
print(f"    E Sharpe：混合動能 / 63日年化波動率")
print(f"    F 趨勢：  40日低點反彈>20% 且距高點<5%（V型轉強）")
print(f"    G 多因子：B+C+D+E+F 排名加總")
print(f"\n  ⚠️  Survivorship bias：僅含現有 S&P500 成份股")
