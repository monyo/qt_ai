"""
倉位加權策略回測

測試六種配置策略（進退場邏輯相同，只有倉位大小不同）：
  A（基準）：等權重，每檔各 1/N
  B（低波動優先）：權重 ∝ 1/vol（波動率平價）
  C（上限平價）：B 但單檔上限 2× 等權重
  D（高波動優先）：權重 ∝ vol（高波動 = 給更多錢）
  E（動能加權）：權重 ∝ momentum_score（動能越強越多）
  F（Sharpe 加權）：權重 ∝ momentum_score / vol（報酬/風險比）

資料：S&P500 2021-2025，月度再平衡，全組合模擬
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf

# ── 參數 ──────────────────────────────────────────────────────────────────
OHLCV_PATH  = "data/_protection_bt_ohlcv.pkl"
CACHE_PATH  = "data/_protection_bt_prices.pkl"
MOM_SHORT   = 21
MOM_LONG    = 252
VOL_WIN     = 63
TOP_N       = 5
REBAL_FREQ  = 21
FIXED_STOP  = 0.15
TRAIL_STOP  = 0.25
INIT_CASH   = 100_000.0
MAX_WEIGHT_MULT = 2.0   # 策略 C：單檔上限 = 等權重 × 此倍數

# ── 載入資料 ──────────────────────────────────────────────────────────────
print("載入資料...")
if os.path.exists(OHLCV_PATH):
    ohlcv = pd.read_pickle(OHLCV_PATH)
    prices = ohlcv["Close"]
else:
    prices = pd.read_pickle(CACHE_PATH)

prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
prices = prices.sort_index()
trading_days = prices.index
n_dates = len(trading_days)
symbols = list(prices.columns)
close_arr = prices.values.astype(float)
sym_idx = {s: i for i, s in enumerate(symbols)}
print(f"資料：{len(symbols)} 支 / {n_dates} 交易日")

spy_raw = yf.Ticker("SPY").history(
    start="2020-12-01", end="2026-03-01", auto_adjust=True
)["Close"]
spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)

START_TI = MOM_LONG + 5
rebal_tis = list(range(START_TI, n_dates - REBAL_FREQ - 5, REBAL_FREQ))
print(f"再平衡點：{len(rebal_tis)} 個\n")


# ── 工具函式 ──────────────────────────────────────────────────────────────
def price_at(ti, sym):
    if ti < 0 or ti >= n_dates:
        return None
    v = close_arr[ti, sym_idx[sym]]
    return float(v) if np.isfinite(v) else None


def mom_mixed_at(sym, ti):
    if ti < MOM_LONG:
        return None
    si = sym_idx[sym]
    def p(off):
        idx = ti - off
        if idx < 0: return None
        v = close_arr[idx, si]
        return float(v) if np.isfinite(v) else None
    p0, p21, p252 = p(0), p(MOM_SHORT), p(MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
        return None
    return 0.5 * (p0/p21 - 1) + 0.5 * (p0/p252 - 1)


def vol_at(sym, ti):
    si = sym_idx[sym]
    seg = close_arr[max(0, ti - VOL_WIN):ti + 1, si]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 20:
        return None
    lr = np.log(seg[1:] / seg[:-1])
    return float(lr.std() * np.sqrt(252)) if len(lr) >= 15 else None


def compute_weights(syms, ti, strategy):
    """回傳 {sym: weight}，weights 加總為 1"""
    n = len(syms)
    if n == 0:
        return {}
    eq_w = 1.0 / n

    if strategy == "A":
        return {s: eq_w for s in syms}

    # 計算各股波動率與動能分數
    vols = {}
    moms = {}
    for s in syms:
        v = vol_at(s, ti)
        vols[s] = v if v and v > 0 else 0.30  # fallback：30% vol
        m = mom_mixed_at(s, ti)
        moms[s] = max(m, 0.001) if m and m > 0 else 0.001  # 只選正動能標的

    if strategy == "B":
        # 1/vol 加權
        raw = {s: 1.0 / vols[s] for s in syms}
    elif strategy == "C":
        # 1/vol 上限版
        raw = {s: 1.0 / vols[s] for s in syms}
        cap = eq_w * MAX_WEIGHT_MULT
        raw = {s: min(w, cap) for s, w in raw.items()}
    elif strategy == "D":
        # vol 正向加權（高波動多給）
        raw = {s: vols[s] for s in syms}
    elif strategy == "E":
        # 動能分數加權
        raw = {s: moms[s] for s in syms}
    elif strategy == "F":
        # Sharpe 加權：momentum / vol
        raw = {s: moms[s] / vols[s] for s in syms}
    else:
        raw = {s: eq_w for s in syms}

    total = sum(raw.values())
    return {s: w / total for s, w in raw.items()}


# ── 全組合模擬 ────────────────────────────────────────────────────────────
def run_portfolio(strategy):
    cash = INIT_CASH
    # 持倉：{sym: {"shares": float, "avg_price": float, "peak": float}}
    positions = {}
    portfolio_values = []  # (ti, total_value)

    # 記錄每日組合價值
    for ti in range(START_TI, n_dates):
        # 每日評估持倉價值
        pos_value = 0.0
        for sym, pos in list(positions.items()):
            px = price_at(ti, sym)
            if px:
                pos_value += pos["shares"] * px
        total_value = cash + pos_value
        portfolio_values.append(total_value)

        # 是否到再平衡點
        if ti not in set(rebal_tis):
            continue

        # ── 停損檢查 ──
        to_sell = []
        for sym, pos in list(positions.items()):
            px = price_at(ti, sym)
            if not px:
                continue
            if px > pos["peak"]:
                pos["peak"] = px
            fixed_px = pos["avg_price"] * (1 - FIXED_STOP)
            trail_px = pos["peak"] * (1 - TRAIL_STOP)
            stop_px = max(fixed_px, trail_px)
            if px < stop_px:
                to_sell.append(sym)

        for sym in to_sell:
            px = price_at(ti, sym)
            if px:
                cash += positions[sym]["shares"] * px
            del positions[sym]

        # ── 動能排名 ──
        scores = {}
        for sym in symbols:
            m = mom_mixed_at(sym, ti)
            if m is not None:
                scores[sym] = m
        top = sorted(scores, key=scores.get, reverse=True)[:TOP_N]

        # ── 賣出不在 top 的持倉 ──
        for sym in list(positions.keys()):
            if sym not in top:
                px = price_at(ti, sym)
                if px:
                    cash += positions[sym]["shares"] * px
                del positions[sym]

        # ── 買入新的 top ──
        total_v = cash + sum(
            positions[s]["shares"] * (price_at(ti, s) or 0)
            for s in positions
        )
        weights = compute_weights(top, ti, strategy)

        # 計算每檔目標持倉金額
        for sym in top:
            target_amt = total_v * weights.get(sym, 0)
            px = price_at(ti, sym)
            if not px:
                continue
            target_shares = target_amt / px

            if sym in positions:
                # 調整至目標
                diff_shares = target_shares - positions[sym]["shares"]
                cash -= diff_shares * px
                positions[sym]["shares"] = target_shares
                if px > positions[sym]["peak"]:
                    positions[sym]["peak"] = px
            else:
                # 新建倉
                if cash >= target_amt * 0.95:  # 有足夠現金
                    cash -= target_shares * px
                    positions[sym] = {
                        "shares": target_shares,
                        "avg_price": px,
                        "peak": px,
                    }

    return np.array(portfolio_values)


# ── SPY 基準 ──────────────────────────────────────────────────────────────
spy_start_ti = START_TI
spy_start_date = trading_days[spy_start_ti]
spy_idx = spy_raw.index.searchsorted(spy_start_date)
spy_vals = spy_raw.iloc[spy_idx:spy_idx + (n_dates - START_TI)].values
if len(spy_vals) > 0:
    spy_curve = INIT_CASH * spy_vals / spy_vals[0]
else:
    spy_curve = None


# ── 績效指標計算 ──────────────────────────────────────────────────────────
def calc_metrics(vals, label):
    vals = np.array(vals)
    if len(vals) < 2:
        return {}
    n_days = len(vals)
    years = n_days / 252

    total_ret = (vals[-1] / vals[0]) - 1
    cagr = (vals[-1] / vals[0]) ** (1 / years) - 1

    # MDD
    peak = np.maximum.accumulate(vals)
    drawdowns = (vals - peak) / peak
    mdd = drawdowns.min()

    # Sharpe（日報酬）
    daily_ret = np.diff(vals) / vals[:-1]
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

    # Monthly vol
    monthly_vals = vals[::21]
    monthly_ret = np.diff(monthly_vals) / monthly_vals[:-1]
    monthly_vol = monthly_ret.std() * np.sqrt(12)

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    print(f"\n  {'─'*54}")
    print(f"  策略 {label}")
    print(f"  {'─'*54}")
    print(f"  最終價值：${vals[-1]:>10,.0f}（初始 ${vals[0]:,.0f}）")
    print(f"  CAGR：{cagr*100:+.2f}%  總報酬：{total_ret*100:+.1f}%")
    print(f"  MDD：{mdd*100:.2f}%  Calmar：{calmar:.3f}")
    print(f"  Sharpe：{sharpe:.3f}  月波動率：{monthly_vol*100:.1f}%")

    return {
        "label": label, "final": vals[-1], "cagr": cagr,
        "mdd": mdd, "calmar": calmar, "sharpe": sharpe,
        "monthly_vol": monthly_vol,
    }


# ── 執行 ──────────────────────────────────────────────────────────────────
print("=" * 58)
print("  倉位加權策略回測  S&P500 2021-2025")
print("=" * 58)
print("  A：等權重（現狀）")
print("  B：低波動優先（1/vol）")
print(f"  C：上限平價（1/vol，單檔上限 {MAX_WEIGHT_MULT:.0f}× 等權重）")
print("  D：高波動優先（vol 正向）")
print("  E：動能加權（momentum_score）")
print("  F：Sharpe 加權（momentum / vol）")

results = {}
for strat in ["A", "B", "C", "D", "E", "F"]:
    print(f"\n  運行策略 {strat}...")
    vals = run_portfolio(strat)
    results[strat] = calc_metrics(vals, strat)

if spy_curve is not None:
    results["SPY"] = calc_metrics(spy_curve, "SPY B&H")

# ── 比較表 ────────────────────────────────────────────────────────────────
print(f"\n\n  {'═'*60}")
print(f"  {'策略':<8} {'CAGR':>8} {'MDD':>8} {'Calmar':>8} {'Sharpe':>8} {'月波動':>8}")
print(f"  {'═'*60}")
for label in ["A", "B", "C", "D", "E", "F", "SPY"]:
    r = results.get(label)
    if r:
        print(f"  {r['label']:<8} {r['cagr']*100:>+7.2f}% {r['mdd']*100:>+7.2f}% "
              f"{r['calmar']:>8.3f} {r['sharpe']:>8.3f} {r['monthly_vol']*100:>7.1f}%")
print(f"  {'═'*60}")

# ── 波動率平價的倉位分布分析 ────────────────────────────────────────────
print(f"\n\n  {'─'*54}")
print("  波動率平價實際倉位分布（策略 B 各持倉占比統計）")
print(f"  {'─'*54}")

sample_weights = []
eq_w = 1.0 / TOP_N
for ti in rebal_tis[:20]:  # 取前 20 個再平衡點
    scores = {}
    for sym in symbols:
        m = mom_mixed_at(sym, ti)
        if m is not None:
            scores[sym] = m
    top = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
    w = compute_weights(top, ti, "B")
    for sym, wt in w.items():
        v = vol_at(sym, ti)
        sample_weights.append({"sym": sym, "weight": wt, "vol": v or 0.30})

df_w = pd.DataFrame(sample_weights)
print(f"  平均權重：{df_w['weight'].mean()*100:.1f}%（等權重應為 {eq_w*100:.0f}%）")
print(f"  最大權重：{df_w['weight'].max()*100:.1f}%  最小：{df_w['weight'].min()*100:.1f}%")
print(f"  超過 2× 等權重的比例：{(df_w['weight'] > eq_w*2).mean()*100:.0f}%")
print(f"  低於 0.5× 等權重的比例：{(df_w['weight'] < eq_w*0.5).mean()*100:.0f}%")
print(f"  對應波動率：avg {df_w['vol'].mean()*100:.0f}%  "
      f"min {df_w['vol'].min()*100:.0f}%  max {df_w['vol'].max()*100:.0f}%")
