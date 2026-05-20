"""
體制適應性參數回測

假設：熊市/高波動體制下，收緊停損可降低 MDD，且不顯著犧牲 CAGR。

測試四種策略（進退場邏輯相同，只有體制偵測 + 停損參數不同）：
  A（基準）：靜態停損 固定 -15%、追蹤 -25%
  B（廣度體制）：廣度 < 40% 時收緊為 固定 -10%、追蹤 -15%
  C（VIX 體制）：VIX > 25 時收緊為 固定 -10%、追蹤 -15%
  D（雙重確認）：廣度 < 40% 且 VIX > 25 同時成立才收緊

體制切換只影響「新進場」的停損，持倉中已有部位維持進場當時的設定。

回測期間包含 2022 熊市、2025 Q1 急跌，可充分測試體制切換效果。
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf

# ── 參數 ──────────────────────────────────────────────────────────────────
OHLCV_PATH   = "data/_protection_bt_ohlcv.pkl"
CACHE_PATH   = "data/_protection_bt_prices.pkl"
VIX_CACHE    = "data/_ml_vix.pkl"
MOM_SHORT    = 21
MOM_LONG     = 252
MA_WIN       = 50
VOL_WIN      = 63
TOP_N        = 5
REBAL_FREQ   = 21
HOLD_MAX     = 63
INIT_CASH    = 100_000.0

# 正常體制停損
NORMAL_FIXED  = 0.15
NORMAL_TRAIL  = 0.25
# 熊市體制停損
BEAR_FIXED    = 0.10
BEAR_TRAIL    = 0.15

# 體制觸發門檻
BREADTH_BEAR  = 0.40   # 廣度低於此值 = 熊市
VIX_BEAR      = 25.0   # VIX 高於此值 = 高波動

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

# SPY
spy_raw = yf.Ticker("SPY").history(
    start="2020-12-01", end="2026-03-01", auto_adjust=True
)["Close"]
spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)

# VIX
print("下載 VIX...")
if os.path.exists(VIX_CACHE):
    vix_raw = pd.read_pickle(VIX_CACHE)
else:
    vix_raw = yf.Ticker("^VIX").history(
        start="2020-12-01", end="2026-03-01"
    )["Close"]
    vix_raw.index = pd.to_datetime(vix_raw.index).tz_localize(None)
    vix_raw.to_pickle(VIX_CACHE)
vix_raw.index = pd.to_datetime(vix_raw.index)
if vix_raw.index.tz is not None:
    vix_raw.index = vix_raw.index.tz_localize(None)

# ── 廣度預計算 ────────────────────────────────────────────────────────────
print("計算廣度...")
ma50 = prices.rolling(MA_WIN, min_periods=30).mean()
above_ma50 = (prices > ma50).astype(int)
valid_count = (prices.notna() & ma50.notna()).sum(axis=1)
above_count = above_ma50.sum(axis=1)
breadth_series = (above_count / valid_count).where(valid_count > 100)
breadth_series.index = trading_days

START_TI = MOM_LONG + 5
rebal_tis = list(range(START_TI, n_dates - HOLD_MAX - 5, REBAL_FREQ))
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


def get_breadth(ti):
    b = breadth_series.iloc[ti] if 0 <= ti < len(breadth_series) else None
    return float(b) if b is not None and pd.notna(b) else None


def get_vix(ti):
    ref = trading_days[ti]
    idx = vix_raw.index.searchsorted(ref)
    if idx >= len(vix_raw):
        return None
    v = float(vix_raw.iloc[idx])
    return v if np.isfinite(v) else None


def is_bear_regime(ti, strategy):
    b = get_breadth(ti)
    vix = get_vix(ti)
    breadth_bear = (b is not None) and (b < BREADTH_BEAR)
    vix_bear     = (vix is not None) and (vix > VIX_BEAR)

    if strategy == "A":
        return False
    elif strategy == "B":
        return breadth_bear
    elif strategy == "C":
        return vix_bear
    elif strategy == "D":
        return breadth_bear and vix_bear
    return False


def spy_ret(ti, offset):
    ref_date = trading_days[ti]
    idx = spy_raw.index.searchsorted(ref_date)
    if idx + offset >= len(spy_raw):
        return None
    return float(spy_raw.iloc[idx + offset]) / float(spy_raw.iloc[idx]) - 1


# ── 全組合模擬（每日追蹤停損，再平衡點換股）────────────────────────────
def run_portfolio(strategy):
    cash = INIT_CASH
    # {sym: {"shares", "avg_price", "peak", "fixed_stop", "trail_stop"}}
    positions = {}
    portfolio_values = []
    regime_log = []
    rebal_set = set(rebal_tis)

    for ti in range(START_TI, n_dates):
        # ── 每日停損檢查 ──
        to_sell = []
        for sym, pos in positions.items():
            px = price_at(ti, sym)
            if not px:
                continue
            if px > pos["peak"]:
                pos["peak"] = px
            stop_px = max(
                pos["avg_price"] * (1 - pos["fixed_stop"]),
                pos["peak"]     * (1 - pos["trail_stop"])
            )
            if px < stop_px:
                to_sell.append(sym)
        for sym in to_sell:
            px = price_at(ti, sym)
            if px:
                cash += positions[sym]["shares"] * px
            del positions[sym]

        # 每日計算組合總值
        pos_value = sum(
            pos["shares"] * (price_at(ti, sym) or pos["avg_price"])
            for sym, pos in positions.items()
        )
        portfolio_values.append(cash + pos_value)

        if ti not in rebal_set:
            continue

        # ── 再平衡點：判斷體制、換股 ──
        bear = is_bear_regime(ti, strategy)
        regime_log.append({
            "ti": ti,
            "date": str(trading_days[ti].date()),
            "bear": bear,
            "breadth": get_breadth(ti),
            "vix": get_vix(ti),
        })
        fixed_s = BEAR_FIXED if bear else NORMAL_FIXED
        trail_s = BEAR_TRAIL if bear else NORMAL_TRAIL

        # 動能排名
        scores = {s: m for s in symbols if (m := mom_mixed_at(s, ti)) is not None}
        top = sorted(scores, key=scores.get, reverse=True)[:TOP_N]

        # 賣出不在 top
        for sym in list(positions):
            if sym not in top:
                px = price_at(ti, sym)
                if px:
                    cash += positions[sym]["shares"] * px
                del positions[sym]

        # 買入 / 調整
        total_v = cash + sum(
            pos["shares"] * (price_at(ti, sym) or 0)
            for sym, pos in positions.items()
        )
        per_slot = total_v / TOP_N

        for sym in top:
            px = price_at(ti, sym)
            if not px:
                continue
            target_shares = per_slot / px
            if sym in positions:
                diff = target_shares - positions[sym]["shares"]
                cash -= diff * px
                positions[sym]["shares"] = target_shares
                if px > positions[sym]["peak"]:
                    positions[sym]["peak"] = px
                # 體制切換時更新停損參數
                positions[sym]["fixed_stop"] = fixed_s
                positions[sym]["trail_stop"] = trail_s
            else:
                if cash >= per_slot * 0.95:
                    cash -= target_shares * px
                    positions[sym] = {
                        "shares": target_shares,
                        "avg_price": px,
                        "peak": px,
                        "fixed_stop": fixed_s,
                        "trail_stop": trail_s,
                    }

    return np.array(portfolio_values), pd.DataFrame(regime_log)


# ── SPY 基準 ──────────────────────────────────────────────────────────────
spy_start = trading_days[START_TI]
spy_idx = spy_raw.index.searchsorted(spy_start)
spy_slice = spy_raw.iloc[spy_idx:spy_idx + (n_dates - START_TI)].values
spy_curve = INIT_CASH * spy_slice / spy_slice[0] if len(spy_slice) > 0 else None


# ── 績效計算 ──────────────────────────────────────────────────────────────
def calc_metrics(vals, label):
    vals = np.array(vals)
    years = len(vals) / 252
    cagr = (vals[-1] / vals[0]) ** (1 / years) - 1
    peak = np.maximum.accumulate(vals)
    mdd = ((vals - peak) / peak).min()
    daily = np.diff(vals) / vals[:-1]
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    monthly = (np.diff(vals[::21]) / vals[::21][:-1]).std() * np.sqrt(12)
    return {"label": label, "cagr": cagr, "mdd": mdd,
            "calmar": calmar, "sharpe": sharpe, "monthly_vol": monthly}


# ── 執行 ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("  體制適應性停損回測  S&P500 2021-2025")
print("=" * 60)
print(f"  正常體制：固定 -{NORMAL_FIXED*100:.0f}%  追蹤 -{NORMAL_TRAIL*100:.0f}%")
print(f"  熊市體制：固定 -{BEAR_FIXED*100:.0f}%  追蹤 -{BEAR_TRAIL*100:.0f}%")
print(f"  廣度門檻：{BREADTH_BEAR*100:.0f}%  VIX 門檻：{VIX_BEAR:.0f}")
print()

results = {}
regime_data = {}
for strat in ["A", "B", "C", "D"]:
    print(f"  運行策略 {strat}...")
    vals, rlog = run_portfolio(strat)
    results[strat] = calc_metrics(vals, strat)
    regime_data[strat] = rlog

if spy_curve is not None:
    results["SPY"] = calc_metrics(spy_curve, "SPY")

# ── 比較表 ────────────────────────────────────────────────────────────────
print(f"\n\n  {'═'*64}")
print(f"  {'策略':<10} {'CAGR':>8} {'MDD':>8} {'Calmar':>8} {'Sharpe':>8} {'月波動':>8}")
print(f"  {'═'*64}")
descs = {"A": "靜態（現狀）", "B": "廣度體制", "C": "VIX體制", "D": "雙重確認", "SPY": "SPY B&H"}
for k in ["A", "B", "C", "D", "SPY"]:
    r = results[k]
    print(f"  {descs[k]:<10} {r['cagr']*100:>+7.2f}% {r['mdd']*100:>+7.2f}% "
          f"{r['calmar']:>8.3f} {r['sharpe']:>8.3f} {r['monthly_vol']*100:>7.1f}%")
print(f"  {'═'*64}")

# ── 體制觸發分析 ──────────────────────────────────────────────────────────
print(f"\n\n  {'─'*60}")
print("  體制觸發頻率（再平衡點中進入熊市體制的比例）")
print(f"  {'─'*60}")
for strat in ["B", "C", "D"]:
    rlog = regime_data[strat]
    if len(rlog) == 0:
        continue
    bear_n = rlog["bear"].sum()
    total_n = len(rlog)
    print(f"  策略 {strat}：{bear_n}/{total_n} 個再平衡點觸發熊市體制 ({bear_n/total_n*100:.0f}%)")

    # 各體制下的 breadth / VIX 分布
    bear_rows = rlog[rlog["bear"]]
    if len(bear_rows) > 0 and "breadth" in bear_rows.columns:
        avg_b = bear_rows["breadth"].mean()
        avg_v = bear_rows["vix"].mean() if "vix" in bear_rows.columns else None
        print(f"    觸發時平均廣度：{avg_b*100:.0f}%  平均VIX：{avg_v:.1f}" if avg_v else
              f"    觸發時平均廣度：{avg_b*100:.0f}%")

    # 列出觸發的時間段
    if len(bear_rows) > 0:
        dates = bear_rows["date"].tolist()
        print(f"    觸發日期：{', '.join(dates[:6])}{'...' if len(dates) > 6 else ''}")

# ── 2022 熊市區間績效對比 ─────────────────────────────────────────────────
print(f"\n\n  {'─'*60}")
print("  2022 熊市期間表現（SPY -19.4%，2022-01 ~ 2022-12）")
print(f"  {'─'*60}")

# 找 2022 的 ti 範圍
ti_2022_start = int(prices.index.searchsorted(pd.Timestamp("2022-01-01")))
ti_2022_end   = int(prices.index.searchsorted(pd.Timestamp("2023-01-01")))

for strat in ["A", "B", "C", "D"]:
    print(f"  運行策略 {strat} 2022 區間分析...")
vals_by_strat = {}
for strat in ["A", "B", "C", "D"]:
    v, _ = run_portfolio(strat)
    # 對齊到 2022 區間
    offset = START_TI  # portfolio_values[0] 對應 trading_days[START_TI]
    i_start = max(0, ti_2022_start - offset)
    i_end   = min(len(v), ti_2022_end - offset)
    if i_start < i_end:
        seg = v[i_start:i_end]
        ret_2022 = (seg[-1] / seg[0] - 1) * 100
        pk = np.maximum.accumulate(seg)
        mdd_2022 = ((seg - pk) / pk).min() * 100
        vals_by_strat[strat] = (ret_2022, mdd_2022)

print(f"\n  {'策略':<12} {'2022 報酬':>10} {'2022 MDD':>10}")
print(f"  {'─'*36}")
for strat in ["A", "B", "C", "D"]:
    if strat in vals_by_strat:
        ret, mdd = vals_by_strat[strat]
        print(f"  {descs[strat]:<12} {ret:>+9.1f}%  {mdd:>+9.1f}%")
