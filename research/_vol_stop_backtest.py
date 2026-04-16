"""
_vol_stop_backtest.py

兩個問題：

【問題 1】停損後事件研究（回答「冷卻期有沒有意義？」）
  停損觸發後，標的接下來 21/42/63 天的報酬分佈是正還是負？
  按進場時波動率分組，看高波動 vs 低波動的差異。

【問題 2】波動率調整停損（回答「更寬鬆的停損能提升 alpha？」）
  比較四種停損策略：
    A  標準        固定 -15% / 追蹤 -25%（現行）
    B  寬鬆        固定 -20% / 追蹤 -30%
    C  波動率分層  low vol: -15%/-25%, mid: -20%/-30%, high: -25%/-35%
    D  波動率分層  low vol: -15%/-25%, mid: -22%/-33%, high: -30%/-40%

  月度選股，top-5 動能，持有最多 63 天。

執行：
    conda run -n qt_env python _vol_stop_backtest.py
"""

import os, sys, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ── 資料 ─────────────────────────────────────────────────────────────────────
OHLCV_PATH = "data/_protection_bt_ohlcv.pkl"
if not os.path.exists(OHLCV_PATH):
    print(f"❌ 找不到 {OHLCV_PATH}")
    sys.exit(1)

print("載入 OHLCV 快取...")
ohlcv    = pd.read_pickle(OHLCV_PATH)
df_close = ohlcv["Close"]
df_high  = ohlcv["High"]

df_close.index = pd.to_datetime(df_close.index)
df_high.index  = pd.to_datetime(df_high.index)
if df_close.index.tz is not None:
    df_close.index = df_close.index.tz_convert(None)
    df_high.index  = df_high.index.tz_convert(None)

trading_days = df_close.index.sort_values()
n_dates      = len(trading_days)
symbols      = list(df_close.columns)
print(f"  {len(symbols)} 支股票  /  {n_dates} 交易日  ({trading_days[0].date()} ~ {trading_days[-1].date()})")

# SPY 基準
print("取得 SPY 報酬...")
spy_raw = yf.Ticker("SPY").history(start="2020-12-01", end="2026-03-01", auto_adjust=True)["Close"]
spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None)
def spy_ret(t0, t1):
    """t0 ~ t1 之間 SPY 報酬（ti 為 trading_days 索引）"""
    try:
        d0 = trading_days[t0]
        d1 = trading_days[t1]
        p0 = spy_raw.asof(d0)
        p1 = spy_raw.asof(d1)
        if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
            return None
        return float(p1 / p0 - 1)
    except Exception:
        return None

# ── 參數 ─────────────────────────────────────────────────────────────────────
MOM_SHORT  = 21
MOM_LONG   = 252
VOL_WIN    = 63
TOP_N      = 5
HOLD_MAX   = 63      # 最長持有天數
REBAL_FREQ = 21      # 每 21 交易日重選

# 月度再平衡點（2021 之後）
START_TI = VOL_WIN + MOM_LONG + 5
rebal_tis = list(range(START_TI, n_dates - HOLD_MAX - 5, REBAL_FREQ))

# 停損策略定義：(fixed, trailing)
STOP_STRATEGIES = {
    "A 標準 -15%/-25%":  ("fixed", -0.15, -0.25),
    "B 寬鬆 -20%/-30%":  ("fixed", -0.20, -0.30),
    "C 分層 mid=-20%":   ("vol",   -0.20, -0.30),   # 分層，中段參數
    "D 分層 mid=-22%":   ("vol",   -0.22, -0.33),   # 分層，更寬中段
}

# 波動率分層門檻
VOL_LOW  = 0.35   # < 35%: 低波動
VOL_HIGH = 0.60   # > 60%: 高波動

def get_stops_vol(vol, mid_fixed, mid_trail):
    """根據波動率分層決定停損"""
    if vol is None or vol < VOL_LOW:
        return -0.15, -0.25          # 低波動：標準
    elif vol < VOL_HIGH:
        return mid_fixed, mid_trail  # 中波動：參數化
    else:
        return -0.25, -0.35          # 高波動：寬鬆

# ── 工具函式 ─────────────────────────────────────────────────────────────────
def price_at(sym, ti):
    if ti < 0 or ti >= n_dates:
        return None
    v = df_close[sym].iloc[ti]
    return float(v) if pd.notna(v) else None

def high_at(sym, ti):
    if ti < 0 or ti >= n_dates:
        return None
    v = df_high[sym].iloc[ti]
    return float(v) if pd.notna(v) else None

def mom_mixed_at(sym, ti):
    if ti < MOM_LONG:
        return None
    p0   = price_at(sym, ti)
    p21  = price_at(sym, ti - MOM_SHORT)
    p252 = price_at(sym, ti - MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
        return None
    return 0.5 * (p0/p21 - 1) + 0.5 * (p0/p252 - 1)

def vol_at(sym, ti):
    """63 日年化波動率"""
    if ti < VOL_WIN:
        return None
    seg = df_close[sym].iloc[ti - VOL_WIN: ti].dropna()
    if len(seg) < 20:
        return None
    lr  = np.log(seg / seg.shift(1)).dropna()
    if len(lr) < 15:
        return None
    return float(lr.std() * np.sqrt(252))

def simulate_hold(sym, ti_entry, fixed_stop, trail_stop):
    """
    從 ti_entry 持有最多 HOLD_MAX 天，模擬停損。
    回傳 (exit_ti, exit_price, stop_triggered)
    """
    p0      = price_at(sym, ti_entry)
    if not p0:
        return None, None, False
    fixed_px  = p0 * (1 + fixed_stop)   # fixed_stop 是負數
    high_wm   = p0
    stop_hit  = False
    exit_ti   = min(ti_entry + HOLD_MAX, n_dates - 1)
    exit_px   = None

    for dt in range(1, HOLD_MAX + 1):
        t2 = ti_entry + dt
        if t2 >= n_dates:
            break
        # 用當日高點更新 watermark
        h2 = high_at(sym, t2)
        if h2 and pd.notna(h2):
            high_wm = max(high_wm, h2)
        c2 = price_at(sym, t2)
        if not c2:
            continue
        trail_px = high_wm * (1 + trail_stop)   # trail_stop 是負數
        if c2 <= fixed_px or c2 <= trail_px:
            exit_ti = t2
            exit_px = c2
            stop_hit = True
            break

    if exit_px is None:
        exit_px = price_at(sym, exit_ti)
    return exit_ti, exit_px, stop_hit

# ═══════════════════════════════════════════════════════════════════════════════
# 問題 1：停損後事件研究
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  【問題 1】停損後事件研究")
print("="*70)
print("  掃描中（使用標準 -15%/-25% 停損）...\n")

post_stop_events = []  # (fwd_21, fwd_42, fwd_63, vol_cat, market_ret_on_stop)

for ti in rebal_tis:
    # 選取當期動能前 20 支（放寬池，讓停損事件夠多）
    scores = []
    for sym in symbols:
        m = mom_mixed_at(sym, ti)
        if m and m > 0:
            scores.append((sym, m))
    scores.sort(key=lambda x: -x[1])
    pool = [s for s, _ in scores[:20]]

    for sym in pool:
        p_entry = price_at(sym, ti)
        if not p_entry:
            continue
        v = vol_at(sym, ti)

        exit_ti, exit_px, stop_hit = simulate_hold(sym, ti, -0.15, -0.25)
        if not stop_hit or exit_px is None:
            continue

        # 市場在停損日的報酬（判斷是宏觀驅動還是個股驅動）
        mkt = spy_ret(ti, exit_ti)

        # 停損後的 forward returns
        fwds = {}
        for fwd in [21, 42, 63]:
            t_fwd = exit_ti + fwd
            if t_fwd >= n_dates:
                fwds[fwd] = None
                continue
            p_fwd = price_at(sym, t_fwd)
            spy_f = spy_ret(exit_ti, exit_ti + fwd)
            if p_fwd and spy_f is not None:
                raw_ret  = p_fwd / exit_px - 1
                fwds[fwd] = raw_ret - spy_f   # alpha vs SPY
            else:
                fwds[fwd] = None

        if v is None:
            vc = "unknown"
        elif v < VOL_LOW:
            vc = "低 (<35%)"
        elif v < VOL_HIGH:
            vc = "中 (35-60%)"
        else:
            vc = "高 (>60%)"

        post_stop_events.append({
            "fwd_21":  fwds[21],
            "fwd_42":  fwds[42],
            "fwd_63":  fwds[63],
            "vol_cat": vc,
            "mkt_ret": mkt,
            "vol":     v,
        })

print(f"  停損事件數：{len(post_stop_events)}\n")

# ── 輸出 ───────────────────────────────────────────────────────────────────
def fmt_stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "N/A"
    arr = np.array(v)
    pos_pct = np.mean(arr > 0) * 100
    return (f"平均 {np.mean(arr)*100:>+5.1f}%  中位 {np.median(arr)*100:>+5.1f}%  "
            f"正報酬率 {pos_pct:>4.0f}%  N={len(arr)}")

print(f"  {'分組':<14}  21日 alpha                                    42日 alpha                                    63日 alpha")
print(f"  {'─'*66}")

# 全體
print(f"  {'全體':<14}  {fmt_stats([e['fwd_21'] for e in post_stop_events])}")
print(f"  {'':14}  {fmt_stats([e['fwd_42'] for e in post_stop_events])}")
print(f"  {'':14}  {fmt_stats([e['fwd_63'] for e in post_stop_events])}")
print()

# 按波動率分組
for vc in ["低 (<35%)", "中 (35-60%)", "高 (>60%)"]:
    sub = [e for e in post_stop_events if e["vol_cat"] == vc]
    if not sub:
        continue
    print(f"  {vc:<14}  21日: {fmt_stats([e['fwd_21'] for e in sub])}")
    print(f"  {'':14}  42日: {fmt_stats([e['fwd_42'] for e in sub])}")
    print(f"  {'':14}  63日: {fmt_stats([e['fwd_63'] for e in sub])}")
    print()

# 宏觀驅動 vs 個股驅動（市場當日跌 >3% 視為宏觀）
macro = [e for e in post_stop_events if e["mkt_ret"] is not None and e["mkt_ret"] < -0.03]
idio  = [e for e in post_stop_events if e["mkt_ret"] is not None and e["mkt_ret"] >= -0.03]
print(f"  {'宏觀驅動停損':<14}  63日: {fmt_stats([e['fwd_63'] for e in macro])}")
print(f"  {'個股驅動停損':<14}  63日: {fmt_stats([e['fwd_63'] for e in idio])}")

# ═══════════════════════════════════════════════════════════════════════════════
# 問題 2：波動率分層停損 portfolio simulation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  【問題 2】波動率調整停損 投組模擬")
print("="*70)
print("  月度選股 Top-5，持有最多 63 天\n")

def run_portfolio(strategy_name, stop_type, mid_fixed, mid_trail):
    """
    完整投組模擬。
    stop_type: "fixed" = 全部用 (mid_fixed, mid_trail)
               "vol"   = 按波動率分層
    回傳：月度 alpha list, stop_rate (被停損比例)
    """
    monthly_alphas = []
    stop_count     = 0
    total_count    = 0

    for ti in rebal_tis:
        # 選 top-5
        scores = []
        for sym in symbols:
            m = mom_mixed_at(sym, ti)
            if m and m > 0:
                scores.append((sym, m, vol_at(sym, ti)))
        scores.sort(key=lambda x: -x[1])
        picks = scores[:TOP_N]
        if not picks:
            continue

        period_rets = []
        for sym, _, v in picks:
            p0 = price_at(sym, ti)
            if not p0:
                continue

            if stop_type == "fixed":
                fs, ts = mid_fixed, mid_trail
            else:
                fs, ts = get_stops_vol(v, mid_fixed, mid_trail)

            exit_ti, exit_px, stop_hit = simulate_hold(sym, ti, fs, ts)
            if exit_px is None:
                continue

            ret    = exit_px / p0 - 1
            mkt    = spy_ret(ti, exit_ti)
            if mkt is None:
                continue
            alpha  = ret - mkt
            period_rets.append(alpha)
            total_count += 1
            if stop_hit:
                stop_count += 1

        if period_rets:
            monthly_alphas.append(np.mean(period_rets))

    stop_rate = stop_count / total_count if total_count else 0
    return monthly_alphas, stop_rate

results = {}
for name, (st, mf, mt) in STOP_STRATEGIES.items():
    print(f"  計算 {name}...")
    alphas, stop_rate = run_portfolio(name, st, mf, mt)
    results[name] = (alphas, stop_rate)

# ── 輸出摘要 ──────────────────────────────────────────────────────────────────
print(f"\n  {'策略':<24}  {'月平均α':>7}  {'月中位α':>7}  {'α>0月%':>7}  {'停損率':>7}  {'勝率vs A':>8}")
print(f"  {'─'*70}")

base_alphas = np.array(results["A 標準 -15%/-25%"][0])

for name, (alphas, stop_rate) in results.items():
    arr   = np.array(alphas)
    mean_a = np.mean(arr) * 100
    med_a  = np.median(arr) * 100
    pos_pct = np.mean(arr > 0) * 100
    vs_base = np.mean(arr > base_alphas[:len(arr)]) * 100 if name != "A 標準 -15%/-25%" else "—"
    vs_str  = f"{vs_base:>6.1f}%" if isinstance(vs_base, float) else f"{'—':>7}"
    print(f"  {name:<24}  {mean_a:>+6.2f}%  {med_a:>+6.2f}%  {pos_pct:>6.1f}%  "
          f"{stop_rate*100:>6.1f}%  {vs_str}")

# ── 波動率分層詳細分析（策略 C）──────────────────────────────────────────────
print(f"\n  {'─'*70}")
print("  策略 C（分層）各波動率組別停損頻率：")

for vc_label, vol_lo, vol_hi, fs, ts in [
    ("低波動 <35%",   0,       0.35,  -0.15, -0.25),
    ("中波動 35-60%", 0.35,    0.60,  -0.20, -0.30),
    ("高波動 >60%",   0.60,    9.99,  -0.25, -0.35),
]:
    sc, tc = 0, 0
    for ti in rebal_tis:
        scores = []
        for sym in symbols:
            m = mom_mixed_at(sym, ti)
            if m and m > 0:
                v = vol_at(sym, ti)
                if v is None:
                    continue
                if vol_lo <= v < vol_hi:
                    scores.append((sym, m, v))
        scores.sort(key=lambda x: -x[1])
        for sym, _, v in scores[:TOP_N]:
            p0 = price_at(sym, ti)
            if not p0:
                continue
            _, _, stop_hit = simulate_hold(sym, ti, fs, ts)
            tc += 1
            if stop_hit:
                sc += 1
    if tc > 0:
        print(f"    {vc_label:<14}  停損率 {sc/tc*100:.1f}%  (N={tc})")

print(f"\n  ⚠️  Survivorship bias：僅含 S&P500 現有成份股")
print(f"  ⚠️  不含交易成本、滑點")
