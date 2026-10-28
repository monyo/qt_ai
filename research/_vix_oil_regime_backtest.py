"""
VIX+油價 Regime 過濾回測

現行 src/market_environment.py 用 VIX + WTI(USO) 21日報酬分成四種體制
（平靜牛市 / 健康風險偏好 / 恐慌下跌 / 滯脹恐慌），但目前只是顯示用文字說明，
沒有接到任何 ADD 決策邏輯。本回測驗證：把這個 regime label 接進選股數量，
是否真的能提升動能選股的遠期報酬 / alpha。

四種體制定義（與 src/market_environment.py._interpret 完全一致）：
  健康風險偏好：VIX 不高 + 油漲 ≥5%     → 系統註解：「動能有效」
  平靜牛市    ：VIX 不高 + 油穩/跌      → 系統註解：「個股 alpha 偏低，需更嚴格」
  恐慌下跌    ：VIX 高   + 油穩/跌      → 系統註解：「訊號可靠性下降，只做最高信心ADD」
  滯脹恐慌    ：VIX 高   + 油漲         → 系統註解：「動能轉能源，alpha 偏高」

測試四種策略：
  A（基準）    ：top5 mom_mixed，不管 regime（現行系統實際行為）
  B（比例縮放）：健康=5 / 平靜=3 / 恐慌下跌=2 / 滯脹恐慌=3
  C（恐慌暫停）：恐慌下跌時空手（現金），其餘 top5
  D（只做順風）：只有健康風險偏好時進場 top5，其餘全部空手

資料：S&P500 2019-2026（沿用 data/_protection_bt_prices.pkl），VIX+USO 現抓
月度再平衡（每月第一個交易日）
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf

from src.market_environment import _interpret

CACHE_PATH = "data/_protection_bt_prices.pkl"
MOM_SHORT  = 21
MOM_LONG   = 252
TOP_N      = 5

# ── 載入股價 ──────────────────────────────────────────────────────────
if not os.path.exists(CACHE_PATH):
    print(f"找不到快取 {CACHE_PATH}")
    raise SystemExit(1)

print(f"讀取快取：{CACHE_PATH}")
prices = pd.read_pickle(CACHE_PATH)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_convert(None)
trading_days = prices.index.sort_values()
print(f"股價資料：{prices.shape[1]} 檔 / {prices.shape[0]} 交易日 "
      f"({trading_days.min().date()} ~ {trading_days.max().date()})\n")

# ── 抓 VIX + USO + SPY（現抓，避免用到舊快取）──────────────────────────
print("下載 VIX / USO / SPY...")
raw = yf.download(["^VIX", "USO", "SPY"], start="2019-01-01",
                   end=str((trading_days.max() + pd.Timedelta(days=1)).date()),
                   auto_adjust=True, progress=False)
close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
close.index = pd.to_datetime(close.index).tz_localize(None)
close = close.ffill()

vix = close["^VIX"]
oil = close["USO"]
spy = close["SPY"]

vix_ma63  = vix.rolling(63, min_periods=20).mean()
vix_ratio = vix / vix_ma63
oil_ret21 = oil.pct_change(21) * 100

print(f"VIX/油價資料：{vix.dropna().shape[0]} 交易日\n")


# ── 工具函式 ──────────────────────────────────────────────────────────
def tidx(d) -> int:
    return int(trading_days.searchsorted(pd.Timestamp(d)))

def price_at(sym, ti):
    if ti < 0 or ti >= len(trading_days):
        return None
    v = prices.iloc[ti].get(sym)
    return float(v) if v is not None and pd.notna(v) else None

def mom_mixed_at(sym, ti):
    if ti < MOM_LONG:
        return None
    p0, p21, p252 = price_at(sym, ti), price_at(sym, ti - MOM_SHORT), price_at(sym, ti - MOM_LONG)
    if not (p0 and p21 and p252 and p21 > 0 and p252 > 0):
        return None
    return 0.5 * (p0/p21 - 1)*100 + 0.5 * (p0/p252 - 1)*100

def fwd_return_sym(sym, ti, offset):
    p0, p1 = price_at(sym, ti), price_at(sym, ti + offset)
    return (p1/p0 - 1)*100 if (p0 and p1 and p0 > 0) else None

def spy_fwd(ref_date, offset):
    s = spy.dropna()
    ti = s.index.searchsorted(pd.Timestamp(ref_date))
    if ti + offset >= len(s):
        return None
    return (float(s.iloc[ti+offset]) / float(s.iloc[ti]) - 1) * 100

def regime_at(ref_date):
    """回傳 (label, n_days_lookback_ok)"""
    idx = vix.index.searchsorted(pd.Timestamp(ref_date))
    if idx >= len(vix):
        idx = len(vix) - 1
    v, vr, o = vix.iloc[idx], vix_ratio.iloc[idx], oil_ret21.iloc[idx]
    if pd.isna(v) or pd.isna(vr) or pd.isna(o):
        return None
    label, _, _ = _interpret(float(v), float(vr), float(o))
    return label


# ── 主回測循環 ─────────────────────────────────────────────────────────
rebalance_dates = pd.bdate_range("2020-06-01", str(trading_days.max().date()), freq="BMS")
results = {k: [] for k in ["A", "B", "C", "D"]}
regime_log = []

print("回測中...")
for ref in rebalance_dates:
    ti = tidx(str(ref.date()))
    if ti < MOM_LONG or ti >= len(trading_days):
        continue

    regime = regime_at(ref.date())
    if regime is None:
        continue
    regime_log.append({"date": str(ref.date()), "regime": regime})

    pool = []
    for sym in prices.columns:
        m = mom_mixed_at(sym, ti)
        if m is None or m <= 0:
            continue
        pool.append({"sym": sym, "mom": m})
    if len(pool) < TOP_N:
        continue
    df_pool = pd.DataFrame(pool).nlargest(20, "mom")

    n_map_B = {"健康風險偏好": 5, "平靜牛市": 3, "恐慌下跌": 2, "滯脹恐慌": 3}
    n_B = n_map_B.get(regime, TOP_N)
    n_C = 0 if regime == "恐慌下跌" else TOP_N
    n_D = TOP_N if regime == "健康風險偏好" else 0

    picks = {
        "A": df_pool.head(TOP_N)["sym"].tolist(),
        "B": df_pool.head(n_B)["sym"].tolist(),
        "C": df_pool.head(n_C)["sym"].tolist() if n_C > 0 else [],
        "D": df_pool.head(n_D)["sym"].tolist() if n_D > 0 else [],
    }

    spy_1m = spy_fwd(ref.date(), 21)
    spy_3m = spy_fwd(ref.date(), 63)

    for key, syms in picks.items():
        if not syms:
            results[key].append({
                "date": str(ref.date()), "regime": regime, "picks": [],
                "avg_1m": 0.0, "avg_3m": 0.0,
                "alpha_1m": (0.0 - spy_1m) if spy_1m is not None else None,
                "alpha_3m": (0.0 - spy_3m) if spy_3m is not None else None,
                "n_picks": 0,
            })
            continue
        fwd_1m = [x for x in [fwd_return_sym(s, ti, 21) for s in syms] if x is not None]
        fwd_3m = [x for x in [fwd_return_sym(s, ti, 63) for s in syms] if x is not None]
        if not fwd_1m:
            continue
        avg_1m = np.mean(fwd_1m)
        avg_3m = np.mean(fwd_3m) if fwd_3m else None
        results[key].append({
            "date": str(ref.date()), "regime": regime, "picks": syms,
            "avg_1m": avg_1m, "avg_3m": avg_3m,
            "alpha_1m": avg_1m - spy_1m if spy_1m is not None else None,
            "alpha_3m": (avg_3m - spy_3m) if (avg_3m is not None and spy_3m is not None) else None,
            "n_picks": len(syms),
        })

print()
print("=" * 76)
print("  VIX+油價 Regime 回測  2020-06 ~ 今  S&P500 月度")
print("=" * 76)

df_rl = pd.DataFrame(regime_log)
print(f"\n  Regime 分布（{len(df_rl)} 個月）")
print(df_rl["regime"].value_counts().to_string())

desc = {
    "A": "A. 純動能（基準，現行系統實際行為）  top5，不管regime",
    "B": "B. 比例縮放     健康=5 平靜=3 恐慌下跌=2 滯脹=3",
    "C": "C. 恐慌暫停     恐慌下跌→空手，其餘top5",
    "D": "D. 只做順風     只有健康風險偏好→top5，其餘空手",
}

summary = {}
for key in ["A", "B", "C", "D"]:
    rows = results[key]
    if not rows:
        continue
    df = pd.DataFrame(rows)
    n = len(df)
    avg_1m = df["avg_1m"].mean()
    avg_3m = df["avg_3m"].dropna().mean()
    alpha_1m = df["alpha_1m"].dropna().mean()
    alpha_3m = df["alpha_3m"].dropna().mean()
    win_a = (df["alpha_1m"].dropna() > 0).sum() / df["alpha_1m"].dropna().shape[0] * 100
    avg_n = df["n_picks"].mean()
    # 簡易 CAGR/波動估計：用月報酬複利
    monthly_ret = df["avg_1m"] / 100
    cum = (1 + monthly_ret).cumprod()
    total_ret = (cum.iloc[-1] - 1) * 100 if len(cum) else None
    mdd = ((cum / cum.cummax()) - 1).min() * 100 if len(cum) else None
    summary[key] = {"alpha_1m": alpha_1m, "alpha_3m": alpha_3m,
                     "total_ret": total_ret, "mdd": mdd}
    print(f"\n  {desc[key]}")
    print(f"    平均 1M 報酬: {avg_1m:+.2f}%   3M: {avg_3m:+.2f}%   (n={n})")
    print(f"    vs SPY alpha: 1M {alpha_1m:+.2f}%   3M {alpha_3m:+.2f}%")
    print(f"    月跑贏SPY: {win_a:.0f}%   平均選股數: {avg_n:.1f}")
    print(f"    月報酬複利累積: {total_ret:+.1f}%   簡易月度MDD: {mdd:.1f}%")

print()
print("=" * 76)
print("  相對基準 A 的改善量")
print(f"  {'策略':<46} {'1M alpha 差':>12} {'3M alpha 差':>12} {'累積報酬差':>12}")
print("  " + "-" * 76)
for key in ["B", "C", "D"]:
    if key in summary and "A" in summary:
        d1 = summary[key]["alpha_1m"] - summary["A"]["alpha_1m"]
        d3 = summary[key]["alpha_3m"] - summary["A"]["alpha_3m"]
        dtr = (summary[key]["total_ret"] - summary["A"]["total_ret"]) if (summary[key]["total_ret"] is not None and summary["A"]["total_ret"] is not None) else None
        dtr_str = f"{dtr:+.1f}%" if dtr is not None else "N/A"
        print(f"  {desc[key]:<46} {d1:>+11.2f}%  {d3:>+11.2f}%  {dtr_str:>11}")

# ── Regime 分組：策略 A 在各 regime 下的實際表現 ────────────────────────
print()
print("=" * 76)
print("  Regime 分組分析（策略 A，按當月 regime 分組）")
print("  核心問題：系統的 regime_note 說法（健康>平靜>恐慌下跌）是否被數據驗證？")
print()

df_A = pd.DataFrame(results["A"])
print(f"  {'Regime':<14} {'1M報酬':>9} {'1M alpha':>10} {'3M alpha':>10}  n")
print("  " + "-" * 58)
for label in ["健康風險偏好", "平靜牛市", "恐慌下跌", "滯脹恐慌"]:
    sub = df_A[df_A["regime"] == label]
    if len(sub) == 0:
        print(f"  {label:<14}  (無樣本)")
        continue
    r1 = sub["avg_1m"].mean()
    a1 = sub["alpha_1m"].dropna().mean()
    a3 = sub["alpha_3m"].dropna().mean()
    print(f"  {label:<14} {r1:>+8.2f}%  {a1:>+9.2f}%  {a3:>+9.2f}%  {len(sub)}")
