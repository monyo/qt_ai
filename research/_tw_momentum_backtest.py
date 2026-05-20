"""
台股動能策略回測

測試動能策略移植至台股的效果，對比 0050.TW B&H。

策略：
  A（基準）：純動能 top5，等權重，忽略交易成本
  B（含稅）：A + 0.3% 證交稅（賣出課徵）
  C（含稅+停損）：B + Fixed -15% / Trailing -25% 停損

注意事項：
  - Survivorship bias：使用當前高流動性股票，已排除歷史下市標的
  - 資料來源：yfinance（.TW / .TWO 後綴）
  - 基準：0050.TW（元大台灣50 ETF）B&H

資料快取：data/_tw_bt_prices.pkl
"""
import os, sys, time, warnings
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import yfinance as yf

# ── 參數 ──────────────────────────────────────────────────────────────────
CACHE_PATH   = "data/_tw_bt_prices.pkl"
MOM_SHORT    = 21
MOM_LONG     = 252
TOP_N        = 5
REBAL_FREQ   = 21
FIXED_STOP   = 0.15
TRAIL_STOP   = 0.25
INIT_CASH    = 100_000.0
TX_TAX       = 0.003    # 0.3% 證交稅（賣出課徵）
MAX_STOCKS   = 150      # 只取成交量前 N 支
DOWNLOAD_BATCH = 15     # 每批下載數量
DOWNLOAD_DELAY = 6      # 批次間隔秒數

# ── 載入或下載資料 ─────────────────────────────────────────────────────────
if os.path.exists(CACHE_PATH):
    print(f"載入快取：{CACHE_PATH}")
    prices = pd.read_pickle(CACHE_PATH)
else:
    print("快取不存在，開始下載台股歷史價格...")
    print(f"（預計需要 5-10 分鐘，請耐心等候）\n")

    with open("data/tw_liquid_stocks.json") as f:
        tw_data = json.load(f)
    stocks = tw_data["stocks"][:MAX_STOCKS]
    tickers = [s["symbol"] for s in stocks]
    print(f"目標：{len(tickers)} 支，分批下載（每批 {DOWNLOAD_BATCH} 支，間隔 {DOWNLOAD_DELAY} 秒）")

    all_close = {}
    failed = []
    for i in range(0, len(tickers), DOWNLOAD_BATCH):
        batch = tickers[i:i+DOWNLOAD_BATCH]
        batch_num = i // DOWNLOAD_BATCH + 1
        total_batches = (len(tickers) + DOWNLOAD_BATCH - 1) // DOWNLOAD_BATCH
        print(f"  批次 {batch_num}/{total_batches}：{batch[0]} ~ {batch[-1]}", flush=True)
        try:
            raw = yf.download(batch, start="2020-01-01", end="2026-04-18",
                              auto_adjust=True, progress=False)
            if "Close" not in raw:
                failed.extend(batch)
                continue
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(name=batch[0])
            for sym in close.columns:
                s = close[sym].dropna()
                if len(s) > 200:
                    all_close[sym] = s
        except Exception as e:
            print(f"  ⚠ 批次失敗：{e}")
            failed.extend(batch)
        if i + DOWNLOAD_BATCH < len(tickers):
            time.sleep(DOWNLOAD_DELAY)

    if not all_close:
        print("❌ 所有下載失敗，請稍後重試")
        sys.exit(1)

    prices = pd.DataFrame(all_close)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index()

    print(f"\n下載完成：{prices.shape[1]} 支 × {prices.shape[0]} 交易日")
    if failed:
        print(f"失敗：{len(failed)} 支")
    pd.to_pickle(prices, CACHE_PATH)
    print(f"已快取至 {CACHE_PATH}")

prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
prices = prices.sort_index()

trading_days = prices.index
n_dates = len(trading_days)
symbols = list(prices.columns)
close_arr = prices.values.astype(float)
sym_idx = {s: i for i, s in enumerate(symbols)}
print(f"\n資料：{len(symbols)} 支 / {n_dates} 交易日（{trading_days[0].date()} ~ {trading_days[-1].date()}）")

START_TI = MOM_LONG + 5
rebal_tis = list(range(START_TI, n_dates - REBAL_FREQ - 5, REBAL_FREQ))
print(f"再平衡點：{len(rebal_tis)} 個\n")


# ── 下載 0050 基準 ─────────────────────────────────────────────────────────
print("下載 0050.TW 基準...")
try:
    tw50_raw = yf.download("0050.TW", start="2020-01-01", end="2026-04-18",
                           auto_adjust=True, progress=False)["Close"]
    if isinstance(tw50_raw, pd.DataFrame):
        tw50_raw = tw50_raw.iloc[:, 0]
    tw50_raw.index = pd.to_datetime(tw50_raw.index).tz_localize(None)
    print(f"0050.TW：{len(tw50_raw)} 筆")
except Exception as e:
    print(f"⚠ 0050 下載失敗：{e}")
    tw50_raw = None


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


def tw50_fwd(ti, offset=REBAL_FREQ):
    if tw50_raw is None:
        return None
    ref = trading_days[ti]
    idx = tw50_raw.index.searchsorted(ref)
    if idx + offset >= len(tw50_raw):
        return None
    return float(tw50_raw.iloc[idx + offset]) / float(tw50_raw.iloc[idx]) - 1


# ── 全組合模擬 ────────────────────────────────────────────────────────────
def run_portfolio(strategy):
    """
    strategy:
      "A" 純動能（無交易成本）
      "B" 含 0.3% 證交稅
      "C" 含稅 + 停損
    """
    cash = INIT_CASH
    positions = {}   # {sym: {shares, avg_price, peak}}
    port_values = []

    rebal_set = set(rebal_tis)

    for ti in range(START_TI, n_dates):
        # 每日評估總值
        pos_val = sum(
            pos["shares"] * (price_at(ti, sym) or pos["avg_price"])
            for sym, pos in positions.items()
        )
        port_values.append(cash + pos_val)

        if ti not in rebal_set:
            # 非再平衡日：只更新 peak（策略 C 停損用）
            if strategy == "C":
                for sym, pos in list(positions.items()):
                    px = price_at(ti, sym)
                    if px and px > pos["peak"]:
                        pos["peak"] = px
            continue

        # ── 策略 C：停損檢查 ──
        if strategy == "C":
            to_stop = []
            for sym, pos in list(positions.items()):
                px = price_at(ti, sym)
                if not px: continue
                if px > pos["peak"]:
                    pos["peak"] = px
                fixed_px = pos["avg_price"] * (1 - FIXED_STOP)
                trail_px = pos["peak"] * (1 - TRAIL_STOP)
                if px < max(fixed_px, trail_px):
                    to_stop.append(sym)
            for sym in to_stop:
                px = price_at(ti, sym)
                if px:
                    proceeds = px * positions[sym]["shares"] * (1 - TX_TAX)
                    cash += proceeds
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
                    if strategy in ("B", "C"):
                        cash += px * positions[sym]["shares"] * (1 - TX_TAX)
                    else:
                        cash += px * positions[sym]["shares"]
                del positions[sym]

        # ── 買入/調整 top ──
        total_v = cash + sum(
            positions[s]["shares"] * (price_at(ti, s) or 0)
            for s in positions
        )
        target_each = total_v / TOP_N

        for sym in top:
            px = price_at(ti, sym)
            if not px: continue
            target_shares = target_each / px

            if sym in positions:
                diff = target_shares - positions[sym]["shares"]
                if diff < 0:   # 減倉 → 賣出課稅
                    proceeds = abs(diff) * px
                    if strategy in ("B", "C"):
                        proceeds *= (1 - TX_TAX)
                    cash += proceeds
                else:           # 加倉 → 買入不課稅
                    cash -= diff * px
                positions[sym]["shares"] = target_shares
                if px > positions[sym]["peak"]:
                    positions[sym]["peak"] = px
            else:
                if cash >= target_each * 0.9:
                    cash -= target_shares * px
                    positions[sym] = {
                        "shares": target_shares,
                        "avg_price": px,
                        "peak": px,
                    }

    return np.array(port_values)


# ── 0050 基準曲線 ─────────────────────────────────────────────────────────
def make_tw50_curve():
    if tw50_raw is None:
        return None
    ref_date = trading_days[START_TI]
    idx = tw50_raw.index.searchsorted(ref_date)
    vals = tw50_raw.iloc[idx:idx + (n_dates - START_TI)].values
    if len(vals) == 0:
        return None
    return INIT_CASH * vals / vals[0]


# ── 績效指標 ──────────────────────────────────────────────────────────────
def calc_metrics(vals, label):
    vals = np.array(vals, dtype=float)
    if len(vals) < 2:
        return {}
    n_days = len(vals)
    years = n_days / 252
    total_ret = vals[-1] / vals[0] - 1
    cagr = (vals[-1] / vals[0]) ** (1/years) - 1
    peak = np.maximum.accumulate(vals)
    mdd = ((vals - peak) / peak).min()
    daily_ret = np.diff(vals) / vals[:-1]
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    print(f"\n  {'─'*54}")
    print(f"  策略 {label}")
    print(f"  {'─'*54}")
    print(f"  最終值：${vals[-1]:>10,.0f}  CAGR：{cagr*100:+.2f}%  總報酬：{total_ret*100:+.0f}%")
    print(f"  MDD：{mdd*100:.2f}%  Calmar：{calmar:.3f}  Sharpe：{sharpe:.3f}")

    return {"label": label, "cagr": cagr, "mdd": mdd,
            "calmar": calmar, "sharpe": sharpe, "final": vals[-1]}


# ── 執行 ──────────────────────────────────────────────────────────────────
print("=" * 58)
print("  台股動能策略回測  2021-2025")
print("=" * 58)
print("  A：純動能 top5（無交易成本）")
print("  B：純動能 top5（含 0.3% 證交稅）")
print("  C：純動能 top5（含稅 + Fixed -15% / Trail -25% 停損）")
print("  基準：0050.TW B&H")

results = {}
for strat in ["A", "B", "C"]:
    print(f"\n  運行策略 {strat}...", flush=True)
    vals = run_portfolio(strat)
    results[strat] = calc_metrics(vals, strat)

tw50_curve = make_tw50_curve()
if tw50_curve is not None:
    results["0050"] = calc_metrics(tw50_curve, "0050 B&H")

# ── 比較表 ────────────────────────────────────────────────────────────────
print(f"\n\n  {'═'*58}")
print(f"  {'策略':<10} {'CAGR':>8} {'MDD':>8} {'Calmar':>8} {'Sharpe':>8}")
print(f"  {'═'*58}")
for label in ["A", "B", "C", "0050"]:
    r = results.get(label)
    if r:
        print(f"  {r['label']:<10} {r['cagr']*100:>+7.2f}% {r['mdd']*100:>+7.2f}%"
              f" {r['calmar']:>8.3f} {r['sharpe']:>8.3f}")
print(f"  {'═'*58}")

# ── 交易稅影響 ────────────────────────────────────────────────────────────
if "A" in results and "B" in results:
    tax_drag = results["A"]["cagr"] - results["B"]["cagr"]
    print(f"\n  0.3% 證交稅年化拖累：{tax_drag*100:.2f}% CAGR")

if "B" in results and "0050" in results:
    alpha = results["B"]["cagr"] - results["0050"]["cagr"]
    print(f"  動能策略（含稅）vs 0050 年化 alpha：{alpha*100:+.2f}%")

print(f"\n  ※ Survivorship bias 注意：使用現有高流動性股票，歷史下市標的已排除")
