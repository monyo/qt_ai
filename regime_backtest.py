#!/usr/bin/env python3
"""市場體制回測工具

比較以下策略在持倉標的 10 年歷史中的表現：
- Buy & Hold（基準）
- Fixed-15%（現有系統）
- Regime Only（SPY > MA200 才持倉，否則現金）
- Regime + Fixed-15%（SPY 進熊市 OR 跌破成本 -15% → 出場）

市場體制定義（SPY 200日均線）：
- BULL：SPY 收盤 > MA200 → 正常持倉
- BEAR：SPY 收盤 < MA200 → 出場持現金，等到回 BULL 再重新進場

Usage:
    python regime_backtest.py
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

PORTFOLIO_PATH = "data/portfolio.json"
PERIOD = "10y"
FIXED_COOLDOWN = 5   # 固定停損後冷卻天數

STRATEGIES = [
    {"name": "Buy & Hold",        "fixed": None,  "regime": False},
    {"name": "Fixed-15%",         "fixed": -0.15, "regime": False},
    {"name": "Regime Only",       "fixed": None,  "regime": True},
    {"name": "Regime+Fixed-15%",  "fixed": -0.15, "regime": True},
]


def load_symbols():
    with open(PORTFOLIO_PATH) as f:
        data = json.load(f)
    return [s for s, p in data.get("positions", {}).items() if not p.get("core", False)]


def fetch_data(symbol):
    try:
        df = yf.Ticker(symbol).history(period=PERIOD)
        if df.empty or len(df) < 500:
            return None
        return df[["Close"]].copy()
    except Exception:
        return None


def fetch_spy_regime():
    """下載 SPY，計算 MA200，產出每日體制旗標（True = BULL）"""
    spy = yf.Ticker("SPY").history(period=PERIOD)
    if spy.empty:
        return None
    spy = spy[["Close"]].copy()
    spy["MA200"] = spy["Close"].rolling(200).mean()
    spy["is_bull"] = spy["Close"] > spy["MA200"]
    # MA200 計算前 200 天為 NaN → 視為 BULL（無訊號不出場）
    spy["is_bull"] = spy["is_bull"].fillna(True)
    return spy


def align_regime(spy_regime, stock_df):
    """將 SPY 體制訊號對齊至個股交易日（向前填充）

    Returns:
        numpy bool array，長度與 stock_df 相同
    """
    # 去掉 timezone，統一為 date level
    spy_series = spy_regime["is_bull"].copy()
    spy_series.index = spy_series.index.normalize().tz_localize(None)

    stock_dates = stock_df.index.normalize().tz_localize(None)

    aligned = spy_series.reindex(stock_dates, method="ffill")
    # 仍有 NaN（stock 比 SPY 早開始）→ 視為 BULL
    return aligned.fillna(True).values.astype(bool)


def simulate_strategy(df, regime_signal, fixed_threshold, use_regime):
    """模擬完整 10Y 資產曲線（含重新進場）"""
    prices = df["Close"].values
    n = len(prices)

    # B&H 快速路徑
    if fixed_threshold is None and not use_regime:
        cum = prices / prices[0]
        peak = np.maximum.accumulate(cum)
        mdd = ((cum - peak) / peak).min() * 100
        ret = (cum[-1] - 1) * 100
        cagr = (cum[-1] ** (252 / (n - 1)) - 1) * 100
        return dict(Return=ret, CAGR=cagr, MDD=mdd, Trades=1, TimeIn=100.0)

    in_pos = True
    entry_px = prices[0]
    cooldown = 0
    trades = 1
    days_in = 0
    daily_ret = np.zeros(n - 1)

    for i in range(1, n):
        px = prices[i]
        r = px / prices[i - 1] - 1
        bull = bool(regime_signal[i]) if use_regime else True

        if cooldown > 0:
            cooldown -= 1

        if in_pos:
            # 體制出場
            exit_now = use_regime and not bull
            # 固定停損
            if not exit_now and fixed_threshold is not None:
                if (px - entry_px) / entry_px <= fixed_threshold:
                    exit_now = True
            if exit_now:
                in_pos = False
                cooldown = FIXED_COOLDOWN

        elif cooldown == 0 and (not use_regime or bull):
            in_pos = True
            entry_px = px
            trades += 1

        daily_ret[i - 1] = r if in_pos else 0.0
        if in_pos:
            days_in += 1

    cum = np.cumprod(1 + daily_ret)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    ret = (cum[-1] - 1) * 100
    cagr = (cum[-1] ** (252 / (n - 1)) - 1) * 100
    time_in = days_in / (n - 1) * 100

    return dict(Return=round(ret, 2), CAGR=round(cagr, 2), MDD=round(mdd, 2),
                Trades=trades, TimeIn=round(time_in, 1))


def bear_periods(spy_regime):
    """找出 SPY 主要熊市期間（連續在 MA200 以下超過 10 天）"""
    is_bull = spy_regime["is_bull"]
    periods = []
    start = None
    for dt, bull in is_bull.items():
        if not bull and start is None:
            start = dt
        elif bull and start is not None:
            dur = (dt - start).days
            if dur > 10:
                periods.append((start.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d"), dur))
            start = None
    return periods


def run_backtest():
    symbols = load_symbols()
    print(f"\n=== 市場體制回測 | 期間: {PERIOD} | 標的: {len(symbols)} 支 | {date.today()} ===\n")

    print("  下載 SPY 體制訊號...", end=" ", flush=True)
    spy = fetch_spy_regime()
    if spy is None:
        print("失敗，Regime 策略無法執行")
        return

    bear_days = (~spy["is_bull"]).sum()
    total_days = len(spy)
    print(f"完成  熊市期間: {bear_days}/{total_days} 天 ({bear_days/total_days*100:.1f}%)")

    print("\n  主要熊市期間（SPY < MA200 > 10 天）：")
    for s, e, d in bear_periods(spy):
        print(f"    {s} → {e}（{d} 天）")
    print()

    all_rows = []
    succeeded = []

    for sym in symbols:
        print(f"  分析 {sym}...", end=" ", flush=True)
        df = fetch_data(sym)
        if df is None:
            print("數據不足，跳過")
            continue

        reg = align_regime(spy, df)

        for s in STRATEGIES:
            try:
                m = simulate_strategy(df, reg, s["fixed"], s["regime"])
                all_rows.append({"Symbol": sym, "Strategy": s["name"], **m})
            except Exception as e:
                all_rows.append({"Symbol": sym, "Strategy": s["name"],
                                 "Return": float("nan"), "CAGR": float("nan"),
                                 "MDD": float("nan"), "Trades": 0, "TimeIn": float("nan")})

        print(f"完成（{len(df)} 天）")
        succeeded.append(sym)

    if not all_rows:
        print("\n沒有成功分析任何標的。")
        return

    df_res = pd.DataFrame(all_rows)

    # ── 個股詳細 ──────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print("  個股詳細比較")
    print(f"{'='*95}")
    for sym in succeeded:
        sub = df_res[df_res["Symbol"] == sym].copy()
        print(f"\n--- {sym} ---")
        print(sub[["Strategy", "Return", "CAGR", "MDD", "Trades", "TimeIn"]].to_string(index=False))

    # ── 策略總結 ───────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print("  策略總結（所有標的平均）")
    print(f"{'='*95}\n")

    summ = df_res.groupby("Strategy").agg(
        avg_ret=("Return", "mean"),
        avg_cagr=("CAGR", "mean"),
        avg_mdd=("MDD", "mean"),
        avg_trades=("Trades", "mean"),
        avg_timein=("TimeIn", "mean"),
    ).round(2)

    order = [s["name"] for s in STRATEGIES]
    summ = summ.reindex(order)

    bh_cagr = summ.loc["Buy & Hold", "avg_cagr"]
    summ["vs_BH"] = (summ["avg_cagr"] - bh_cagr).round(2)
    summ["Calmar"] = (summ["avg_cagr"] / summ["avg_mdd"].abs().replace(0, 0.01)).round(3)
    summ.columns = ["平均Return%", "平均CAGR%", "平均MDD%", "平均交易次數", "市場參與率%", "vs B&H", "Calmar比"]
    print(summ.to_string())

    # ── 核心比較：Fixed-15% vs Regime+Fixed-15% ───────────────────
    print(f"\n{'='*95}")
    print("  核心比較：現有系統 vs 加入體制偵測")
    print(f"{'='*95}\n")

    cur = summ.loc["Fixed-15%"]
    enh = summ.loc["Regime+Fixed-15%"]

    print(f"  {'':30} {'CAGR%':>8} {'MDD%':>8} {'市場參與':>10} {'Calmar':>8}")
    print(f"  {'Fixed-15%（現有系統）':30} {cur['平均CAGR%']:>8.1f} {cur['平均MDD%']:>8.1f} {cur['市場參與率%']:>10.1f} {cur['Calmar比']:>8.3f}")
    print(f"  {'Regime+Fixed-15%（加強版）':30} {enh['平均CAGR%']:>8.1f} {enh['平均MDD%']:>8.1f} {enh['市場參與率%']:>10.1f} {enh['Calmar比']:>8.3f}")

    dcagr = enh["平均CAGR%"] - cur["平均CAGR%"]
    dmdd  = enh["平均MDD%"]  - cur["平均MDD%"]
    dcal  = enh["Calmar比"]  - cur["Calmar比"]

    print(f"\n  加入體制偵測的影響：CAGR {dcagr:+.1f}%，MDD {dmdd:+.1f}%，Calmar {dcal:+.3f}")

    if enh["Calmar比"] > cur["Calmar比"]:
        print("\n  ✓ 加入市場體制偵測可改善風險調整報酬（Calmar 更高）")
        print(f"    代價：每年少賺 {abs(dcagr):.1f}%，換來 MDD 改善 {abs(dmdd):.1f}%")
    else:
        print("\n  ✗ 加入市場體制偵測未改善風險調整報酬")
        print("    現有 Fixed-15% 已夠用，不需加入體制濾網")

    csv_path = f"data/backtest_regime_{date.today().strftime('%Y%m%d')}.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n詳細結果已儲存至: {csv_path}")


if __name__ == "__main__":
    run_backtest()
