"""動能回看週期比較回測

比較不同動能回看週期（21d, 63d, 126d, 252d）及混合策略對選股效果的影響。
模擬每月重新排名、買入前20名、等權重持有的策略。

學術參考：Jegadeesh & Titman (1993) 指出 6-12 個月動能因子最有效。
"""

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import numpy as np
from src.data_loader import fetch_stock_data, get_sp500_tickers
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

TOP_N = 20  # 每期持有檔數
REBALANCE_DAYS = 21  # 每 21 個交易日重新排名


def calc_momentum(df, loc, period):
    """計算單一標的在指定位置的動能"""
    if loc < period or loc >= len(df):
        return None
    lookback_loc = loc - period
    current_price = df['Close'].iloc[loc]
    past_price = df['Close'].iloc[lookback_loc]
    if past_price > 0:
        return (current_price / past_price - 1) * 100
    return None


def simulate_strategy(stock_data, spy_df, start_idx, score_fn, label):
    """模擬單一策略的月度再平衡

    Args:
        stock_data: {symbol: DataFrame}
        spy_df: SPY DataFrame
        start_idx: 起始索引
        score_fn: func(df, loc) -> float or None，計算排名分數
        label: 策略名稱

    Returns:
        dict with results
    """
    all_dates = spy_df.index
    n_dates = len(all_dates)

    monthly_returns = []
    monthly_turnover = []
    cumulative = 1.0
    peak = 1.0
    max_dd = 0
    prev_holdings = set()

    idx = start_idx
    while idx + REBALANCE_DAYS < n_dates:
        rebal_date = all_dates[idx]
        next_date = all_dates[min(idx + REBALANCE_DAYS, n_dates - 1)]

        # 計算每檔股票的分數
        scores = {}
        for sym, df in stock_data.items():
            try:
                loc = df.index.searchsorted(rebal_date)
                score = score_fn(df, loc)
                if score is not None:
                    scores[sym] = score
            except (IndexError, KeyError):
                continue

        if len(scores) < TOP_N:
            idx += REBALANCE_DAYS
            continue

        # 排名取前 TOP_N
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_stocks = [s[0] for s in sorted_scores[:TOP_N]]
        current_holdings = set(top_stocks)

        # 計算換手率
        if prev_holdings:
            turnover = len(current_holdings - prev_holdings)
            monthly_turnover.append(turnover)
        prev_holdings = current_holdings

        # 計算持有期報酬（等權重）
        period_returns = []
        for sym in top_stocks:
            df = stock_data[sym]
            try:
                loc_start = df.index.searchsorted(rebal_date)
                loc_end = df.index.searchsorted(next_date)
                if loc_start < len(df) and loc_end < len(df):
                    ret = (df['Close'].iloc[loc_end] / df['Close'].iloc[loc_start] - 1)
                    period_returns.append(ret)
            except (IndexError, KeyError):
                continue

        if period_returns:
            avg_return = np.mean(period_returns)
            monthly_returns.append(avg_return)
            cumulative *= (1 + avg_return)
            peak = max(peak, cumulative)
            dd = (cumulative - peak) / peak
            max_dd = min(max_dd, dd)

        idx += REBALANCE_DAYS

    n_months = len(monthly_returns)
    total_return = (cumulative - 1) * 100
    years = n_months * REBALANCE_DAYS / 252
    cagr = ((cumulative) ** (1 / years) - 1) * 100 if years > 0 else 0
    win_rate = sum(1 for r in monthly_returns if r > 0) / n_months * 100 if n_months > 0 else 0
    avg_turnover = np.mean(monthly_turnover) if monthly_turnover else 0

    return {
        "label": label,
        "total_return": total_return,
        "cagr": cagr,
        "mdd": max_dd * 100,
        "avg_turnover": avg_turnover,
        "win_rate": win_rate,
        "n_months": n_months,
    }


def run_backtest():
    # 取得 S&P 500 成分股
    print("正在取得 S&P 500 成分股清單...")
    tickers = get_sp500_tickers()
    print(f"共 {len(tickers)} 檔")

    # 批次取得所有股票 3 年數據
    print(f"正在取得 {len(tickers)} 檔股票 + SPY 的 3 年數據...")
    stock_data = {}
    all_symbols = tickers + ["SPY"]

    def fetch_one(sym):
        try:
            df = fetch_stock_data(sym, period="3y")
            if df is not None and not df.empty and len(df) >= 300:
                return sym, df
        except Exception:
            pass
        return sym, None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in all_symbols}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  已完成 {done}/{len(all_symbols)}...")
            sym, df = future.result()
            if df is not None:
                stock_data[sym] = df

    spy_df = stock_data.pop("SPY", None)
    if spy_df is None:
        print("無法取得 SPY 數據")
        return

    print(f"成功取得 {len(stock_data)} 檔數據")

    all_dates = spy_df.index
    n_dates = len(all_dates)
    start_idx = 260

    if start_idx >= n_dates - REBALANCE_DAYS:
        print("數據不足以進行回測")
        return

    # === 定義所有策略 ===
    strategies = []

    # 純週期策略
    for period in [21, 252]:
        p = period
        strategies.append({
            "label": f"純{p}天",
            "score_fn": lambda df, loc, _p=p: calc_momentum(df, loc, _p),
        })

    # 混合策略：不同比例的 21天 + 252天
    for short_w, long_w in [(70, 30), (50, 50), (30, 70)]:
        strategies.append({
            "label": f"混合{short_w}/{long_w}",
            "score_fn": lambda df, loc, _sw=short_w, _lw=long_w: (
                (lambda m21, m252:
                    (_sw / 100 * m21 + _lw / 100 * m252) if m21 is not None and m252 is not None else None
                )(calc_momentum(df, loc, 21), calc_momentum(df, loc, 252))
            ),
        })

    # === 執行回測 ===
    results = []
    for s in strategies:
        print(f"\n正在回測 {s['label']}...")
        r = simulate_strategy(stock_data, spy_df, start_idx, s["score_fn"], s["label"])
        results.append(r)

    # SPY 基準
    spy_monthly = []
    spy_cum = 1.0
    spy_peak = 1.0
    spy_mdd = 0
    idx = start_idx
    while idx + REBALANCE_DAYS < n_dates:
        loc_end = min(idx + REBALANCE_DAYS, n_dates - 1)
        ret = (spy_df['Close'].iloc[loc_end] / spy_df['Close'].iloc[idx] - 1)
        spy_monthly.append(ret)
        spy_cum *= (1 + ret)
        spy_peak = max(spy_peak, spy_cum)
        dd = (spy_cum - spy_peak) / spy_peak
        spy_mdd = min(spy_mdd, dd)
        idx += REBALANCE_DAYS

    spy_n = len(spy_monthly)
    spy_years = spy_n * REBALANCE_DAYS / 252
    spy_total = (spy_cum - 1) * 100
    spy_cagr = ((spy_cum) ** (1 / spy_years) - 1) * 100 if spy_years > 0 else 0
    spy_wr = sum(1 for r in spy_monthly if r > 0) / spy_n * 100 if spy_n > 0 else 0

    # === 印出結果 ===
    start_date = pd.Timestamp(all_dates[start_idx]).strftime('%Y-%m-%d')
    end_date = pd.Timestamp(all_dates[-1]).strftime('%Y-%m-%d')

    print("\n" + "=" * 85)
    print("  動能回看週期 + 混合策略比較")
    print(f"  回測期間: {start_date} ~ {end_date}")
    print(f"  持有檔數: {TOP_N}  |  再平衡: 每 {REBALANCE_DAYS} 交易日")
    print("=" * 85)

    print(f"\n  {'策略':>12}  {'累積報酬':>8}  {'CAGR':>6}  {'MDD':>7}  {'CAGR/MDD':>8}  {'月均換手':>8}  {'勝率':>6}  {'vs SPY':>8}")
    print(f"  {'-'*82}")

    for r in results:
        vs_spy = r["total_return"] - spy_total
        cagr_mdd = r["cagr"] / abs(r["mdd"]) if r["mdd"] != 0 else 0
        print(f"  {r['label']:>12}  {r['total_return']:>+7.1f}%  {r['cagr']:>+5.1f}%  {r['mdd']:>6.1f}%  {cagr_mdd:>7.2f}   {r['avg_turnover']:>5.1f}檔    {r['win_rate']:>5.1f}%  {vs_spy:>+7.1f}%")

    print(f"  {'SPY':>12}  {spy_total:>+7.1f}%  {spy_cagr:>+5.1f}%  {spy_mdd*100:>6.1f}%  {'--':>8}  {'0.0':>5}檔    {spy_wr:>5.1f}%  {'--':>8}")

    # === 分析 ===
    print("\n" + "=" * 85)
    print("  分析")
    print("=" * 85)

    best_return = max(results, key=lambda x: x["cagr"])
    best_risk_adj = max(results, key=lambda x: x["cagr"] / abs(x["mdd"]) if x["mdd"] != 0 else 0)
    lowest_turnover = min(results, key=lambda x: x["avg_turnover"])

    print(f"\n  最高報酬:      {best_return['label']} (CAGR {best_return['cagr']:+.1f}%)")
    print(f"  最佳風險調整:  {best_risk_adj['label']} (CAGR/MDD = {best_risk_adj['cagr']/abs(best_risk_adj['mdd']):.2f})")
    print(f"  最低換手率:    {lowest_turnover['label']} (月均 {lowest_turnover['avg_turnover']:.1f} 檔)")

    # 混合 vs 純策略比較
    pure_21 = next(r for r in results if r["label"] == "純21天")
    pure_252 = next(r for r in results if r["label"] == "純252天")

    print(f"\n  --- 混合策略 vs 純策略 ---")
    for r in results:
        if "混合" not in r["label"]:
            continue
        print(f"\n  {r['label']}:")
        print(f"    vs 純21天:  報酬 {r['total_return']-pure_21['total_return']:+.1f}%  MDD {r['mdd']-pure_21['mdd']:+.1f}%  換手 {r['avg_turnover']-pure_21['avg_turnover']:+.1f}檔")
        print(f"    vs 純252天: 報酬 {r['total_return']-pure_252['total_return']:+.1f}%  MDD {r['mdd']-pure_252['mdd']:+.1f}%  換手 {r['avg_turnover']-pure_252['avg_turnover']:+.1f}檔")


if __name__ == "__main__":
    run_backtest()
