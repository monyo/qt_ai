"""趨勢狀態指標回測

驗證「趨勢狀態」（從40日低點反彈% + 距40日高點%）是否對未來報酬有預測力。
特別關注：當 1Y Alpha 為負但趨勢轉強時，是否應該持有而非出場？

方法：
- S&P 500 全部成分股，3 年歷史數據
- 每月快照，計算趨勢狀態 → 測量未來 21 日報酬
- 分組比較：轉強 vs 轉弱 vs 盤整
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


def calculate_trend_state(closes_40d):
    """計算趨勢狀態指標

    Args:
        closes_40d: 最近 40 個交易日的收盤價 Series

    Returns:
        dict: {bounce_pct, from_high_pct, state}
        - state: "轉強", "轉弱", "盤整"
    """
    if len(closes_40d) < 40:
        return None

    current = closes_40d.iloc[-1]
    low_40d = closes_40d.min()
    high_40d = closes_40d.max()

    if low_40d == 0 or high_40d == 0:
        return None

    bounce_pct = (current / low_40d - 1) * 100
    from_high_pct = (current / high_40d - 1) * 100

    # 分類
    if bounce_pct > 20 and from_high_pct > -5:
        state = "轉強"
    elif from_high_pct < -15:
        state = "轉弱"
    else:
        state = "盤整"

    return {
        "bounce_pct": round(bounce_pct, 1),
        "from_high_pct": round(from_high_pct, 1),
        "state": state,
    }


def calculate_1y_alpha(df, idx, benchmark_df):
    """計算某個時間點的 1Y Alpha"""
    date = df.index[idx]

    # 找到約 252 天前的位置
    start_idx = max(0, idx - 252)
    if idx - start_idx < 200:
        return None

    stock_ret = (df['Close'].iloc[idx] / df['Close'].iloc[start_idx] - 1) * 100

    # 找 benchmark 對應日期範圍
    bench_start = benchmark_df.index.searchsorted(df.index[start_idx])
    bench_end = benchmark_df.index.searchsorted(date)
    if bench_start >= len(benchmark_df) or bench_end >= len(benchmark_df):
        return None
    if bench_end <= bench_start:
        return None

    bench_ret = (benchmark_df['Close'].iloc[bench_end] / benchmark_df['Close'].iloc[bench_start] - 1) * 100
    return round(stock_ret - bench_ret, 1)


def run_backtest():
    # 取得 S&P 500 成分股
    print("正在取得 S&P 500 成分股清單...")
    tickers = get_sp500_tickers()
    print(f"共 {len(tickers)} 檔")

    # 取得 SPY benchmark
    print("正在取得 SPY 數據...")
    spy_df = fetch_stock_data("SPY", period="3y")
    if spy_df.empty:
        print("無法取得 SPY 數據")
        return

    # 批次取得所有股票數據
    print(f"正在取得 {len(tickers)} 檔股票的 3 年數據...")
    stock_data = {}

    def fetch_one(sym):
        try:
            df = fetch_stock_data(sym, period="3y")
            if df is not None and not df.empty and len(df) >= 300:
                return sym, df
        except Exception:
            pass
        return sym, None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  已完成 {done}/{len(tickers)}...")
            sym, df = future.result()
            if df is not None:
                stock_data[sym] = df

    print(f"成功取得 {len(stock_data)} 檔數據")

    # 設定快照日期：每月一次，過去 2 年（需要前 1 年算 alpha，後 21 天算 forward return）
    # 所以在數據的第 252~len-21 天範圍內，每 21 天取一次快照
    results = []

    for sym, df in stock_data.items():
        n = len(df)
        # 從第 292 天開始（252天 alpha + 40天趨勢），到倒數第 22 天
        start = 292
        end = n - 22

        if start >= end:
            continue

        # 每 21 天一個快照
        for idx in range(start, end, 21):
            # 1. 趨勢狀態
            closes_40d = df['Close'].iloc[idx-39:idx+1]
            ts = calculate_trend_state(closes_40d)
            if ts is None:
                continue

            # 2. 21日動能
            if idx < 21:
                continue
            momentum = (df['Close'].iloc[idx] / df['Close'].iloc[idx-21] - 1) * 100

            # 3. 1Y Alpha
            alpha = calculate_1y_alpha(df, idx, spy_df)
            if alpha is None:
                continue

            # 4. 未來 21 日報酬
            fwd_return = (df['Close'].iloc[idx + 21] / df['Close'].iloc[idx] - 1) * 100

            results.append({
                "symbol": sym,
                "date": df.index[idx],
                "close": df['Close'].iloc[idx],
                "momentum": round(momentum, 1),
                "alpha_1y": alpha,
                "bounce_pct": ts["bounce_pct"],
                "from_high_pct": ts["from_high_pct"],
                "trend_state": ts["state"],
                "fwd_21d_return": round(fwd_return, 2),
            })

    df_results = pd.DataFrame(results)
    print(f"\n總樣本數: {len(df_results)}")

    # === 分析 1: 趨勢狀態 vs 未來報酬 ===
    print("\n" + "=" * 60)
    print("  分析 1：趨勢狀態 vs 未來 21 日報酬")
    print("=" * 60)

    for state in ["轉強", "盤整", "轉弱"]:
        group = df_results[df_results["trend_state"] == state]
        if len(group) == 0:
            continue
        avg_ret = group["fwd_21d_return"].mean()
        median_ret = group["fwd_21d_return"].median()
        win_rate = (group["fwd_21d_return"] > 0).mean() * 100
        print(f"\n  [{state}] 樣本: {len(group)}")
        print(f"    平均未來報酬: {avg_ret:+.2f}%")
        print(f"    中位數報酬:   {median_ret:+.2f}%")
        print(f"    勝率:         {win_rate:.1f}%")

    # === 分析 2: 1Y Alpha 為負時，趨勢狀態是否有區分力？ ===
    print("\n" + "=" * 60)
    print("  分析 2：1Y Alpha 為負時，趨勢狀態的區分力")
    print("  （這是 TPL 的情境：alpha 差但趨勢轉強）")
    print("=" * 60)

    neg_alpha = df_results[df_results["alpha_1y"] < 0]
    print(f"\n  1Y Alpha < 0 的樣本: {len(neg_alpha)}")

    for state in ["轉強", "盤整", "轉弱"]:
        group = neg_alpha[neg_alpha["trend_state"] == state]
        if len(group) == 0:
            continue
        avg_ret = group["fwd_21d_return"].mean()
        median_ret = group["fwd_21d_return"].median()
        win_rate = (group["fwd_21d_return"] > 0).mean() * 100
        print(f"\n  [Alpha<0 + {state}] 樣本: {len(group)}")
        print(f"    平均未來報酬: {avg_ret:+.2f}%")
        print(f"    中位數報酬:   {median_ret:+.2f}%")
        print(f"    勝率:         {win_rate:.1f}%")

    # === 分析 3: 動能為負時，趨勢狀態是否有區分力？ ===
    print("\n" + "=" * 60)
    print("  分析 3：動能為負時，趨勢狀態的區分力")
    print("  （這是 UEC 的情境：動能差但可能在 V 轉）")
    print("=" * 60)

    neg_mom = df_results[df_results["momentum"] < 0]
    print(f"\n  動能 < 0 的樣本: {len(neg_mom)}")

    for state in ["轉強", "盤整", "轉弱"]:
        group = neg_mom[neg_mom["trend_state"] == state]
        if len(group) == 0:
            continue
        avg_ret = group["fwd_21d_return"].mean()
        median_ret = group["fwd_21d_return"].median()
        win_rate = (group["fwd_21d_return"] > 0).mean() * 100
        print(f"\n  [動能<0 + {state}] 樣本: {len(group)}")
        print(f"    平均未來報酬: {avg_ret:+.2f}%")
        print(f"    中位數報酬:   {median_ret:+.2f}%")
        print(f"    勝率:         {win_rate:.1f}%")

    # === 分析 4: 倒V（動能正但趨勢轉弱）===
    print("\n" + "=" * 60)
    print("  分析 4：動能正但趨勢轉弱（倒 V 見頂風險）")
    print("  （這是 TSLA 的情境：favorite 保護但技術面走壞）")
    print("=" * 60)

    pos_mom_weak = df_results[(df_results["momentum"] > 0) & (df_results["trend_state"] == "轉弱")]
    pos_mom_strong = df_results[(df_results["momentum"] > 0) & (df_results["trend_state"] == "轉強")]

    for label, group in [("動能>0 + 轉弱", pos_mom_weak), ("動能>0 + 轉強", pos_mom_strong)]:
        if len(group) == 0:
            continue
        avg_ret = group["fwd_21d_return"].mean()
        median_ret = group["fwd_21d_return"].median()
        win_rate = (group["fwd_21d_return"] > 0).mean() * 100
        print(f"\n  [{label}] 樣本: {len(group)}")
        print(f"    平均未來報酬: {avg_ret:+.2f}%")
        print(f"    中位數報酬:   {median_ret:+.2f}%")
        print(f"    勝率:         {win_rate:.1f}%")

    # === 總結 ===
    print("\n" + "=" * 60)
    print("  總結")
    print("=" * 60)

    # 計算趨勢狀態的增量價值
    all_avg = df_results["fwd_21d_return"].mean()
    strong_avg = df_results[df_results["trend_state"] == "轉強"]["fwd_21d_return"].mean()
    weak_avg = df_results[df_results["trend_state"] == "轉弱"]["fwd_21d_return"].mean()

    print(f"\n  全體平均未來報酬: {all_avg:+.2f}%")
    print(f"  轉強組平均:       {strong_avg:+.2f}%  (差異: {strong_avg - all_avg:+.2f}%)")
    print(f"  轉弱組平均:       {weak_avg:+.2f}%  (差異: {weak_avg - all_avg:+.2f}%)")
    print(f"  轉強 vs 轉弱差距: {strong_avg - weak_avg:+.2f}%")

    spread = strong_avg - weak_avg
    if spread > 1.0:
        print(f"\n  ✅ 趨勢狀態指標有顯著區分力（月差 {spread:+.2f}%），建議納入系統")
    elif spread > 0.3:
        print(f"\n  🟡 趨勢狀態指標有一定區分力（月差 {spread:+.2f}%），可考慮作為輔助參考")
    else:
        print(f"\n  ❌ 趨勢狀態指標區分力不足（月差 {spread:+.2f}%），不建議納入系統")


if __name__ == "__main__":
    run_backtest()
