#!/usr/bin/env python3
"""停損策略比較工具

Usage:
    python stop_loss_compare.py NVDA SHOP UEC GOOG CVS TSLA MU LLY DASH ZG
"""
import argparse
import os
import pandas as pd
from datetime import date

from src.data_loader import fetch_stock_data
from src.stop_loss_backtester import compare_strategies_for_symbol


STRATEGIES = [
    {"name": "Buy & Hold", "type": None, "threshold": None},
    {"name": "Fixed -35%", "type": "fixed", "threshold": -0.35},
    {"name": "Fixed -20%", "type": "fixed", "threshold": -0.20},
    {"name": "Fixed -15%", "type": "fixed", "threshold": -0.15},
    {"name": "Trailing -15%", "type": "trailing", "threshold": -0.15},
    {"name": "Trailing -20%", "type": "trailing", "threshold": -0.20},
]


def run_comparison(symbols, period="3y"):
    """比較多檔股票在不同停損策略下的表現"""
    os.makedirs("data", exist_ok=True)

    all_results = []

    print(f"\n{'='*80}")
    print(f"  停損策略比較  |  測試期間: {period}  |  {date.today()}")
    print(f"{'='*80}\n")

    for symbol in symbols:
        symbol = symbol.upper().replace('.', '-')
        print(f"分析 {symbol}...", end=" ")

        try:
            df = fetch_stock_data(symbol, period=period)
            if df is None or df.empty or len(df) < 100:
                print("數據不足，跳過")
                continue

            results = compare_strategies_for_symbol(df, STRATEGIES)

            for r in results:
                r["Symbol"] = symbol
            all_results.extend(results)

            print("完成")

        except Exception as e:
            print(f"失敗: {e}")
            continue

    if not all_results:
        print("\n沒有成功分析任何標的。")
        return

    # 整理成 DataFrame
    df_results = pd.DataFrame(all_results)

    # 重新排列欄位順序
    cols = ["Symbol", "Strategy", "Return%", "Market%", "MDD%", "CAGR%", "Trade_Count", "Stop_Count"]
    df_results = df_results[cols]

    # === 輸出個股詳細結果 ===
    print(f"\n{'='*80}")
    print("  個股詳細比較")
    print(f"{'='*80}\n")

    for symbol in df_results["Symbol"].unique():
        print(f"\n--- {symbol} ---")
        symbol_df = df_results[df_results["Symbol"] == symbol].copy()
        symbol_df = symbol_df.drop(columns=["Symbol"])
        print(symbol_df.to_string(index=False))

    # === 輸出策略總結 ===
    print(f"\n{'='*80}")
    print("  策略總結（所有標的平均）")
    print(f"{'='*80}\n")

    summary = df_results.groupby("Strategy").agg({
        "Return%": "mean",
        "MDD%": "mean",
        "CAGR%": "mean",
        "Stop_Count": "sum",
    }).round(2)

    # 保持策略順序
    strategy_order = [s["name"] for s in STRATEGIES]
    summary = summary.reindex(strategy_order)

    print(summary.to_string())

    # === 找出最佳策略 ===
    print(f"\n{'='*80}")
    print("  建議")
    print(f"{'='*80}\n")

    best_return = summary["Return%"].idxmax()
    best_mdd = summary["MDD%"].idxmax()  # MDD 是負數，所以 max 是最小虧損
    best_risk_adj = (summary["Return%"] / abs(summary["MDD%"].replace(0, 0.01))).idxmax()

    print(f"  最高報酬策略: {best_return} (平均 {summary.loc[best_return, 'Return%']:.1f}%)")
    print(f"  最小回撤策略: {best_mdd} (平均 MDD {summary.loc[best_mdd, 'MDD%']:.1f}%)")
    print(f"  風險調整最佳: {best_risk_adj} (報酬/回撤比最高)")

    # === 儲存 CSV ===
    csv_path = f"data/stop_loss_compare_{date.today().strftime('%Y%m%d')}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n詳細結果已儲存至: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="比較不同停損策略的歷史表現")
    parser.add_argument("symbols", nargs="+", help="要測試的股票 symbol")
    parser.add_argument("--period", default="3y", help="測試期間 (default: 3y)")
    args = parser.parse_args()

    run_comparison(args.symbols, period=args.period)


if __name__ == "__main__":
    main()
