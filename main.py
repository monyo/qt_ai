from src.data_loader import fetch_stock_data
from src.strategy import apply_double_factor_strategy
from src.analytics import calculate_full_metrics
from src.visualizer import plot_result
import pandas as pd

def main():
    # --- 壓力測試設定區 ---
    # 測試 2022 通膨大回撤
    test_start = "2022-01-01"
    test_end = "2023-01-01"
    
    # 或者測試 2020 疫情崩盤
    # test_start = "2020-01-01"
    # test_end = "2020-07-01"
    
    targets = ["SPY", "QQQ", "NVDA", "GOOGL"]
    # --------------------

    print(f"🕵️ 歷史壓力測試區間: {test_start} 至 {test_end}")
    print(f"{'Symbol':<8} | {'Market%':>10} | {'Strategy%':>10} | {'MDD%':>8} | {'Win%':>7}")
    print("-" * 65)

    for symbol in targets:
        # 使用自定義日期抓取數據
        df = fetch_stock_data(symbol, start=test_start, end=test_end)
        
        if df.empty:
            print(f"無法取得 {symbol} 的數據")
            continue
        
        # 1. 應用策略
        df = apply_double_factor_strategy(df)
        
        # 2. 使用重構後的分析中樞
        df, metrics = calculate_full_metrics(df)
        
        # 3. 顯示結果
        print(f"{symbol:<8} | {metrics['Market%']:>10.2f}% | {metrics['Return%']:>10.2f}% | {metrics['MDD%']:>8.2f}% | {metrics['WinRate%']:>7.2f}%")
        
        # 4. 畫圖讓你直觀看避險效果
        plot_result(df, f"{symbol}_StressTest")

if __name__ == "__main__":
    main()
