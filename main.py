from src.data_loader import fetch_stock_data
from src.strategy import apply_ma_strategy
from src.visualizer import plot_result
import pandas as pd

def calculate_performance(df):
    """
    計算策略報酬率
    """
    # 假設我們在信號出現的隔天以開盤價交易（最保守的估計）
    # 計算每日報酬率
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 計算策略報酬率 (如果 Signal 是 1，我們就賺/賠當天的 Daily_Return)
    # 我們將 Signal 往後位移一格 (shift)，模擬「今天看到信號，明天進場」
    df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)
    
    # 計算累積報酬率 (複利計算)
    df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod()
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    return df

def main():
    symbol = "NVDA"
    df = fetch_stock_data(symbol, period="3y")
    
    if df.empty: return

    # 執行策略 (現在你可以用 pandas-ta 了)
    df = apply_ma_strategy(df)
    
    # 計算績效
    df = calculate_performance(df)
    
    # 輸出結果
    print(f"\n--- {symbol} 三年回測報告 ---")
    market_final = (df['Cumulative_Market_Return'].iloc[-1] - 1) * 100
    strategy_final = (df['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
    
    print(f"市場總報酬率 (Buy & Hold): {market_final:.2f}%")
    print(f"策略總報酬率: {strategy_final:.2f}%")
    
    # 看看最後幾天的明細
    print("\n最近五日明細:")
    print(df[['Close', 'Signal', 'Cumulative_Strategy_Return']].tail())

    plot_result(df, symbol)

if __name__ == "__main__":
    main()
