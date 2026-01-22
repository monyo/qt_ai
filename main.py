from src.data_loader import fetch_stock_data
from src.strategy import apply_double_factor_strategy
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

def calculate_metrics(df):
    """計算簡單績效指標"""
    df['Daily_Return'] = df['Close'].pct_change()
    # 模擬持倉：將信號填充 (例如 1 之後都是 1，直到出現 -1)
    # 先將 0 轉為 NaN，然後執行前向填充，最後將剩餘的 NaN 補回 0
    df['Position'] = df['Signal'].replace(0, float('nan')).ffill().shift(1).fillna(0)
    # 強制轉換回整數型別，避免 Pandas 的轉型警告
    df['Position'] = df['Position'].astype(int)
    # 強制將賣出訊號 (-1) 視為空手 (0)
    df['Position'] = df['Position'].apply(lambda x: 1 if x == 1 else 0)
    
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    
    cum_market = (1 + df['Daily_Return']).cumprod().iloc[-1]
    cum_strategy = (1 + df['Strategy_Return']).cumprod().iloc[-1]
    
    return (cum_market - 1) * 100, (cum_strategy - 1) * 100

def main():
    # 準備你的珍珠與對照組
    #targets = ["NVDA", "AAPL", "TSLA", "MSFT", "KO"]
    targets = ["AOS", "ACN", "AES"]
    results = []

    print(f"{'Symbol':<8} | {'Market %':<12} | {'Strategy %':<12} | {'Beat?':<6}")
    print("-" * 50)

    for symbol in targets:
        df = fetch_stock_data(symbol)
        if df.empty: continue
        
        # 應用雙因子策略
        df = apply_double_factor_strategy(df)
        
        # 計算績效
        mkt_ret, str_ret = calculate_metrics(df)
        
        beat = "YES" if str_ret > mkt_ret else "NO"
        print(f"{symbol:<8} | {mkt_ret:>11.2f}% | {str_ret:>11.2f}% | {beat:<6}")
        
        results.append({
            "Symbol": symbol, 
            "Market_Return": mkt_ret, 
            "Strategy_Return": str_ret
        })

if __name__ == "__main__":
    main()
