import matplotlib.pyplot as plt

def plot_result(df, symbol):
    plt.figure(figsize=(12, 6))
    
    # 畫出市場累積報酬率
    plt.plot(df['Cumulative_Market_Return'], label='Market (Buy & Hold)', color='gray', alpha=0.5)
    
    # 畫出策略累積報酬率
    plt.plot(df['Cumulative_Strategy_Return'], label='MA20 Strategy', color='blue')
    
    plt.title(f"{symbol} Strategy vs Market Performance")
    plt.legend()
    plt.grid(True)
    
    # 儲存圖片
    plt.savefig('backtest_result.png')
    print("\n圖表已儲存為 backtest_result.png")
    plt.show()
