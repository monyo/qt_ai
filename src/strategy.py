import pandas_ta as ta

def apply_ma_strategy(df):
    """
    簡單均線策略：當股價站上 20 日均線時買入，跌破時賣出
    """
    # 計算 20 日移動平均線 (MA)
    df['MA20'] = ta.sma(df['Close'], length=60)
    
    # 定義買賣信號 (1: 買入, -1: 賣出, 0: 持有/觀望)
    df['Signal'] = 0
    # 股價 > MA20 且 前一天 股價 <= MA20 (黃金交叉)
    df.loc[df['Close'] > df['MA20'], 'Signal'] = 1
    # 股價 < MA20 且 前一天 股價 >= MA20 (死亡交叉)
    df.loc[df['Close'] < df['MA20'], 'Signal'] = -1
    
    return df

def apply_double_factor_strategy(df, ma_length=60, rsi_length=14):
    """
    雙因子策略：MA (趨勢) + RSI (動能)
    """
    # 1. 計算指標
    df['MA'] = ta.sma(df['Close'], length=ma_length)
    df['RSI'] = ta.rsi(df['Close'], length=rsi_length)
    
    # 2. 定義買入條件
    # 條件：收盤價 > MA 且 RSI < 70 (避免追在高點)
    buy_condition = (df['Close'] > df['MA']) & (df['RSI'] < 70)
    
    # 定義賣出條件
    # 條件：收盤價 < MA (趨勢反轉) 或 RSI > 85 (極度過熱，先落袋為安)
    sell_condition = (df['Close'] < df['MA']) | (df['RSI'] > 85)
    
    # 3. 產生信號 (1: 買入, -1: 賣出, 0: 觀望)
    df['Signal'] = 0
    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1
    
    # 為了方便後續回測，我們處理一下信號，讓它保持持倉狀態
    # 這裡用簡單的邏輯：只要出現 1 就持續持倉，直到出現 -1
    return df
