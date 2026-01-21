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
