import pandas_ta as ta
from src.indicators import add_ma_indicators, add_rsi_indicators

def apply_ma_strategy(df):
    """
    簡單均線策略：當股價站上 20 日均線時買入，跌破時賣出
    """
    # 計算 20 日移動平均線 (MA)
    df['MA20'] = ta.sma(df['Close'], length=20)
    
    # 定義買賣信號 (1: 買入, -1: 賣出, 0: 持有/觀望)
    df['Signal'] = 0
    # 股價 > MA20 且 前一天 股價 <= MA20 (黃金交叉)
    df.loc[df['Close'] > df['MA20'], 'Signal'] = 1
    # 股價 < MA20 且 前一天 股價 >= MA20 (死亡交叉)
    df.loc[df['Close'] < df['MA20'], 'Signal'] = -1
    
    return df

def apply_double_factor_strategy(df, ma_length=60, rsi_length=14):
    """
    雙因子策略：MA(趨勢) + RSI(動能)
    輸出 Signal（事件）：
      1 = 買入事件（想在下一交易日開盤後買）
     -1 = 賣出事件（想在下一交易日開盤後賣/平倉）
      0 = 無動作
    """
    df = df.copy()

    # 1) 計算指標（你的 indicators 工具）
    df = add_ma_indicators(df, ma_length)
    df = add_rsi_indicators(df, rsi_length)

    ma_col = f"MA{ma_length}"
    if ma_col not in df.columns:
        raise KeyError(f"Missing MA column: {ma_col}. Available: {list(df.columns)[:20]} ...")

    if 'RSI' not in df.columns:
        raise KeyError("Missing RSI column. Check add_rsi_indicators output column name.")

    # 2) 條件
    buy_condition = (df['Close'] > df[ma_col]) & (df['RSI'] < 70)
    sell_condition = (df['Close'] < df[ma_col]) | (df['RSI'] > 85)

    # 3) 事件訊號
    df['Signal'] = 0
    df.loc[buy_condition, 'Signal'] = 1
    # 避免同一天買賣同時成立時被覆蓋（你也可以反過來讓賣優先）
    df.loc[sell_condition & ~buy_condition, 'Signal'] = -1

    return df
