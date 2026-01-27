import pandas_ta as ta

def add_ma_indicators(df, length=60):
    df[f'MA{length}'] = ta.sma(df['Close'], length=length)
    return df

def add_rsi_indicators(df, length=14):
    df['RSI'] = ta.rsi(df['Close'], length=length)
    return df
