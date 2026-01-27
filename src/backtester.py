import pandas as pd

def run_backtest(df, cost_bps=0.0):
    """
    模擬 long-only 買賣並產出績效
    - Signal:  1=買入事件, -1=賣出事件, 0=無動作
    - Position: 0/1（隔天生效，避免 look-ahead）
    - cost_bps: 單邊成本（bps），例如 10 代表 0.10% = 0.001
    """
    df = df.copy()
    df = df.sort_index()

    # 1) 基礎報酬
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)

    # 2) long-only 狀態機：Signal(事件) -> Position(0/1 持倉狀態)
    signals = df['Signal'].fillna(0).astype(int).tolist()

    pos = 0
    positions = []
    entry_event = []
    exit_event = []

    for sig in signals:
        e_in = 0
        e_out = 0

        if sig == 1 and pos == 0:
            pos = 1
            e_in = 1
        elif sig == -1 and pos == 1:
            pos = 0
            e_out = 1

        positions.append(pos)
        entry_event.append(e_in)
        exit_event.append(e_out)

    # 事件（今天收盤才知道）
    df['Entry_Event'] = pd.Series(entry_event, index=df.index)
    df['Exit_Event']  = pd.Series(exit_event, index=df.index)

    # 隔天才持倉（假設 t 收盤產生訊號，t+1 才成交/持倉）
    df['Position'] = pd.Series(positions, index=df.index).shift(1).fillna(0)

    # 交易旗標（真正部位變動日）
    df['Trade_Flag'] = df['Position'].diff().abs().fillna(0)

    # 3) 策略報酬（先不含成本）
    df['Strategy_Return_gross'] = df['Daily_Return'] * df['Position']

    # 4) 扣成本（只在交易日扣，cost_bps 是單邊；這裡用部位變動日扣一次）
    cost_rate = cost_bps / 10000.0
    df['Cost'] = df['Trade_Flag'] * cost_rate
    df['Strategy_Return'] = df['Strategy_Return_gross'] - df['Cost']

    # 5) 累積報酬
    df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod()
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()

    # 6) 指標
    final_return = (df['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
    market_return = (df['Cumulative_Market_Return'].iloc[-1] - 1) * 100

    peak = df['Cumulative_Strategy_Return'].cummax()
    drawdown = (df['Cumulative_Strategy_Return'] - peak) / peak
    max_drawdown = drawdown.min() * 100

    active_days = df[df['Position'] != 0]
    win_rate = (active_days['Strategy_Return'] > 0).sum() / len(active_days) * 100 if len(active_days) > 0 else 0

    metrics = {
        "Return%": round(final_return, 2),
        "Market%": round(market_return, 2),
        "WinRate%": round(win_rate, 2),   # 這是「持倉日勝率」
        "MDD%": round(max_drawdown, 2),
    }

    return df, metrics