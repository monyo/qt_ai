import pandas as pd

def run_backtest(df):
    """
    負責模擬買賣過程並產出績效
    一站式計算所有績效與風險指標
    輸入：帶有 Signal 的 DataFrame
    輸出：帶有累積報酬率欄位的 DataFrame, 以及指標字典
    """
    # 1. 基礎報酬計算
    df['Daily_Return'] = df['Close'].pct_change()
    df['Position'] = df['Signal'].replace(0, float('nan')).ffill().shift(1).fillna(0)
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']

    # 2. 累積報酬率 (用於畫圖)
    df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod()
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()

    # 3. 計算最終指標
    final_return = (df['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
    market_return = (df['Cumulative_Market_Return'].iloc[-1] - 1) * 100

    # 4. 最大回撤 (MDD)
    peak = df['Cumulative_Strategy_Return'].cummax()
    drawdown = (df['Cumulative_Strategy_Return'] - peak) / peak
    max_drawdown = drawdown.min() * 100

    # 5. 勝率 (當天有持倉且賺錢的比例)
    active_days = df[df['Position'] != 0]
    win_rate = (active_days['Strategy_Return'] > 0).sum() / len(active_days) * 100 if len(active_days) > 0 else 0

    metrics = {
        "Return%": round(final_return, 2),
        "Market%": round(market_return, 2),
        "WinRate%": round(win_rate, 2),
        "MDD%": round(max_drawdown, 2)
    }

    return df, metrics
