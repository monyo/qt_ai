import numpy as np
import pandas as pd


def backtest_stop_loss_strategy(
    df,
    stop_loss_type=None,
    threshold=-0.35,
    entry_signal=None,
    cooldown_days=1,
):
    """回測停損策略

    Args:
        df: DataFrame，需含 'Close' 欄位
        stop_loss_type: None (Buy & Hold), "fixed", or "trailing"
        threshold: 停損門檻，如 -0.35 表示 -35%
        entry_signal: Series 或 None。1=進場訊號，-1=出場訊號，0=無訊號
                      若為 None，則採用 Buy & Hold（一開始就進場）
        cooldown_days: 停損後等待 N 天才能重新進場

    Returns:
        df_result: 含 Position, Strategy_Return, Cumulative 等欄位
        metrics: dict 含 Return%, MDD%, Trade_Count, CAGR% 等
    """
    df = df.copy()
    n = len(df)

    # 準備欄位
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)

    # 初始化
    position = np.zeros(n)         # 0 或 1
    entry_price = np.zeros(n)      # 進場價格
    high_watermark = np.zeros(n)   # 最高點（trailing 用）
    stop_loss_triggered = np.zeros(n, dtype=bool)  # 是否觸發停損

    in_position = False
    current_entry_price = 0.0
    current_high = 0.0
    cooldown_counter = 0
    trade_count = 0

    for i in range(n):
        price = df['Close'].iloc[i]

        # 判斷進場訊號
        if entry_signal is not None:
            want_enter = (entry_signal.iloc[i] == 1)
            want_exit = (entry_signal.iloc[i] == -1)
        else:
            # Buy & Hold: 第一天進場，之後維持
            want_enter = (i == 0) or (not in_position and cooldown_counter == 0)
            want_exit = False

        # Cooldown 計數
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # 進場邏輯
        if not in_position and want_enter and cooldown_counter == 0:
            in_position = True
            current_entry_price = price
            current_high = price
            trade_count += 1

        # 更新 high watermark
        if in_position:
            current_high = max(current_high, price)

        # 停損檢查
        triggered = False
        if in_position and stop_loss_type is not None:
            if stop_loss_type == "fixed":
                # 從進場價計算
                pnl = (price - current_entry_price) / current_entry_price
                if pnl <= threshold:
                    triggered = True
            elif stop_loss_type == "trailing":
                # 從最高點計算回撤
                drawdown = (price - current_high) / current_high
                if drawdown <= threshold:
                    triggered = True

        # 出場邏輯
        if in_position and (triggered or want_exit):
            in_position = False
            if triggered:
                cooldown_counter = cooldown_days
            stop_loss_triggered[i] = triggered

        # 記錄狀態
        position[i] = 1 if in_position else 0
        entry_price[i] = current_entry_price if in_position else 0
        high_watermark[i] = current_high if in_position else 0

    # 計算報酬（position 需要 shift 1 避免 look-ahead bias）
    df['Position'] = pd.Series(position, index=df.index).shift(1).fillna(0)
    df['Entry_Price'] = pd.Series(entry_price, index=df.index)
    df['High_Watermark'] = pd.Series(high_watermark, index=df.index)
    df['Stop_Loss_Triggered'] = pd.Series(stop_loss_triggered, index=df.index)

    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    df['Cumulative_Market'] = (1 + df['Daily_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

    # 計算 metrics
    final_return = (df['Cumulative_Strategy'].iloc[-1] - 1) * 100
    market_return = (df['Cumulative_Market'].iloc[-1] - 1) * 100

    # 最大回撤
    cummax = df['Cumulative_Strategy'].cummax()
    drawdown = (df['Cumulative_Strategy'] - cummax) / cummax
    mdd = drawdown.min() * 100

    # 年化報酬（CAGR）
    years = n / 252  # 假設 252 交易日/年
    if df['Cumulative_Strategy'].iloc[-1] > 0 and years > 0:
        cagr = ((df['Cumulative_Strategy'].iloc[-1]) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    # 停損觸發次數
    stop_count = df['Stop_Loss_Triggered'].sum()

    metrics = {
        "Return%": round(final_return, 2),
        "Market%": round(market_return, 2),
        "MDD%": round(mdd, 2),
        "CAGR%": round(cagr, 2),
        "Trade_Count": trade_count,
        "Stop_Count": int(stop_count),
    }

    return df, metrics


def compare_strategies_for_symbol(df, strategies):
    """對單一標的比較多種停損策略

    Args:
        df: DataFrame 含 Close
        strategies: list of dict，每個 dict 含 name, type, threshold

    Returns:
        results: list of dict，每個 dict 含 strategy name + metrics
    """
    results = []

    for s in strategies:
        _, metrics = backtest_stop_loss_strategy(
            df,
            stop_loss_type=s.get("type"),
            threshold=s.get("threshold", -0.35),
        )
        results.append({
            "Strategy": s["name"],
            **metrics,
        })

    return results
