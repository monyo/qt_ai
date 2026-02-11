def check_stop_loss(positions, current_prices, threshold=-0.35):
    """檢查持倉是否觸發極端停損（從成本價計算）

    Args:
        positions: 持倉 dict
        current_prices: {symbol: price}
        threshold: 停損閾值（預設 -35%）

    Returns:
        list of dict: [{"symbol": str, "pnl_pct": float}, ...]
    """
    triggered = []
    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue
        price = current_prices.get(symbol)
        if price is None:
            continue
        pnl_pct = (price - pos["avg_price"]) / pos["avg_price"]
        if pnl_pct <= threshold:
            triggered.append({
                "symbol": symbol,
                "pnl_pct": round(pnl_pct * 100, 2),
                "reason": "extreme_stop",
            })
    return triggered


def check_trailing_stop(positions, current_prices, threshold=-0.15):
    """檢查持倉是否觸發移動停利（從最高價計算）

    Args:
        positions: 持倉 dict（需含 high_since_entry 欄位）
        current_prices: {symbol: price}
        threshold: 回撤閾值（預設 -15%）

    Returns:
        list of dict: [{"symbol": str, "drawdown_pct": float, "high_price": float}, ...]
    """
    triggered = []
    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue
        price = current_prices.get(symbol)
        high_price = pos.get("high_since_entry")
        if price is None or high_price is None:
            continue

        drawdown_pct = (price - high_price) / high_price
        if drawdown_pct <= threshold:
            triggered.append({
                "symbol": symbol,
                "drawdown_pct": round(drawdown_pct * 100, 2),
                "high_price": high_price,
                "current_price": price,
                "reason": "trailing_stop",
            })
    return triggered


def check_ma200_stop(positions, current_prices, ma200_prices):
    """檢查持倉是否跌破 MA200

    Args:
        positions: 持倉 dict
        current_prices: {symbol: price}
        ma200_prices: {symbol: ma200_value}

    Returns:
        list of dict: [{"symbol": str, "ma200": float, "current_price": float}, ...]
    """
    triggered = []
    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue
        price = current_prices.get(symbol)
        ma200 = ma200_prices.get(symbol)
        if price is None or ma200 is None:
            continue

        if price < ma200:
            triggered.append({
                "symbol": symbol,
                "ma200": round(ma200, 2),
                "current_price": price,
                "below_pct": round((price - ma200) / ma200 * 100, 2),
                "reason": "ma200_stop",
            })
    return triggered


def check_all_exit_conditions(positions, current_prices, ma200_prices,
                               trailing_threshold=-0.15, hard_threshold=-0.35):
    """檢查所有出場條件，回傳需要出場的持倉

    優先順序：
    1. 移動停利（-15% from high）
    2. MA200 停損
    3. 極端停損（-35% from cost）

    Returns:
        dict: {symbol: {"reason": str, "details": dict}}
    """
    exits = {}

    # 1. 移動停利
    trailing_stops = check_trailing_stop(positions, current_prices, trailing_threshold)
    for item in trailing_stops:
        exits[item["symbol"]] = {
            "reason": "trailing_stop",
            "message": f"移動停利觸發（從高點 ${item['high_price']:.2f} 回撤 {item['drawdown_pct']:.1f}%）",
            "details": item,
        }

    # 2. MA200 停損（如果還沒被移動停利抓到）
    ma200_stops = check_ma200_stop(positions, current_prices, ma200_prices)
    for item in ma200_stops:
        if item["symbol"] not in exits:
            exits[item["symbol"]] = {
                "reason": "ma200_stop",
                "message": f"跌破 MA200（${item['ma200']:.2f}，目前 {item['below_pct']:.1f}%）",
                "details": item,
            }

    # 3. 極端停損（最後防線）
    hard_stops = check_stop_loss(positions, current_prices, hard_threshold)
    for item in hard_stops:
        if item["symbol"] not in exits:
            exits[item["symbol"]] = {
                "reason": "extreme_stop",
                "message": f"極端停損觸發（從成本 {item['pnl_pct']:.1f}%）",
                "details": item,
            }

    return exits


def check_position_limit(portfolio, max_stocks=30):
    """回傳還能買幾檔個股"""
    individual_count = sum(
        1 for pos in portfolio["positions"].values()
        if not pos.get("core", False)
    )
    return max(max_stocks - individual_count, 0)
