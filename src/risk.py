def check_stop_loss(positions, current_prices, threshold=-0.35):
    """檢查持倉是否觸發硬停損，回傳觸發的 symbol 列表"""
    triggered = []
    for symbol, pos in positions.items():
        if pos.get("core", False):
            continue
        price = current_prices.get(symbol)
        if price is None:
            continue
        pnl_pct = (price - pos["avg_price"]) / pos["avg_price"]
        if pnl_pct <= threshold:
            triggered.append({"symbol": symbol, "pnl_pct": round(pnl_pct * 100, 2)})
    return triggered


def check_position_limit(portfolio, max_stocks=30):
    """回傳還能買幾檔個股"""
    individual_count = sum(
        1 for pos in portfolio["positions"].values()
        if not pos.get("core", False)
    )
    return max(max_stocks - individual_count, 0)
